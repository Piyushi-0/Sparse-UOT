import torch
from torch import nn
import torch.nn.functional as F
from algos import get_indices
from utils import *

# NOTE: Operating with batch-size 1 only.

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MIN_EXPERT_CAPACITY = 4

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class Experts(nn.Module):
    def __init__(self,
        dim,
        dim_out,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim_out)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        method='scot',
        **kwargs):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.method = method
        self.kwargs = kwargs
    
    def forward(self, x):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval
        
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating).view(x.shape[-2], self.num_gates)
        raw_gates = raw_gates.softmax(dim=-1)

        expert_capacity = max(min(group_size, int((group_size * capacity_factor) / num_gates)), MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        kwargs = self.kwargs
        if self.method == "prp" or self.method == "uotmmd":
            kwargs['data'] = x
        indices = get_indices(self.method, raw_gates, expert_capacity_f, self.kwargs)    
        
        index_1, index_2 = indices[:, 0], indices[:, 1]
        
        gate_1 = raw_gates.gather(1, index_1.view(-1, 1)).squeeze(-1)
        mask_1 = F.one_hot(index_1, num_gates).float()
        
        gate_2 = raw_gates.gather(1, index_2.view(-1, 1)).squeeze(-1)
        mask_2 = F.one_hot(index_2, num_gates).float()
        
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom
        
        # load-balancing
        density_1 = mask_1.mean(dim=-2)
        density_1_proxy = raw_gates.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean()*float(num_gates**2)

        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float() 
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)
        mask_2_count = mask_2.sum(dim=-2, keepdim=True)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor.unsqueeze(0), combine_tensor.unsqueeze(0), loss, mask_1_count, mask_2_count

class MoE(nn.Module):
    def __init__(self,
        dim,
        dim_out,
        args,
        G_e,
        hidden_dim = None,
        activation = nn.ReLU,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None,
        method='scot',
        logger=None):
        super().__init__()
        
        self.num_experts = args.num_experts

        gating_kwargs = {'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval, 'method': method,\
                        'lda3': args.lda3, 'max_itr': args.max_itr, 'lda': args.lda, 'ws': args.ws, 'G_e': G_e, \
                        'ktype': args.ktype, 'khp': args.khp, 'K': 2, 'logger': logger}
        
        self.gate = Top2Gating(dim, num_gates = self.num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, dim_out, num_experts = self.num_experts, hidden_dim = hidden_dim, activation = activation))
        # self.experts.apply(initialize_weights)
        self.loss_coef = loss_coef

    def forward(self, inputs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss, cnt1, cnt2 = self.gate(inputs)
        
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.unsqueeze(1)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef, cnt1, cnt2
