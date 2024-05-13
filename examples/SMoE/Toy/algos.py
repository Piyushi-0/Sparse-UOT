import ot
import torch
import numpy as np
from sparse_ot.utils import get_G, postprocess_gamma
from sparse_ot.uot_mmd import solve_apgd
from sparse_ot.matroid_col_k import get_gamma
from ot.smooth import SparsityConstrained

def get_indices(method, gates, exp_cap, kwargs):
    if "scot" in method:
        return get_indices_scot(gates, exp_cap, kwargs)
    elif "prp" in method:
        return get_indices_prp(gates, exp_cap, kwargs)
    elif "ot" in method and "ssot" not in method:
        return get_indices_ot(gates, exp_cap, kwargs)
    elif "uotkl" in method:
        return get_indices_uotkl(gates, exp_cap, kwargs)
    elif "uotmmd" in method:
        return get_indices_uotmmd(gates, exp_cap, kwargs)
    elif "ssot" in method:
        return get_indices_ssot(gates, exp_cap, kwargs)

# def get_top_per_row(pi):
#     indices = torch.nonzero(pi.T, as_tuple=True)
#     return torch.split(indices[1], tuple(torch.bincount(indices[0])))

def get_indices_uotmmd(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    dtype = gates.dtype
    C = -gates.T.detach()
    C = C/C.max()
    m, n = C.shape
    mu = torch.full((m,), exp_cap, device=device, dtype=dtype)
    nu = torch.ones((n,), device=device, dtype=dtype)
    x = kwargs["data"].view(n, -1)
    if kwargs["ktype"] == 'I':
        G = torch.eye(n, device=device, dtype=dtype)
    else:
        G = get_G(x=x, y=x, ktype=kwargs["ktype"], khp=kwargs["khp"])
    pi = solve_apgd(C, kwargs["G_e"], G, mu, nu, kwargs["max_itr"], kwargs["lda"]).T
    return pi.topk(K, dim=1)[1]

def get_indices_uotkl(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    dtype = gates.dtype
    C = -gates.T.detach()
    C = C/C.max()
    m, n = C.shape
    mu = torch.full((m,), exp_cap, device=device, dtype=dtype)
    nu = torch.ones((n,), device=device, dtype=dtype)
    eps = kwargs["lda"]
    lda = kwargs["lda3"]
    pi = ot.unbalanced.sinkhorn_unbalanced(mu, nu, C, eps, lda, 'sinkhorn_stabilized').T
    return pi.topk(K, dim=1)[1]

def get_indices_ot(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    dtype = gates.dtype
    C = -gates.T.detach()
    C = C/C.max()
    m, n = C.shape
    mu = torch.full((m,), exp_cap, device=device, dtype=dtype)
    nu = torch.ones((n,), device=device, dtype=dtype)
    eps = kwargs["lda3"]
    pi = ot.bregman.sinkhorn(mu, nu, C, eps, 'sinkhorn_stabilized').T
    return pi.topk(K, dim=1)[1]

def get_indices_prp(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    dtype = gates.dtype
    C = -gates.T.detach()
    C = C/C.max()
    m, n = C.shape
    mu = torch.full((m,), exp_cap, device=device, dtype=dtype)
    nu = torch.ones((n,), device=device, dtype=dtype)

    x = kwargs["data"].view(n, -1)
    if kwargs["ktype"] == 'I':
        G = torch.eye(n, device=device, dtype=dtype)
    else:
        G = get_G(x=x, y=x, ktype=kwargs["ktype"], khp=kwargs["khp"])
    pi, S_i, S_j = get_gamma(C, kwargs["G_e"], G, mu, nu, max_itr=kwargs["max_itr"], lda=kwargs["lda"],\
                                K=K, ws=kwargs["ws"], lda3=kwargs["lda3"], conv_crit=1)
    pi = postprocess_gamma(pi, S_i, S_j, m, n).T

    return pi.topk(K, dim=1)[1]

def get_indices_scot(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    lda3 = kwargs["lda3"]
    C = -gates.T.detach().cpu().numpy()
    C = C/C.max()
    m, n = C.shape
    mu = np.full((m,), exp_cap)
    nu = np.ones((n,))
    
    regul = SparsityConstrained(gamma=lda3, max_nz=K)
    alpha, beta, res = ot.smooth.solve_dual(mu, nu, C, regul, max_iter=kwargs["max_itr"])
    if not res.success:
        kwargs["logger"].info(f"WARNING: {res.success}, {lda3}")
    pi = (torch.from_numpy(ot.smooth.get_plan_from_dual(alpha, beta, C, regul)).to(device)).T
    
    return pi.topk(K, dim=1)[1]

def get_indices_ssot(gates, exp_cap, kwargs):
    K = kwargs["K"]
    device = gates.device
    lda3 = kwargs["lda3"]
    C = -gates.T.detach().cpu().numpy()
    C = C/C.max()
    m, n = C.shape
    mu = np.full((m,), exp_cap)
    nu = np.ones((n,))
    pi = (torch.from_numpy(ot.smooth.smooth_ot_dual(mu, nu, C, lda3, reg_type='l2', numItermax=kwargs["max_itr"])).T).to(device)
    
    return pi.topk(K, dim=1)[1]
