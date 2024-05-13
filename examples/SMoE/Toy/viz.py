from smoe import *
from utils import *
from sklearn.model_selection import train_test_split
from utils import GELU_
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

SEED = 0
alpha = 1
num_experts = 3

device = torch.device("cuda")
dtype = torch.float

G_e = torch.eye(num_experts, device=device, dtype=dtype)

num_pts = 220
data_all, label_all = generate_data(num_pts)
data_all = data_all/data_all.max()

data_train_all, data_test, label_train_all, label_test = train_test_split(data_all, label_all, random_state=SEED, stratify=label_all, train_size=120)
data, data_val, label, label_val = train_test_split(data_train_all, label_train_all, random_state=SEED, stratify=label_train_all, train_size=100)

data= data.to(device)
data_val = data_val.to(device)
data_test = data_test.to(device)
label = label.to(device)
label_val = label_val.to(device)
label_test = label_test.to(device)

n, d = data.shape
c = torch.unique(label).shape[0]

def get_xy(exp_c0, exp_c1):
    x = []
    for i in exp_c0:
        x.append(i)
    for i in exp_c1:
        x.append(i)
    x = np.stack(x)
    y = np.hstack([np.zeros(len(exp_c0)), np.ones(len(exp_c1))])
    return x, y.astype(np.int64)

def get_ops(moe):
    with torch.no_grad():
        ops = {0: {0: [], 1: []}, 1: {0: [], 1: []}, 2: {0: [], 1: []}}
        for i, x_i in enumerate(data_test):
            x_i = x_i.unsqueeze(0)
            outputs = torch.einsum('...nd,...dh->...nh', x_i, moe.experts.w1)
            ops[0][label_test[i].item()].append(outputs[0][0].cpu().numpy())
            ops[1][label_test[i].item()].append(outputs[1][0].cpu().numpy())
            ops[2][label_test[i].item()].append(outputs[2][0].cpu().numpy())
    return ops

def plot(ops, method):
    title = {'prp': 'Proposed', 'scot': 'SCOT', 'init_moe': 'MoE (NeurIPS\'22)', 'topk': 'MoE (ICLR\'17)', 'uotmmd': 'MMD-UOT',
              'ot': 'OT', 'uotkl': 'KL-UOT'}[method]
    markers = ['o', 's', '^']
    plt.figure(figsize=(6, 6))
    for e in range(num_experts):
        x, y = get_xy(ops[e][0], ops[e][1])

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000, random_state=SEED)
        x_tsne = tsne.fit_transform(x)
        x_0 = x_tsne[y==0]
        x_1 = x_tsne[y==1]
        plt.scatter(x_0[:, 0], x_0[:, 1], marker=markers[e], s=50, label=f'Expert{e}_C0', c='g', alpha=0.4)
        plt.scatter(x_1[:, 0], x_1[:, 1], marker=markers[e], s=50, label=f'Expert{e}_C1', c='b', alpha=0.4)
        plt.xticks(fontsize=14, weight='bold')
        plt.yticks(fontsize=14, weight='bold')
        #plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, s=75, marker=markers[e], alpha=0.4, label=f'Expert_{e}')
    if method == "prp":
        leg = plt.legend(loc='lower right',prop={'size': 10})
        LH = leg.legendHandles
        LH[0].set_color('g')
        LH[1].set_color('b')
        LH[2].set_color('g')
        LH[3].set_color('b')
        LH[4].set_color('g')
        LH[5].set_color('b')
    #plt.title(f'{title}')
    plt.savefig(f'{title}.jpg', bbox_inches = 'tight', pad_inches = 0.05)
    plt.show()
    
method = "topk"
path = "logs/xor220_topk_100_1_3.pt"

class arg:
    lda3 = 10.
    max_itr = 100
    ws = 1
    ktype = "imq_v2"
    khp = 100.
    lda = 100.
    num_experts = 3

args = arg()

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

moe = MoE(
    dim = d,
    dim_out=c,
    args = args,
    G_e = G_e,
    hidden_dim = d * 4,
    activation = nn.GELU if hasattr(nn, 'GELU') else GELU_,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = alpha,              # multiplier on the auxiliary expert balancing auxiliary loss
    method = method,
    logger=None
)
moe.to(device)

moe.load_state_dict(torch.load(path))
moe.eval()

ops = get_ops(moe)
plot(ops, method)

method = "scot"
path = "logs/xor220_scot_100_1_3_10.0_100.pt"

class arg:
    lda3 = 10.
    max_itr = 100
    ws = 1
    ktype = "imq_v2"
    khp = 100.
    lda = 100.
    num_experts = 3

args = arg()

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

moe = MoE(
    dim = d,
    dim_out=c,
    args = args,
    G_e = G_e,
    hidden_dim = d * 4,
    activation = nn.GELU if hasattr(nn, 'GELU') else GELU_,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = alpha,              # multiplier on the auxiliary expert balancing auxiliary loss
    method = method,
    logger=None
)
moe.to(device)

moe.load_state_dict(torch.load(path))
moe.eval()

ops = get_ops(moe)
plot(ops, method)

method = "prp"
path = "logs/xor220_prp_100_1_3_10.0_100_100.0_100.0_imq_v2_1.pt"

class arg:
    lda3 = 10.
    max_itr = 100
    ws = 1
    ktype = "imq_v2"
    khp = 100.
    lda = 100.
    num_experts = 3

args = arg()

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

moe = MoE(
    dim = d,
    dim_out=c,
    args = args,
    G_e = G_e,
    hidden_dim = d * 4,
    activation = nn.GELU if hasattr(nn, 'GELU') else GELU_,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = alpha,              # multiplier on the auxiliary expert balancing auxiliary loss
    method = method,
    logger=None
)
moe.to(device)

moe.load_state_dict(torch.load(path))
moe.eval()

ops = get_ops(moe)
plot(ops, method)
