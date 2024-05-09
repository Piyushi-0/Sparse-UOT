import matplotlib.pyplot as plt
import torch
from sparse_ot.sparse_repr_autok import get_gamma
from sparse_ot.utils import get_dist, get_G, get_obj
import seaborn as sns
import argparse
import os
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description = "_")
parser.add_argument('--lda', type=float, default=0.1)
parser.add_argument('--ktype', type=str, default="imq")
parser.add_argument('--khp', type=float, default=1e-4)
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--use_cuda', default=1)
args = parser.parse_args()
sns.set(color_codes=True)

lda = args.lda
ktype = args.ktype
khp = args.khp
device = torch.device(f"cuda:{args.gpu_id}")
lr_x = 0.01
Nsteps = 2450
max_itr = 100
method = "SUOT"

save_in = f"{method}/{max_itr}/{lr_x}/{Nsteps}/{lda}/{ktype}/{khp}/"
os.makedirs(save_in, exist_ok=True)

use_cuda = args.use_cuda
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(dtype)

N, M = (1000, 1000)

X_i = draw_samples("../../data/gradflow/density_a.png", N).float().to(device)
y_j = draw_samples("../../data/gradflow/density_b.png", M).float().to(device)
a_i = (torch.ones(N) / N).float().to(device)
b_j = (torch.ones(M) / M).float().to(device)

G2 = get_G(x=y_j, y=y_j, khp=khp, ktype=ktype)

x_i = X_i.clone()
colors = ((10 * x_i[:, 0]).cos() * (10 * x_i[:, 1]).cos()).detach().cpu().numpy()
x_i.requires_grad = True

plt.figure(figsize=(12, 8))

for i in tqdm(range(Nsteps)):
    C = get_dist(x_i, y_j)
    G1 = get_G(x=x_i, y=x_i, khp=khp, ktype=ktype)
    
    with torch.no_grad():
        pi, S_i, S_j = get_gamma(C, G1, G2, a_i, b_j, max_itr, None, lda, 0, ws=1, conv_crit=1)
    obj = get_obj(C, G1, G2, a_i, b_j, pi, S_i, S_j, lda)
    
    if x_i.grad is not None:
        x_i.grad.data.zero_()
    [g] = torch.autograd.grad(obj, [x_i])
    
    x_i.data -= lr_x*len(x_i)* g
    
    fig, ax = plt.subplots()
    plt.set_cmap('hsv')
    # plt.scatter([10], [10])
    
    display_samples(ax, y_j, [(0.55, 0.55, 0.95)])
    display_samples(ax, x_i, colors)
    
    plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    if i % 50 == 0 or i == Nsteps - 1:
        plt.savefig(f'{save_in}/{i}.jpg', bbox_inches = 'tight', pad_inches = 0.25)
    plt.close(fig)
    plt.show()
torch.save(x_i, f"{save_in}/x_i.pt")
