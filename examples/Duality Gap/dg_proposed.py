import torch
import numpy as np
from torch.linalg import norm
from sparse_ot.utils import postprocess_gamma, createLogHandler, get_dist, get_G
from sparse_ot.matroid_col_k import get_gamma
from sparse_ot.utils import conj_lda3, get_dualobj_lda3, get_primal, get_dualsol_lda3
import os
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Duality Gap')
parser.add_argument('--khp', type=float, default=1.0, help='kernel hyperparameter')
parser.add_argument('--ktype', type=str, default='rbf', help='kernel type')
parser.add_argument('--lda', type=float, default=1.0, help='regularization parameter')
parser.add_argument('--lda3', type=float, default=1.0, help='quad regularization parameter')
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--save_as')
args = parser.parse_args()

khp = args.khp
ktype = args.ktype
lda = args.lda
lda3 = args.lda3
K = args.K
device = torch.device("cuda")
os.makedirs(args.save_as, exist_ok=True)
logger = createLogHandler(f'{args.save_as}/our_inv.csv', str(os.getpid()))

def seed_everything(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dual_obj(alpha, beta, a, b, C, G_inv, lmbda, max_nz):
    lda = lmbda[1]
    lda3 = lmbda[3]
    X = alpha[:, None] + beta - C
    obj = alpha.dot(a) + beta.dot(b) - torch.mv(G_inv[1], alpha).dot(alpha)/(4*lda) - \
        torch.mv(G_inv[2], beta).dot(beta)/(4*lda) - conj_lda3(X, max_nz, lda3)
    return obj

seed_everything(0)

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../../../../data/cifar10/', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                        shuffle=False)

for i, (x_i, _) in enumerate(trainloader):
    if i == 0:
        x = x_i
    elif i == 1:
        y = x_i
        break

x = x.view(x.shape[0], -1).to(device)
y = y.view(y.shape[0], -1).to(device)

C = get_dist(x, y)
C = C/torch.max(C)
m, n = C.shape

G1 = get_G(x=x, y=x, khp=khp, ktype=ktype)
G2 = get_G(x=y, y=y, khp=khp, ktype=ktype)
G = {1: G1, 2: G2}
G_inv = {1: torch.linalg.inv(G1), 2: torch.linalg.inv(G2)}

cn1 = torch.linalg.cond(G1)
cn2 = torch.linalg.cond(G2)

v1 = torch.ones(G1.shape[0], device=C.device)/G1.shape[0]
v2 = torch.ones(G2.shape[0], device=C.device)/G2.shape[0]
max_itr = 1000

gamma, S_i_S, S_j_S = get_gamma(C, G1, G2, v1, v2, max_itr, K, lda, lda3)
sol_primal = postprocess_gamma(gamma, S_i_S, S_j_S, m, n)
gamma1 = sol_primal.sum(1)
gammaT1 = sol_primal.sum(0)
obj_primal = get_primal(sol_primal, v1, v2, C, G, lda, lda3)

sol_dual_our = get_dualsol_lda3(v1, v2, G, lda, gamma1=gamma1, gammaT1=gammaT1)
obj_dual = dual_obj(sol_dual_our["alpha"], sol_dual_our["beta"], v1, v2, C, G_inv, {1: lda, 3: lda3}, K)  # get_dualobj_lda3(sol_dual_our, v1, v2, C, G, lda, lda3, K, gamma1=gamma1, gammaT1=gammaT1)

logger.info(f"{obj_primal}, {obj_dual}, {obj_primal-obj_dual}, {cn1}, {cn2}, {khp}, {ktype}, {lda}, {lda3}, {K}")
