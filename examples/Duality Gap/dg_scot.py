import torch
import numpy as np
from scipy.optimize import minimize
from ot.smooth import SparsityConstrained
from sparse_ot.utils import conj_lda3, createLogHandler, get_dist, get_G, get_primal
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
parser.add_argument('--max_itr', type=int, default=1000)
parser.add_argument('--save_as')
args = parser.parse_args()

khp = args.khp
ktype = args.ktype
lda = args.lda
lda3 = args.lda3
K = args.K
max_itr = args.max_itr
os.makedirs(args.save_as, exist_ok=True)
logger = createLogHandler(f'{args.save_as}/other_inv_{max_itr}.csv', str(os.getpid()))

def seed_everything(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

x = x.view(x.shape[0], -1)
y = y.view(y.shape[0], -1)

Ct = get_dist(x, y)
Ct = Ct/torch.max(Ct)
m, n = Ct.shape

G1 = get_G(x=x, y=x, khp=khp, ktype=ktype)
G2 = get_G(x=y, y=y, khp=khp, ktype=ktype)
G = {1: G1, 2: G2}

G_inv = {1: torch.linalg.inv(G1), 2: torch.linalg.inv(G2)}

C = Ct.cpu().numpy()
G_inv = {1: G_inv[1].cpu().numpy(), 2: G_inv[2].cpu().numpy()}
a = np.ones(G1.shape[0], dtype=np.float32)/G1.shape[0]
b = np.ones(G2.shape[0], dtype=np.float32)/G2.shape[0]

cn1 = torch.linalg.cond(G1)
cn2 = torch.linalg.cond(G2)

def dual_obj(alpha, beta, a, b, C, G_inv, lmbda, max_nz):
    lda = lmbda[1]
    lda3 = lmbda[3]
    X = alpha[:, None] + beta - C
    obj = alpha.dot(a) + beta.dot(b) - torch.mv(G_inv[1], alpha).dot(alpha)/(4*lda) - \
        torch.mv(G_inv[2], beta).dot(beta)/(4*lda) - conj_lda3(X, max_nz, lda3)
    
    bottom_k = torch.topk(X, X.shape[0]-max_nz, dim=0, largest=False).indices
    X_new = torch.scatter(X, 0, bottom_k, 0.0)/lda3  # X.scatter_(0, bottom_k, 0.0)/lda3
    gamma = torch.clamp(X_new, min=0)
    return obj, gamma

def dual_obj_grad(alpha, beta, a, b, C, G_inv, lmbda, regul=None, max_nz=1):
    if regul is None:
        regul = SparsityConstrained(max_nz, lmbda[3])
    
    obj = np.dot(alpha, a) + np.dot(beta, b) - (G_inv[1]@alpha).dot(alpha)/(4*lmbda[1]) - (G_inv[2]@beta).dot(beta)/(4*lmbda[2])

    grad_alpha = a.copy() - G_inv[1]@alpha/(2*lmbda[1])
    grad_beta = b.copy() - G_inv[2]@beta/(2*lmbda[2])
    
    X = alpha[:, np.newaxis] + beta - C
    
    val, G = regul.delta_Omega(X)
    
    obj -= np.sum(val)
    grad_alpha -= G.sum(axis=1)
    grad_beta -= G.sum(axis=0)
    
    return obj, grad_alpha, grad_beta, G
    

def solve_dual(a, b, C, G_inv, lmbda, max_nz, regul=None, method="L-BFGS-B", tol=1e-3, max_iter=500, verbose=False):
    # regul: Regularization obj. Should implement delta_Omega(X).
    def _func(params):
        alpha = params[:len(a)]
        beta = params[len(a):]
        
        obj, grad_alpha, grad_beta, _ = dual_obj_grad(alpha, beta, a, b, C, G_inv, lmbda, regul, max_nz)
        grad = np.concatenate((grad_alpha, grad_beta))
        return -obj, -grad
    # as minimize allows vector only so concatenating
    alpha_init = np.zeros_like(a)
    beta_init = np.zeros_like(b)
    params_init = np.concatenate((alpha_init, beta_init))
    
    res = minimize(_func, params_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=verbose))
    alpha = res.x[:len(a)]
    beta = res.x[len(a):]
    return alpha, beta, res

alpha, beta, res = solve_dual(a, b, C, G_inv, {1: lda, 2: lda, 3: lda3}, K, max_iter=max_itr)
obj_dual = -res.fun

X = torch.from_numpy(alpha[:, None] + beta - C)
bottom_k = torch.topk(X, X.shape[0]-K, dim=0, largest=False).indices
X.scatter_(0, bottom_k, 0.0)
gamma = torch.clamp(X, min=0).float() # primal solution
a = torch.from_numpy(a).float()
b = torch.from_numpy(b).float()

obj_primal = get_primal(gamma, a, b, Ct, G, lda, lda3)

logger.info(f"{max_itr}, {obj_primal}, {obj_dual}, {obj_primal-obj_dual}, {cn1}, {cn2}, {khp}, {ktype}, {lda}, {lda3}, {K}")
