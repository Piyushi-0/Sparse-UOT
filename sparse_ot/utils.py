"""
Some common utility functions.
"""

import torch
import logging
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def get_obj(C, G1, G2, v1, v2, gamma, S_i, S_j, lda, vparts=None):    
    m, n = C.shape
    gamma1 = S_i.bincount(gamma, minlength=m)
    gammaT1 = S_j.bincount(gamma, minlength=n)
    
    if vparts is None:
        reg1 = torch.mv(G1, v1).dot(v1) + torch.dot(torch.mv(G1, gamma1-2*v1), gamma1)
        reg2 = torch.mv(G2, v2).dot(v2) + torch.dot(torch.mv(G2, gammaT1-2*v2), gammaT1)
    else:
        reg1 = vparts[1] + torch.dot(torch.mv(G1, gamma1-2*v1), gamma1)
        reg2 = vparts[2] + torch.dot(torch.mv(G2, gammaT1-2*v2), gammaT1)

    obj = torch.dot(gamma, C[S_i, S_j]) + lda*(reg1+reg2)
    return obj

def get_topk_plan(map_tensor, k):
    indx = torch.topk(map_tensor.flatten(), k)[1]
    topk_plan = np.zeros(map_tensor.numel())
    try:
        topk_plan[indx] = map_tensor.flatten()[indx]
    except:
        topk_plan[indx] = map_tensor.cpu().flatten()[indx]
    return topk_plan.reshape(map_tensor.shape)


def plot(arr):
    plt.clf()
    plt.plot(arr)
    plt.show()


def del_files(path):
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(e)
        os.remove(path)


def createLogHandler(log_file, job_name="_"):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def eye_like(G):
    if(len(G.shape) == 3):
        return torch.eye(*G.shape[-2:], out=torch.empty_like(G)).repeat(G.shape[0],1,1)
    else: 
        return torch.eye(*G.size(),out=torch.empty_like(G))


def get_dist(x, y, p=2, dtype="euc", khp=None):
    if p not in [1, 2]:
        raise NotImplementedError
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(1) if y.dim() == 1 else y

    C = torch.cdist(x, y)

    if p == 2 or "ker" in dtype:
        C = C**2
        if "rbf" in dtype:
            C = 2-2*get_G(dist=C, ktype="rbf", khp=khp, x=x, y=y)
        if "imq" in dtype:
            C = 2/khp**(0.5)-2*get_G(dist=C, ktype="imq", khp=khp, x=x, y=y)
    if "ker" in dtype and p == 1:
        C = C**(0.5)
    return C


def get_G(dist=None, ktype="rbf", khp=None, x=None, y=None, ridge=1e-10):
    """
    # NOTE: if dist is not None, it should be cost matrix**2. 
    If it is None, the function automatically computes euclidean**2.
    """
    if ktype != "lin":
        if khp == None or khp == -1:  # take median heuristic
            khp = 0.5*torch.median(get_dist(x, y, 1).view(-1))
        if dist is None:
            dist = get_dist(x, y)
    if ktype == "lin":
        if x.dim() == 2:
            G = torch.einsum('md,nd->mn', x, y)
        else:
            G = torch.einsum('bmd,nd->bmn', x, y)    
    elif ktype == "rbf":
        G = torch.exp(-dist/(2*khp))
    elif ktype == "imq":
        G = (khp + dist)**(-0.5)
    elif ktype == "imq_v2":
        G = ((1+dist)/khp)**(-0.5)
    if(len(G.shape)==2):
        if G.shape[0] == G.shape[1]:
            G = (G + G.T)/2
    elif(G.shape[1] == G.shape[2]):
        G = (G + G.permute(0,2,1))/2
    G = G + ridge*eye_like(G)
    return G


def offd_dist(M, minn):
    """to get off diagonals in distance matrix

    Args:
        M (FloatTensor): distance matrix
        minn (int): to display upto minn smallest off diagonals

    Returns:
        dictionary: with stats of off diagonals
    """
    import heapq

    def next_nsmallest(numbers, n):
        nsmallest = {}
        for i in range(1, n):
            nsmallest[i] = heapq.nsmallest(i+1, numbers)[-1]
        return nsmallest

    ofdM = M[~torch.eye(M.shape[0], dtype=bool)]
    min_dist = torch.min(ofdM)
    max_dist = torch.max(ofdM)
    med_dist = torch.median(ofdM)
    minn_dist = next_nsmallest(torch.unique(ofdM).cpu().numpy(), minn)

    return {"min_dist": min_dist, "max_dist": max_dist, "med_dist": med_dist,
            "minn_dist": minn_dist}


def seed(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def postprocess_gamma(gamma, S_i, S_j, m, n):
    return torch.sparse_coo_tensor(torch.vstack([S_i, S_j]), gamma,
                                   size=(m, n)).to_dense()

def get_dualsol_lda3(a, b, G, lda1, lda2=None, gamma1=None, gammaT1=None, sol_primal=None, S_i=None, S_j=None):
    lda2 = lda1 if lda2 is None else lda2
    if gamma1 is None:
        m, n = a.shape[0], b.shape[0]
        gamma1 = S_i.bincount(sol_primal, minlength=m)
        gammaT1 = S_j.bincount(sol_primal, minlength=n)
    
    alpha = 2*lda1*G[1]@(a-gamma1)
    beta = 2*lda2*G[2]@(b-gammaT1)
    return {"alpha": alpha, "beta": beta}

def conj_lda3(X, max_nz, lda3):
    bottom_k = torch.topk(X, X.shape[0]-max_nz, dim=0, largest=False).indices
    X.scatter_(0, bottom_k, 0.0)
    max_X = torch.clamp(X, min=0)
    val = torch.linalg.norm(max_X)**2 / (2 * lda3)
    return val

def get_dualobj_lda3(sol_dual_our, a, b, C, G, lda, lda3, max_nz, gamma1=None, gammaT1=None, sol_primal=None, S_i=None, S_j=None):
    if gamma1 is None:
        m, n = a.shape[0], b.shape[0]
        gamma1 = S_i.bincount(sol_primal, minlength=m)
        gammaT1 = S_j.bincount(sol_primal, minlength=n)
    X = sol_dual_our["alpha"][:, None] + sol_dual_our["beta"] - C
    # obj = lda*(torch.mv(G[1], a+gamma1).dot(a-gamma1) + torch.mv(G[2], b+gammaT1).dot(b-gammaT1)) - \
    #       conj_lda3(X, max_nz, lda3)
    # simplifying this to reduce numerical errors.
    obj = lda*(torch.mv(G[1], a).dot(a) - torch.mv(G[1], gamma1).dot(gamma1) +
               torch.mv(G[2], b).dot(b) - torch.mv(G[2], gammaT1).dot(gammaT1)) - \
               conj_lda3(X, max_nz, lda3)

def get_primal(gamma, a, b, C, G, lda, lda3):
    gamma1, gammaT1 = gamma.sum(dim=1), gamma.sum(dim=0)
    vec1 = gamma1-a
    vec2 = gammaT1-b
    obj = torch.tensordot(gamma, C) + lda*(torch.mv(G[1], vec1).dot(vec1) + torch.mv(G[2], vec2).dot(vec2)) + lda3/2*torch.linalg.norm(gamma)**2
    return obj
