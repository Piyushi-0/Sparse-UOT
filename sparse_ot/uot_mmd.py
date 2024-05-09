"""
This code is for solving MMD-regularized UOT using APGD (Accelerated
Projected Gradient Descent).
"""

# TODO: for different support, re-write with MMD combined.

import torch
from tqdm import tqdm
from torch import sqrt
from torch.linalg import norm
import numpy as np


def sq_mnorm(vec, G):
    return torch.dot(torch.mv(G, vec), vec)


def get_marginals(alpha):
    alpha1 = torch.sum(alpha, dim=1)
    alphaT1 = torch.sum(alpha, dim=0)
    return alpha1, alphaT1


def get_G_parts(m, G):
    G_1 = G[:m, :m]
    G_2 = G[m:, m:]
    G_12 = G[:, m:]
    G_21 = G[:, :m]
    return G_1, G_2, G_12, G_21


def get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp=1, test=False):
    """
    to get MMD^2 regularization terms
    Args:
        alpha1 (FloatTensor): first marginal of alpha
        alphaT1 (FloatTensor): second marginal of alpha
        v (dictionary of FloatTensors): keys 1 and 2 for the two distributions
        G ((dictionary of )FloatTensor(s)): keys 1 and 2 for Gram matrix over
                    the supports of the distributions. If different supports,
                    G is gram matrix over the union of supports.
        same_supp (bool, optional): If supports match or not. Defaults to 1.
        test (bool, optional): If to test or not. Defaults to False.
    Returns:
        FloatTensors (reg_1, reg_2): regularization terms
    """
    reg_1 = sq_mnorm(alpha1-v[1], G[1])
    reg_2 = sq_mnorm(alphaT1-v[2], G[2])
    if not same_supp:
        m = v[1].shape[0]
        G_1, G_2, G_12, G_21 = get_G_parts(m, G)
        reg_1 = sq_mnorm(alpha1, G) + \
            sq_mnorm(v[1], G_1)-2*torch.mv(G_21, v[1]).dot(alpha1)
        reg_2 = sq_mnorm(alphaT1, G) + \
            sq_mnorm(v[2], G_2)-2*torch.mv(G_12, v[2]).dot(alphaT1)
    if test:
        assert reg_1 >= 0 and reg_2 >= 0, "reg_1 and reg_2 should be non-neg"
    return reg_1, reg_2


def get_obj(C, G, lda, v, alpha, same_supp=1, verbose=0):
    """to get objective value

    Args:
        C (FloatTensor): cost matrix
        G ((dictionary of )FloatTensor(s)): keys 1 and 2 for Gram matrix over
                    the supports of the distributions. If different supports,
                    G is gram matrix over the union of supports.
        lda (float): regularization coefficient
        v (dictionary of FloatTensors): keys 1 and 2 for the two distributions
        alpha (FloatTensor): transport coupling
        same_supp (bool, optional): If supports match or not. Defaults to 1.
        verbose (int, optional): If to display objective parts. Defaults to 0.
    Returns:
        FloatTensor scalar (obj): objective
    """
    alpha1, alphaT1 = get_marginals(alpha)
    reg_1, reg_2 = get_mmdsq_reg(alpha1, alphaT1, v, G, same_supp)
    E_c = torch.tensordot(alpha, C)
    obj = E_c + lda*(reg_1+reg_2)
    if verbose:
        print("E_c: {}, reg_1: {}, reg_2: {}".format(E_c, reg_1, reg_2))
    return obj


def get_grd(C, G, lda, v, alpha, same_supp=1, verbose=0):
    """to get gradient of objective

    Args:
        C (FloatTensor): cost matrix
        G ((dictionary of )FloatTensor(s)): keys 1 and 2 for Gram matrix over
                    the supports of the distributions. If different supports,
                    G is gram matrix over the union of supports.
        lda (float): regularization coefficient
        v (dictionary of FloatTensors): keys 1 and 2 for the two distributions
        alpha (FloatTensor): transport coupling
        same_supp (bool, optional): If supports match or not. Defaults to 1.
        verbose (int, optional): If to display objective parts. Defaults to 0.
    Returns:
        FloatTensor: gradient wrt alpha
    """
    alpha1, alphaT1 = get_marginals(alpha)
    if same_supp:
        grd_1 = torch.mv(G[1], alpha1-v[1])[:, None]
        grd_2 = torch.mv(G[2], alphaT1-v[2])
    else:
        m = v[1].shape[0]
        _, _, G_12, G_21 = get_G_parts(m, G)
        grd_1 = (torch.mv(G, alpha1)-torch.mv(G_21, v[1]))[:, None]
        grd_2 = torch.mv(G, alphaT1)-torch.mv(G_12, v[2])
    grd = C+2*lda*(grd_1+grd_2)
    if verbose:
        print("grd_norm: {}".format(torch.linalg.norm(grd)))
    return grd


def test_conv(obj_itr):
    """test convergence based on abs relative obj decrease

    Args:
        obj_itr (list): objective over iterations

    Returns:
        bool : if converged
    """
    cur_obj = obj_itr[-1]
    prv_obj = obj_itr[-2]
    rel_dec = abs(prv_obj-cur_obj)/(abs(prv_obj)+1e-10)
    if rel_dec < 1e-6:
        return 1
    return 0


def solve_apgd(C, G1, G2, v1, v2, max_itr, lda, same_supp=1, disable_tqdm=1,
               verbose=0, prog=0, conv_crit=1, tol=1e-6):
    """solve via accelerated projected gd

    Args:
        C (_array_like_): cost matrix between source and target.
        G1 (_array_like_): Gram matrix with samples from source.
        G2 (_array_like_): Gram matrix with samples from target.
        v1 (_vector_): source distribution over samples.
        v2 (_vector_): target distribution over samples.
        max_itr (_int_): for APGD.
        lda (_float_): lambda regularization hyperparameter.
        same_supp (int, optional): If supports match or not. Defaults to 1.
        disable_tqdm (int, optional): to show progress bar. Defaults to 1.
        verbose (int, optional): to return obj over iter or not. Defaults to 0.
        prog(int, optional): to return progress i.e. obj over iter or not. Defaults to 0.

    Returns:
        x_i (FloatTensor): OT plan
        obj_itr (list): objective over iterations, returned if verbose is 1.
    """
    G = {1: G1, 2: G2}
    v = {1: v1, 2: v2}

    y = torch.zeros_like(C)
    x_old = y
    t = 1

    m, n = C.shape
    ss = 1/(2*lda*(sqrt(n**2*norm(G[1])**2 + m**2
                        * norm(G[2])**2 + 2*(torch.sum(G[1])*
                                             torch.sum(G[2])))))
    obj_itr = []
    x_i = y.clone()
    for iteration in tqdm(range(1, max_itr+1), disable=disable_tqdm):
        grd = get_grd(C, G, lda, v, y, same_supp, verbose)

        pos_ind = torch.where(x_i>0)[0]
        index_0 = torch.where(x_i==0)[0]
        nrm = norm(grd[pos_ind])
        cond2 = (grd[index_0]>=0).all()
        if conv_crit:
            if len(pos_ind) and nrm < tol:
                if not len(index_0) or cond2:
                    print(f"converged in {iteration+1} iterations")
                    break

        # update x_i
        x_i = torch.clamp(y-ss*grd, min=0)  # projection descent

        # check convergence based on relative obj decrease
        obj_itr.append(get_obj(C, G, lda, v, x_i, same_supp, verbose).item())
        if iteration > 1 and test_conv(obj_itr):
            break

        # update y:
        if iteration < max_itr:
            t_new = (1+np.sqrt(1+4*t**2))/2
            y = x_i + (t-1)*(x_i-x_old)/t_new
            x_old = x_i.clone()
            t = t_new
    if prog:
        return x_i, obj_itr
    return x_i
