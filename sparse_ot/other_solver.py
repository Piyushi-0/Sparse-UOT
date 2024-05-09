import numpy as np
from scipy.optimize import minimize
from ot.smooth import SparsityConstrained
from sparse_ot.utils import conj_lda3
import torch

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
