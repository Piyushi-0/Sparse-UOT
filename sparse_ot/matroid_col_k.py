"""
This code implements the ColSparse UOT algorithm.. 
"""

import torch
import numpy as np
from torch import sqrt
from torch.linalg import norm
from sparse_ot.utils import postprocess_gamma, get_obj


def seed_everything(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_marginals(gamma, S_i, S_j, m, n):
    gamma1 = S_i.bincount(gamma, minlength=m)
    gammaT1 = S_j.bincount(gamma, minlength=n)
    return gamma1, gammaT1

def get_idx(grd, S_i, S_j, rem_k, m, n):
    """
    grd: gradient matrix
    S_i: row indices of the chosen elements
    S_j: column indices of the chosen elements
    rem_k: remaining number of elements to be chosen from each column (initially k for all)
    m: number of rows
    n: number of columns
    
    returns: chosen element from the optimal base 
    """
    new_grd = torch.clamp(grd, min=0)
    new_grd[S_i, S_j] = -torch.inf # to avoid choosing the same element again
    feas_col = torch.where(rem_k)[0] # columns with remaining elements
    sorted_indices = torch.sort(new_grd[:, feas_col], dim=0)[1]
    
    one_at = (m-rem_k[feas_col]).unsqueeze(0) # per-col sorting is in asc so we have to take the bottom m-rem_k[feas_col] elements from each feasible col
    mask = torch.zeros_like(sorted_indices).scatter_(0, one_at, 1).cumsum(0).to(torch.bool)
    
    M_i = torch.masked_select(sorted_indices, mask) # row indices of elements in the optimal base
    M_j = feas_col.repeat(m, 1).masked_select(mask) # column indices of elements in the optimal base
    
    idx = torch.randperm(M_i.shape[0]) # shuffle
    M_i = M_i[idx]
    M_j = M_j[idx]
    
    chosen_elem = 0 # torch.randint(rem_k.sum().item(), (1,)).item()
    chosen_i = M_i[chosen_elem] # row index of the randomly chosen element from the optimal base
    chosen_j = M_j[chosen_elem] # column index of the randomly chosen element from the optimal base
    
    chosen = chosen_i*n + chosen_j # get the index of the chosen element in the original list (to remove from there)
    
    return chosen_i, chosen_j, chosen

def get_gamma(C, G1, G2, v1, v2, max_itr, K, lda, lda3=0, all_gamma=0, vparts=None, verbose=0, grd_crit=0, conv_crit=0, ws=0, seed=0, tol=1e-6):
    m, n = C.shape
    if K is not None:
        assert K <= m*n, "K should be <= to m*n"
    
    if vparts is None:
        vparts = {1: torch.mv(G1, v1).dot(v1), 2: torch.mv(G2, v2).dot(v2)} 
    if ws:
        return get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, grd_crit, conv_crit, seed, tol)
    else:
        return get_gamma_dash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, grd_crit, conv_crit, seed, tol)

def get_gamma_dash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, grd_crit, conv_crit, seed, tol):
    m, n = C.shape
    seed_everything(seed)
    device = C.device
    gammas = []
    
    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)
    
    # NOTE: the following is when gamma0 is 0
    fixed_grd = C - 2*lda*(g1[:, None] + g2)
    
    tot = n*K
    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device) # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device) # to avoid memory leak
    
    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device) # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device) # to avoid recreating in loop
    gamma0 = torch.zeros(tot, dtype=fixed_grd.dtype, device=device)
    
    gammas = []
    gamma = None
    
    rem_k = torch.ones(n, dtype=torch.long, device=device)*K
    
    u_chosen_i, u_chosen_j, chosen = get_idx(-fixed_grd, [], [], rem_k, m, n) # NOTE: - of uot grd
    
    if verbose:
        print(chosen.item(), u_chosen_i.item(), u_chosen_j.item())
    
    if grd_crit and fixed_grd[u_chosen_i, u_chosen_j]>=0:
        return [gamma0[:1]] if all_gamma else gamma0[:1], S_i[:1], S_j[:1]
    
    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        if k:
            # R is V_minus_L
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor') # row indices for elements in R
            S_j_R = V_minus_L % n # column indices for elements in R
    
            unq_i_R = torch.unique(S_i_R) # unique row indices for elements in R
            unq_j_R = torch.unique(S_j_R) # unique column indices for elements in R
            
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)
            
            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]
            
            grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] + c2[idx_in_c2[S_j_R]])
            if lda3:
                grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            grd = postprocess_gamma(grd, S_i_R, S_j_R, m, n)
            
            u_chosen_i, u_chosen_j, chosen = get_idx(-grd, S_i[:k], S_j[:k], rem_k, m, n) # NOTE: -grd
            
            if verbose:
                print(chosen.item(), u_chosen_i.item(), u_chosen_j.item())
            
            if grd_crit and grd[u_chosen_i, u_chosen_j]>=0:
                break
            
        S_i[k] = u_chosen_i
        S_j[k] = u_chosen_j

        S_i_S = S_i[:k+1] # row indices chosen till now which will go to uot computation
        S_j_S = S_j[:k+1] # column indices chosen till now which will go to uot computation

        # delete chosen
        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]
        rem_k[u_chosen_j] -= 1
        
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        
        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L

        # *** solve UOT-MMD ***
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]

        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        fixed_grd_S = fixed_grd[S_i_S, S_j_S]

        y = gamma0[:k+1].clone()
        grd = fixed_grd_S.clone() + lda3*y # for the 1st gd step
        x_old = y.clone()
        t = 1
        if  verbose:
            obj = get_obj(C, G1, G2, v1, v2, y, S_i_S, S_j_S, lda, vparts).item()
            all_obj[k].append(obj)
            print(f"outer {k}, inner 0, {obj}")
        
        for inner in range(max_itr):
            gamma = torch.clamp(y-ss*grd, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y = (t_dash+1)*gamma - t_dash*x_old
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y, S_i_S, S_j_S, m, n)

            r2 = torch.mv(G1_u, y1)
            c2 = torch.mv(G2_u, yT1)
            grd = fixed_grd_S + 2*lda*(r2[idx_i] + c2[idx_j]) + lda3*gamma

            pos_ind = torch.where(gamma>0)[0]
            index_0 = torch.where(gamma==0)[0]
            nrm = norm(grd[pos_ind])
            cond2 = (grd[index_0]>=0).all()
            if conv_crit:
                if len(pos_ind) and nrm < tol:
                    if not len(index_0) or cond2:
                        # print(f"converged in {inner+1} iterations")
                        break

            if verbose:
                obj = get_obj(C, G1, G2, v1, v2, gamma, S_i_S, S_j_S, lda, vparts).item()
                all_obj[k].append(obj)
                all_nrm[k].append(nrm.item())
                print(f"outer {k}, inner {inner+1}, {obj}, {len(pos_ind)}, {nrm}, {len(index_0)}, {cond2}")
        if verbose:
            print(gamma.cpu().numpy(), grd.cpu().numpy(), ss.item())
        if all_gamma:
            gammas.append(gamma)
    if verbose:
        return gammas if all_gamma else gamma, S_i_S, S_j_S, all_obj, all_nrm
    return gammas if all_gamma else gamma, S_i_S, S_j_S

def get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, grd_crit, conv_crit, seed, tol):
    m, n = C.shape
    seed_everything(seed)
    device = C.device
    gammas = []
    
    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)
    
    # NOTE: the following is when gamma0 is 0
    fixed_grd = C - 2*lda*(g1[:, None] + g2)
    
    tot = n*K
    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device) # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device) # to avoid memory leak
    
    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device) # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device) # to avoid recreating in loop
    dummy_zero = torch.tensor([0], dtype=fixed_grd.dtype, device=device)
    
    gammas = []
    gamma = dummy_zero.clone()
    
    rem_k = torch.ones(n, dtype=torch.long, device=device)*K
    
    u_chosen_i, u_chosen_j, chosen = get_idx(-fixed_grd, [], [], rem_k, m, n) # NOTE: - of uot grd
    
    if verbose:
        print(chosen.item(), u_chosen_i.item(), u_chosen_j.item())
    
    if grd_crit:
        raise NotImplementedError
    
    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        if k:
            # R is V_minus_L
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor') # row indices for elements in R
            S_j_R = V_minus_L % n # column indices for elements in R
    
            unq_i_R = torch.unique(S_i_R) # unique row indices for elements in R
            unq_j_R = torch.unique(S_j_R) # unique column indices for elements in R
            
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)
            
            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]
            
            grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] + c2[idx_in_c2[S_j_R]])
            if lda3:
                grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            grd = postprocess_gamma(grd, S_i_R, S_j_R, m, n)
            
            u_chosen_i, u_chosen_j, chosen = get_idx(-grd, S_i[:k], S_j[:k], rem_k, m, n) # NOTE: -grd
            
            if verbose:
                print(chosen.item(), u_chosen_i.item(), u_chosen_j.item())
            
            if grd_crit:
                raise NotImplementedError
            
        S_i[k] = u_chosen_i
        S_j[k] = u_chosen_j

        S_i_S = S_i[:k+1] # row indices chosen till now which will go to uot computation
        S_j_S = S_j[:k+1] # column indices chosen till now which will go to uot computation

        # delete chosen
        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]
        rem_k[u_chosen_j] -= 1
        
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]

        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L

        # *** solve UOT-MMD ***
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]
        
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        fixed_grd_S = fixed_grd[S_i_S, S_j_S]
        
        y = torch.cat([gamma[:k], dummy_zero])
        y1, yT1 = get_marginals(y, S_i_S, S_j_S, m, n)
        r2 = torch.mv(G1_u, y1)
        c2 = torch.mv(G2_u, yT1)
        grd = fixed_grd_S + 2*lda*(r2[idx_i] + c2[idx_j]) + lda3*y
        
        x_old = y.clone()
        t = 1
        if  verbose:
            obj = get_obj(C, G1, G2, v1, v2, y, S_i_S, S_j_S, lda, vparts).item()
            all_obj[k].append(obj)
            print(f"outer {k}, inner 0, {obj}")

        for inner in range(max_itr):
            gamma = torch.clamp(y-ss*grd, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y = (t_dash+1)*gamma - t_dash*x_old
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y, S_i_S, S_j_S, m, n)

            r2 = torch.mv(G1_u, y1)
            c2 = torch.mv(G2_u, yT1)
            grd = fixed_grd_S + 2*lda*(r2[idx_i] + c2[idx_j]) + lda3*gamma
            
            pos_ind = torch.where(gamma>0)[0]
            index_0 = torch.where(gamma==0)[0]
            nrm = norm(grd[pos_ind])
            cond2 = (grd[index_0]>=0).all()
            if conv_crit:
                if len(pos_ind) and nrm < tol:
                    if not len(index_0) or cond2:
                        # print(f"converged in {inner+1} iterations")
                        break

            if verbose:
                obj = get_obj(C, G1, G2, v1, v2, gamma, S_i_S, S_j_S, lda, vparts).item()
                all_obj[k].append(obj)
                all_nrm[k].append(nrm.item())
                print(f"outer {k}, inner {inner+1}, {obj}, {len(pos_ind)}, {nrm}, {len(index_0)}, {cond2}")
        if verbose:
            print(gamma.cpu().numpy(), grd.cpu().numpy(), ss.item())
        if all_gamma:
            gammas.append(gamma)
    if verbose:
        return gammas if all_gamma else gamma, S_i_S, S_j_S, all_obj, all_nrm
    return gammas if all_gamma else gamma, S_i_S, S_j_S
