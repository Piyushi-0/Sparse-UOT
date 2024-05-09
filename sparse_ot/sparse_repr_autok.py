"""
This code implements the stochastic K-sparse-OT-dash algorithm with
sparse representations ie. gamma and its gradient are of size |S|
where S is the tensor containing support points. 

Here K is auto-chosen.

- get_gamma: Returns the sparse OT plan gamma of dimension K x 1.

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


def get_gamma(C, G1, G2, v1, v2, max_itr, K, lda, lda3=0, all_gamma=0, vparts=None, verbose=0, conv_crit=0, s=None, ws=0, seed=0, tol=1e-6):
    """Computes sparse gamma using K-sparse-OT-dash algorithm.

    Args:
        C (_array_like_): cost matrix between source and target.
        G1 (_array_like_): Gram matrix with samples from source.
        G2 (_array_like_): Gram matrix with samples from target.
        v1 (_vector_): source distribution over samples.
        v2 (_vector_): target distribution over samples.
        max_itr (_int_): for APGD.
        lda (_float_): lambda regularization hyperparameter.
        lda3 (_float_): L2^2 regularization coefficient.
        conv_crit (_int_): whether to have convergence criterion.
        K (_int_): Cardinality of the support set.
        s (None or _int_): None if non-stochastic else
        cardinality of set R in the sparseotdash. Defaults to None.
        seed (_int_, optional): To seed for reproducibility. Defaults to 0.
        ws (_int_): Warm start or not. Defaults to 0.
        all_gamma (_int_): To return gamma of all iterations or not.
                           Defaults to 0.

    Returns:
        gamma     : if all_gamma, returns list of K sparse OT plans of
                    dimension 1 x 1 to K x 1, else returns the final.
        S_i       : i indices of the support set of cardinality K.
        S_j       : j indices of the support set of cardinality K.
        all_obj and all_nrm  : list of objective, grad norm at each iteration if verbose=1.
    """
    if K is not None:
        assert K <= C.shape[0]*C.shape[1], "K should be less than m*n"

    if vparts is None:
        vparts = {1: torch.dot(torch.mv(G1, v1), v1), 2: torch.dot(torch.mv(G2, v2), v2)}
    if not s:
        if ws:
            return get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, seed, tol)
        else:
            return get_gamma_dash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, seed, tol)
    else:
        if ws:
            return get_gamma_sdash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, s, seed, tol)
        else:
            return get_gamma_sdash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, s, seed, tol)


def get_gamma_dash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, seed, tol):
    """
    Implements Sparse-OT-Dash algorithm.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    tot = m*n if K is None else K

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak

    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device)  # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device)
    gamma0 = torch.zeros(tot, dtype=fixed_grd.dtype, device=device)  # initial gamma for APGD
    
    gammas = []
    gamma = None

    vec_grd = fixed_grd.flatten()
    u_0 = torch.argmin(vec_grd)
    neg_max_grd = vec_grd[u_0]
    if neg_max_grd >= 0:
        if verbose:
            return [gamma0[:1]] if all_gamma else gamma0[:1], S_i[:1], S_j[:1], {}, {}
        return [gamma0[:1]] if all_gamma else gamma0[:1], S_i[:1], S_j[:1]
    
    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        if k:
            # R: V_minus_L
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor')  # i indices of the remaining
            S_j_R = V_minus_L % n  # j indices of the remaining

            unq_i_R = torch.unique(S_i_R)  # unique i indices of the remaining
            unq_j_R = torch.unique(S_j_R)  # unique j indices of the remaining            
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)

            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]

            vec_grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] +
                                                       c2[idx_in_c2[S_j_R]])
            if lda3:
                vec_grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            if neg_max_grd >= 0:
                break

        # get the new element based on the chosen index & include
        chosen = V_minus_L[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        # update the remaining elements
        V_minus_L = torch.cat([V_minus_L[:u_0], V_minus_L[u_0+1:]])
        # NOTE: u_0 is an index (not an element) of V_minus_L only, chosen based on the grd

        # get step-size based on the support points
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L
        # *** solve UOT-MMD ***
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        fixed_grd_S = fixed_grd[S_i_S, S_j_S]
        grd = fixed_grd_S.clone()  # for the 1st gd step
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]

        y = gamma0[:k+1].clone()
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
                    if len(index_0) and cond2:
                        print(f"converged in {inner+1} iterations")
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


def get_gamma_sdash(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, s, seed, tol):
    """
    Implements Stochastic Sparse-OT-Dash algorithm.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    tot = m*n if K is None else K

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak

    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device)  # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device)
    gamma0 = torch.zeros(tot, dtype=fixed_grd.dtype, device=device)  # initial gamma for APGD

    gammas = []
    gamma = gamma0[:1].clone()

    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        R = V_minus_L[torch.randint(m*n-k, (s,))]
        if k:
            S_i_R = torch.div(R, n, rounding_mode='floor')
            S_j_R = R % n
            
            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)

            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]

            vec_grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] +
                                                       c2[idx_in_c2[S_j_R]])
            if lda3:
                vec_grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            
            if neg_max_grd >= 0:
                break
        else:
            vec_grd = fixed_grd.take(R)
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            if neg_max_grd >= 0:
                return [gamma] if all_gamma else gamma, S_i[:1], S_j[:1], [], []

        # get the new element based on the chosen index & include
        chosen = R[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        # list of current support points over which OT is computed

        # update the remaining elements
        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]
        # NOTE: u_0 is chosen from R so we need to infer its index first.

        # get step-size based on the support points
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L
        
        # *** solve UOT-MMD ***
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]
        
        fixed_grd_S = fixed_grd[S_i_S, S_j_S]
        grd = fixed_grd_S.clone()  # for the 1st gd step
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]

        y = gamma0[:k+1].clone()
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
                    if len(index_0) and cond2:
                        print(f"converged in {inner+1} iterations")
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

def get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, seed, tol):
    """
    Implements Sparse-OT-Dash algorithm with warm-start.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    tot = m*n if K is None else K

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak

    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device)  # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device)

    dummy_zero = torch.tensor([0], dtype=fixed_grd.dtype, device=device)
    gamma = dummy_zero.clone()
    
    gammas = []

    vec_grd = fixed_grd.flatten()
    u_0 = torch.argmin(vec_grd)
    neg_max_grd = vec_grd[u_0]
    if neg_max_grd >= 0:
        if verbose:
            return [gamma] if all_gamma else gamma, S_i[:1], S_j[:1], {}, {}
        return [gamma] if all_gamma else gamma, S_i[:1], S_j[:1]
    
    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        if k:
            # R: V_minus_L
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor')
            S_j_R = V_minus_L % n
            
            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)

            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]

            vec_grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] +
                                                       c2[idx_in_c2[S_j_R]])
            if lda3:
                vec_grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            if neg_max_grd >= 0:
                break

        # get the new element based on the chosen index & include
        chosen = V_minus_L[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        # update the remaining elements
        V_minus_L = torch.cat([V_minus_L[:u_0], V_minus_L[u_0+1:]])

        # get step-size based on the support points
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L
        # *** solve UOT-MMD ***
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        fixed_grd_S = fixed_grd[S_i_S, S_j_S]
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]

        y = torch.cat([gamma[:k], dummy_zero])
        y1, yT1 = get_marginals(y, S_i_S, S_j_S, m, n)
        r2 = torch.mv(G1_u, y1)
        c2 = torch.mv(G2_u, yT1)
        grd = fixed_grd_S + 2*lda*(r2[idx_i] + c2[idx_j])

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
                    if len(index_0) and cond2:
                        print(f"converged in {inner+1} iterations")
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

def get_gamma_sdash_ws(C, G1, G2, v1, v2, max_itr, K, lda, lda3, all_gamma, vparts, verbose, conv_crit, s, seed, tol):
    """
    Implements Stochastic Sparse-OT-Dash algorithm with warm-start.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    tot = m*n if K is None else K

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(tot, dtype=torch.long, device=device)  # to avoid memory leak

    tmp_idx = torch.zeros(max(m, n), dtype=torch.long, device=device)  # to avoid memory leak
    aranged = torch.arange(max(m, n), device=device)

    dummy_zero = torch.tensor([0], dtype=fixed_grd.dtype, device=device)
    gammas = []
    gamma = dummy_zero.clone()

    all_obj = {}
    all_nrm = {}
    for k in range(tot):
        all_obj[k] = []
        all_nrm[k] = []
        R = V_minus_L[torch.randint(m*n-k, (s,))]
        if k:
            S_i_R = torch.div(R, n, rounding_mode='floor')
            S_j_R = R % n

            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)
            gamma1, gammaT1 = get_marginals(gamma, S_i[:k], S_j[:k], m, n)
            r2 = torch.mv(G1[unq_i_R], gamma1)
            c2 = torch.mv(G2[unq_j_R], gammaT1)

            # r2, c2 entries are with the unique indices' order which needs to be converted to the chosen indices order
            idx_in_r2 = tmp_idx.clone()
            idx_in_r2[unq_i_R] = aranged[:unq_i_R.shape[0]]
            idx_in_c2 = tmp_idx.clone()
            idx_in_c2[unq_j_R] = aranged[:unq_j_R.shape[0]]

            vec_grd = fixed_grd[S_i_R, S_j_R] + 2*lda*(r2[idx_in_r2[S_i_R]] +
                                                       c2[idx_in_c2[S_j_R]])
            if lda3:
                vec_grd += lda3*postprocess_gamma(gamma, S_i[:k], S_j[:k], m, n)[S_i_R, S_j_R]
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            if neg_max_grd >= 0:
                    break
        else:
            vec_grd = fixed_grd.take(R)
            u_0 = torch.argmin(vec_grd)
            neg_max_grd = vec_grd[u_0]
            if neg_max_grd >= 0:
                if verbose:
                    return [gamma] if all_gamma else gamma, S_i[:1], S_j[:1], {}, {}
                return [gamma] if all_gamma else gamma, S_i[:1], S_j[:1]

        # get the new element based on the chosen index & include
        chosen = R[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]        

        # update the remaining elements
        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]
        # NOTE: u_0 is chosen from R so we need to infer its index first.

        # upd unique, upd ss
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        m1, m2 = len(unq_i), len(unq_j)
        L_orig = m2**2*norm(G1_ss)**2 + m1**2*norm(G2_ss)**2 + 2*(torch.sum(G1_ss)*torch.sum(G2_ss))
        L = 2*sqrt(lda**2*L_orig + lda3**2*m1*m2 + 2*lda*lda3*(m2*torch.trace(G1_ss) + m1*torch.trace(G2_ss))) if lda3 else 2*lda*sqrt(L_orig)
        ss = 1/L
        
        # *** solve UOT-MMD ***
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]
        
        fixed_grd_S = fixed_grd[S_i_S, S_j_S]
        # for mapping from unique indices' based order to chosen indices' based order
        idx_in_r2 = tmp_idx.clone()
        idx_in_r2[unq_i] = aranged[:unq_i.shape[0]]
        idx_in_c2 = tmp_idx.clone()
        idx_in_c2[unq_j] = aranged[:unq_j.shape[0]]
        idx_i = idx_in_r2[S_i_S]
        idx_j = idx_in_c2[S_j_S]

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
                    if len(index_0) and cond2:
                        print(f"converged in {inner+1} iterations")
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
