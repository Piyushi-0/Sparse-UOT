"""
This code implements the stochastic K-sparse-OT algorithm with full
representations ie. gamma is of size m x n.

- get_gamma: Returns the sparse OT plan gamma of dimension m x n.

"""


import torch
import numpy as np
from torch import sqrt
from torch.linalg import norm


tol = 1e-6
eps = 1e-9


def seed_everything(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_marginals(gamma):
    gamma1 = torch.sum(gamma, 1)
    gammaT1 = torch.sum(gamma, 0)
    return gamma1, gammaT1


def get_obj(C, G1, G2, v1, v2, gamma, lda, vparts=None):
    def get_mmdsq_reg(gamma1, gammaT1, v1, v2, vparts, G1, G2):
        reg_1 = torch.mv(G1, gamma1-2*v1).dot(gamma1) + vparts[1]
        reg_2 = torch.mv(G2, gammaT1-2*v2).dot(gammaT1) + vparts[2]
        return reg_1, reg_2

    if vparts is None:
        g1 = torch.mv(G1, v1)
        g2 = torch.mv(G2, v2)
        vparts = {1: g1.dot(v1), 2: g2.dot(v2)}

    gamma1, gammaT1 = get_marginals(gamma)
    reg_1, reg_2 = get_mmdsq_reg(gamma1, gammaT1, v1, v2, vparts, G1, G2)
    obj = torch.tensordot(gamma, C) + lda*(reg_1+reg_2)
    return obj


def get_gamma(C, G1, G2, v1, v2, max_itr, lda, K, s=0, seed=0, ws=0,
              all_gamma=0):
    """Computes sparse gamma using K-sparse-OT-dash algorithm.

    Args:
        C (_array_like_): cost matrix between source and target.
        G1 (_array_like_): Gram matrix with samples from source.
        G2 (_array_like_): Gram matrix with samples from target.
        v1 (_vector_): source distribution over samples.
        v2 (_vector_): target distribution over samples.
        max_itr (_int_): for APGD.
        lda (_float_): lambda regularization hyperparameter.
        K (_int_): Cardinality of the support set.
        s (None or _int_): None if non-stochastic else
        cardinality of set R in the sparseotdash. Defaults to None.
        seed (int, optional): To seed for reproducibility. Defaults to 0.
        ws (int): Warm start or not. Defaults to 0.
        all_gamma (int): To return gamma of all iterations or not.
                         Defaults to 0.

    Returns:
        if all_gamma, returns list of K sparse OT plans of
        else returns the final.
    """
    if K is not None:
        assert K <= C.shape[0]*C.shape[1], "K should be less than m*n"

    vparts = None  # {1: torch.dot(torch.mv(G1, v1), v1), 2: torch.dot(torch.mv(G2, v2), v2)}
    if not s:
        if ws:
            return get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, lda,
                                     seed, all_gamma, K, vparts)
        else:
            return get_gamma_dash(C, G1, G2, v1, v2, max_itr, lda,
                                  seed, all_gamma, K, vparts)
    else:
        if ws:
            return get_gamma_sdash_ws(C, G1, G2, v1, v2, max_itr, lda,
                                      s, seed, all_gamma, K, vparts)
        else:
            return get_gamma_sdash(C, G1, G2, v1, v2, max_itr, lda,
                                   s, seed, all_gamma, K, vparts)


def get_gamma_dash(C, G1, G2, v1, v2, max_itr, lda, seed, allg,
                   K, vparts):
    """Implements Sparse-OT-Dash algorithm.

    Returns:
        all_gamma: list of K sparse OT plans corresponding to cardinalities
        1 to K.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak

    gamma0 = torch.zeros((m, n), dtype=fixed_grd.dtype,
                         device=fixed_grd.device)
    g1_0 = gamma0[:, 0].clone()
    g2_0 = gamma0[0, :].clone()
    all_gamma = []
    gamma = None
    u_0 = torch.argmin(fixed_grd)
    for k in range(K):
        if k:
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor')
            S_j_R = V_minus_L % n
            gamma1, gammaT1 = get_marginals(gamma)

            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)

            g1 = torch.mv(G1[unq_i_R], gamma1)
            g2 = torch.mv(G2[unq_j_R], gammaT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i_R, g1)
            g2_new = g2_new.index_copy(0, unq_j_R, g2)

            u_0 = torch.argmin((fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                                ).take(V_minus_L))
        chosen = V_minus_L[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n
        
        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        V_minus_L = torch.cat([V_minus_L[:u_0], V_minus_L[u_0+1:]])

        # upd unique, upd ss
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        ss = 1/(2*lda*(sqrt(len(unq_j)**2*norm(G1_ss)**2 +
                            len(unq_i)**2*norm(G2_ss)**2 +
                            2*(torch.sum(G1_ss)*torch.sum(G2_ss)))))

        # solve for gamma using apgd
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]
        y = gamma0.clone()
        gamma = gamma0.clone()
        x_old = y.clone()
        t = 1

        grd_S = fixed_grd[S_i_S, S_j_S].clone()

        for _ in range(max_itr):
            gamma[S_i_S, S_j_S] = torch.clamp(y[S_i_S, S_j_S]-ss*grd_S, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y[S_i_S, S_j_S] = (t_dash+1)*gamma[S_i_S, S_j_S] - \
                t_dash*x_old[S_i_S, S_j_S]
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y)

            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]

        if allg:
            all_gamma.append(gamma)
    return all_gamma if allg else gamma


def get_gamma_sdash(C, G1, G2, v1, v2, max_itr, lda, s, seed,
                    allg, K, vparts):
    """Implements Stochastic Sparse-OT-Dash algorithm

    Returns:
        all_gamma: list of K sparse OT plans corresponding to cardinalities
        1 to K.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak

    gamma0 = torch.zeros((m, n), dtype=fixed_grd.dtype,
                         device=fixed_grd.device)
    g1_0 = gamma0[:, 0].clone()
    g2_0 = gamma0[0, :].clone()
    all_gamma = []
    gamma = None
    for k in range(K):
        R = V_minus_L[torch.randint(m*n-k, (s,))]
        if k:
            S_i_R = torch.div(R, n, rounding_mode='floor')
            S_j_R = R % n

            gamma1, gammaT1 = get_marginals(gamma)
            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)

            g1 = torch.mv(G1[unq_i_R], gamma1)
            g2 = torch.mv(G2[unq_j_R], gammaT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i_R, g1)
            g2_new = g2_new.index_copy(0, unq_j_R, g2)
            u_0 = torch.argmin((fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                                ).take(R))
        else:
            u_0 = torch.argmin(fixed_grd.take(R))

        chosen = R[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]

        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        ss = 1/(2*lda*(sqrt(len(unq_j)**2*norm(G1_ss)**2 +
                            len(unq_i)**2*norm(G2_ss)**2 +
                            2*(torch.sum(G1_ss)*torch.sum(G2_ss)))))

        # solve for gamma using apgd
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        y = gamma0.clone()
        gamma = gamma0.clone()
        x_old = y.clone()
        t = 1

        grd_S = fixed_grd[S_i_S, S_j_S].clone()

        for _ in range(max_itr):
            gamma[S_i_S, S_j_S] = torch.clamp(y[S_i_S, S_j_S]-ss*grd_S, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y[S_i_S, S_j_S] = (t_dash+1)*gamma[S_i_S, S_j_S] - \
                t_dash*x_old[S_i_S, S_j_S]
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y)

            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]

        if allg:
            all_gamma.append(gamma)
    return all_gamma if allg else gamma


def get_gamma_dash_ws(C, G1, G2, v1, v2, max_itr, lda, seed, allg,
                      K, vparts):
    """Implements Sparse-OT-Dash algorithm with warm start.

    Returns:
        all_gamma: list of K sparse OT plans corresponding to cardinalities
        1 to K.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak

    gamma = torch.zeros((m, n), dtype=fixed_grd.dtype,
                        device=fixed_grd.device)
    g1_0 = gamma[:, 0].clone()
    g2_0 = gamma[0, :].clone()
    all_gamma = []

    u_0 = torch.argmin(fixed_grd)
    for k in range(K):
        if k:
            # R: V_minus_L
            S_i_R = torch.div(V_minus_L, n, rounding_mode='floor')
            S_j_R = V_minus_L % n
            gamma1, gammaT1 = get_marginals(gamma)

            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)

            g1 = torch.mv(G1[unq_i_R], gamma1)
            g2 = torch.mv(G2[unq_j_R], gammaT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i_R, g1)
            g2_new = g2_new.index_copy(0, unq_j_R, g2)

            u_0 = torch.argmin((fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                                ).take(V_minus_L))
        chosen = V_minus_L[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n
        
        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        V_minus_L = torch.cat([V_minus_L[:u_0], V_minus_L[u_0+1:]])

        # upd unique, upd ss
        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        ss = 1/(2*lda*(sqrt(len(unq_j)**2*norm(G1_ss)**2 +
                            len(unq_i)**2*norm(G2_ss)**2 +
                            2*(torch.sum(G1_ss)*torch.sum(G2_ss)))))

        # solve for gamma using apgd
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        y = gamma.clone()
        x_old = y.clone()
        t = 1

        if k:
            y1, yT1 = get_marginals(y)
            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]
        else:
            grd_S = fixed_grd[S_i_S, S_j_S].clone()
        for _ in range(max_itr):
            gamma[S_i_S, S_j_S] = torch.clamp(y[S_i_S, S_j_S]-ss*grd_S, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y[S_i_S, S_j_S] = (t_dash+1)*gamma[S_i_S, S_j_S] - \
                t_dash*x_old[S_i_S, S_j_S]
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y)

            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]
        if allg:
            all_gamma.append(gamma)
    return all_gamma if allg else gamma


def get_gamma_sdash_ws(C, G1, G2, v1, v2, max_itr, lda, s, seed,
                       allg, K, vparts):
    """Implements Stochastic Sparse-OT-Dash algorithm with warm start.

    Returns:
        all_gamma: list of K sparse OT plans corresponding to cardinalities
        1 to K.
    """
    seed_everything(seed)
    m, n = C.shape
    device = C.device

    g1 = torch.mv(G1, v1)
    g2 = torch.mv(G2, v2)

    fixed_grd = C-2*lda*(g1[:, None] + g2)

    V_minus_L = torch.arange(m*n, device=device)
    S_i = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak
    S_j = torch.zeros(K, dtype=torch.long, device=device)  # to avoid memory leak

    gamma = torch.zeros((m, n), dtype=fixed_grd.dtype,
                        device=fixed_grd.device)
    g1_0 = gamma[:, 0].clone()
    g2_0 = gamma[0, :].clone()
    all_gamma = []

    for k in range(K):
        R = V_minus_L[torch.randint(m*n-k, (s,))]
        if k:
            S_i_R = torch.div(R, n, rounding_mode='floor')
            S_j_R = R % n

            gamma1, gammaT1 = get_marginals(gamma)
            unq_i_R = torch.unique(S_i_R)
            unq_j_R = torch.unique(S_j_R)

            g1 = torch.mv(G1[unq_i_R], gamma1)
            g2 = torch.mv(G2[unq_j_R], gammaT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i_R, g1)
            g2_new = g2_new.index_copy(0, unq_j_R, g2)
            u_0 = torch.argmin((fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                                ).take(R))
        else:
            u_0 = torch.argmin(fixed_grd.take(R))

        chosen = R[u_0]
        S_i[k] = torch.div(chosen, n, rounding_mode='floor')
        S_j[k] = chosen % n

        S_i_S = S_i[:k+1]
        S_j_S = S_j[:k+1]

        index = torch.where(V_minus_L != chosen)[0]
        V_minus_L = V_minus_L[index]

        unq_i = torch.unique(S_i_S)
        unq_j = torch.unique(S_j_S)
        G1_ss = G1[unq_i, :][:, unq_i]
        G2_ss = G2[unq_j, :][:, unq_j]
        ss = 1/(2*lda*(sqrt(len(unq_j)**2*norm(G1_ss)**2 +
                            len(unq_i)**2*norm(G2_ss)**2 +
                            2*(torch.sum(G1_ss)*torch.sum(G2_ss)))))

        # solve for gamma using apgd
        G1_u = G1[unq_i]
        G2_u = G2[unq_j]

        y = gamma.clone()
        x_old = y.clone()
        t = 1

        if k:
            y1, yT1 = get_marginals(y)
            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]
        else:
            grd_S = fixed_grd[S_i_S, S_j_S].clone()
        for _ in range(max_itr):
            gamma[S_i_S, S_j_S] = torch.clamp(y[S_i_S, S_j_S]-ss*grd_S, min=0)
            t_new = (1+np.sqrt(1+4*t**2))/2
            t_dash = (t-1)/t_new
            y[S_i_S, S_j_S] = (t_dash+1)*gamma[S_i_S, S_j_S] - \
                t_dash*x_old[S_i_S, S_j_S]
            x_old = gamma.clone()
            t = t_new
            y1, yT1 = get_marginals(y)

            g1 = torch.mv(G1_u, y1)
            g2 = torch.mv(G2_u, yT1)
            g1_new = g1_0.clone()
            g2_new = g2_0.clone()
            g1_new = g1_new.index_copy(0, unq_i, g1)
            g2_new = g2_new.index_copy(0, unq_j, g2)
            grd_S = (fixed_grd+2*lda*(g1_new[:, None] + g2_new)
                     )[S_i_S, S_j_S]
        if allg:
            all_gamma.append(gamma)
    return all_gamma if allg else gamma
