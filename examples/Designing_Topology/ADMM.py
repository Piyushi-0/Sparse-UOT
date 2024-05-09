import matplotlib.pyplot as plt
import ot
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8})

from smooth_ot import *
element_max = np.vectorize(max)


class ADMM:

    def __init__(self,n,m,k,P,u_c,u_d,alpha=1,rho=1,gamma=1):
        self.m = m
        self.n = n
        self.k = k
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.P = P
        self.C = -P
        self.u_c = u_c
        self.u_d = u_d
        self.T = np.zeros((n,m,k), dtype=np.float64)
        self.U = np.zeros((n,m,k), dtype=np.float64)
        self.Z = np.zeros((n,m,k), dtype=np.float64)

        for i in range(k):
            self.T[:,:,i] = np.outer(self.u_c[:,i],self.u_d[:,i])
            self.Z[:,:,i] = self.T[:,:,i]

    def Normalization(self):
        for k in range (self.u_d.shape[1]):
            self.u_d[:,k] /= np.sum(self.u_c[:,k])
        for k in range (self.u_c.shape[1]):
            self.u_c[:,k] /= np.sum(self.u_c[:,k])

    def Update_BalanceT(self):
        '''
        the primal-dual algorithm in (Blondel et al., 2017)
        '''

        #self.T = UpdateBT(self.T,self.Z,self.U,self.P,self.rho,self.u_c,self.u_d)
        regul = SquaredT(gamma=1.0)
        # update K independent optimal transport problems
        for k in range(self.T.shape[2]):
            C = -self.P - self.rho * (self.Z[:, :, k] - self.U[:, :, k])
            alpha, beta = solve_dual( self.u_c[:, k], self.u_d[:, k], C,regul, max_iter=1000)
            self.T[:, :, k] = get_plan_from_dual(alpha, beta, C, regul)

    def Update_UnbalanceT_semi(self):
        '''
               the semi-dual algorithm in (Blondel et al., 2017)
        '''
        regul = SquaredT(gamma=1.0)
        for k in range(self.T.shape[2]):
            C = -self.P - self.rho * (self.Z[:, :, k] - self.U[:, :, k])
            alpha = solve_semi_dual(self.u_c[:, k], self.u_d[:, k],C,  regul, max_iter=1000)
            self.T[:, :, k] = get_plan_from_semi_dual(alpha,self.u_d[:, k] , C, regul)

    def Update_UnbalanceT_CG(self):
        '''
        the scaling algorithm in (Frogner et al., 2015)
        '''

        lamb = 2 / self.rho
        for k in range(self.T.shape[2]):
            C = self.rho * self.U[:, :, k] - self.P - 0.5 * self.rho * np.log(self.Z[:, :, k] + 1e-9)
            uc = self.u_c[:, k]
            ud = self.u_d[:, k]
            self.T[:, :, k] = ot.sinkhorn(uc, ud, C, lamb)


    def UpdateZ(self):
        '''
        Applying soft-thresholding method
        z_ij = max{ 1- alpha/(rho*||r_ij||) , 0}
        r_ij = t_ij + u_ij
        '''
        for i in range(self.Z.shape[0]):
            for j in range(self.Z.shape[1]):
                r_ij = self.T[i, j, :] + self.U[i, j, :]
                Z_ij = self.soft_thresholding (r_ij)+1e-8
                self.Z[i, j, :] = Z_ij

    def soft_thresholding (self,r_ij):
        '''
        z_ij = max{ 1- alpha/(rho*||r_ij||) , 0}
        '''
        norm2 = np.sqrt(np.sum(r_ij ** 2))
        if norm2 > 0:
            z = max(0, 1 - self.alpha / (self.rho * norm2)) * r_ij
        else:
            z = np.zeros_like(r_ij)

        return z

    def UpdateU(self):

        self.U = self.U+(self.T-self.Z)