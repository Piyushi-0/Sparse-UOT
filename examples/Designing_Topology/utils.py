import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
from typing import List, Tuple
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8})
class RealData:
    def __init__(self, num_pairs):
        """
        Initialize data simulator
        :param num_src: the number of "supply" sources
        :param num_dst: the number of "demand" destinations
        :param var: the variance of the demands' dynamics in the range [0, 1]
        """
        self.num_src = 10
        self.num_dst = 10
        self.num_pairs = num_pairs
        self.cost = np.array([110, 99, 80, 90, 123, 173, 133, 73, 93, 148])/173
        self.cost = np.tile(self.cost, (10, 1))
        self.dmean = np.array([1017, 1042, 1358, 2525, 1100, 2150, 1113, 4017, 3296, 2383])
        self.dvar = np.array([194, 323, 248, 340, 381, 404, 524, 556, 1047, 697])
        self.vec_supply = self.dmean.reshape((10, 1))
        self.supply = self.vec_supply / self.vec_supply.sum()
        self.supply = np.tile(self.supply, (1, num_pairs))

    def generate_data(self):
        sample = np.zeros((self.num_dst, self.num_pairs), dtype=int)
        # 对每个需求分布dst，采样pair个数据
        for i in range(self.num_dst):
            seed = self.dmean[i]
            var = self.dvar[i]
            sample[i, :] = np.random.normal(loc=seed, scale=var, size=self.num_pairs)

        sample_normed = sample /self.vec_supply.sum()
        return sample_normed[:, :int(1/2*self.num_pairs)],sample_normed[:, int(1/2*self.num_pairs):]

class SystheticData:
    def __init__(self, num_src: int, num_dst: int = None, var: float = 0.0):
        """
        Initialize data simulator
        :param num_src: the number of "supply" sources
        :param num_dst: the number of "demand" destinations
        :param var: the variance of the demands' dynamics in the range [0, 1]
        """
        self.num_src = num_src
        if num_dst is None:
            self.num_dst = num_src
        else:
            self.num_dst = num_dst
        self.var = var
        # set source and destination points and the distance between them
        self.pts_src = (np.array(list(range(self.num_src))).reshape(self.num_src, 1) + 0.5) / self.num_src
        self.pts_src = np.concatenate((np.zeros_like(self.pts_src), self.pts_src), axis=1)
        self.pts_dst = (np.array(list(range(self.num_dst))).reshape(self.num_dst, 1) + 0.5) / self.num_dst
        self.pts_dst = np.concatenate((np.ones_like(self.pts_dst), self.pts_dst), axis=1)
        
        self.pts_src = normalize(self.pts_src)
        self.pts_dst = normalize(self.pts_dst)
        
        self.cost = euclidean_distances(self.pts_src, self.pts_dst)

    def generate_pairs(self,
                       num_pairs: int = 100,
                       seed: int = 42,
                       integer: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate supply-demand pairs
        :param num_pairs: the number of pairs
        :param seed: the random seed controlling the perturbation
        :param integer: require integer or not
        :return:
        """
        if integer:
            supply = self.num_dst * np.ones((self.num_src, num_pairs))
            demand_ideal = self.num_src * np.ones((self.num_dst, num_pairs))
            perturbation = np.random.RandomState(seed).rand(self.num_dst, num_pairs)
            perturbation = np.round(self.var * ((perturbation - 0.5) * 2) * min([self.num_src, self.num_dst]))
        else:
            supply = np.ones((self.num_src, num_pairs)) / self.num_src
            demand_ideal = np.ones((self.num_dst, num_pairs)) * self.num_src / self.num_dst
            perturbation = np.random.RandomState(seed).rand(self.num_dst, num_pairs)
            perturbation = self.var * ((perturbation - 0.5) * 2) * (self.num_src / self.num_dst)
        demand = perturbation + demand_ideal
        return supply[:, :int(1/2*num_pairs)], demand


    def generate_data(self, num_pairs,balan:bool=True):
        sample = np.zeros((self.num_dst, num_pairs))
        # 对每个需求分布dst，采样pair个数据
        for i in range(self.num_dst):
            seed = np.random.uniform(5, 8)
            sample[i, :] = np.random.normal(loc=seed, scale=0.5, size=num_pairs)
        # scaling column-wise
        if(balan):
            sample_normed = sample / sample.sum(axis=0)
        else:
            sample_normed = sample
        return sample_normed[:,:int(1/2*num_pairs)],sample_normed[:,int(1/2*num_pairs):]

    def plot_supply_chain(self, path_name: str, chain: np.ndarray):
        """
        Plot the supply chain
        :param path_name: the path with image name
        :param chain: the proposed chain, a matrix with size (num_src, num_dst)
        :return:
        """
        plt.figure(figsize=(6, 5))
        plt.scatter(self.pts_src[:, 0], self.pts_src[:, 1], marker='o', s=20, c='blue')
        plt.scatter(self.pts_dst[:, 0], self.pts_dst[:, 1], marker='o', s=20, c='red')
        for i in range(self.num_src):
            for j in range(self.num_dst):
                if chain[i, j] > 1e-8:
                    plt.plot([self.pts_src[i, 0], self.pts_dst[j, 0]],
                             [self.pts_src[i, 1], self.pts_dst[j, 1]],
                             'k-', alpha=chain[i, j])
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'#Chains={np.sum(chain > 1e-8)}_'+path_name)
        plt.close()

    def plot_edge(self, path_name: str, edge: list):
        """
        Plot the supply chain
        :param path_name: the path with image name
        :param chain: the proposed chain, a matrix with size (num_src, num_dst)
        :return:
        """

        plt.figure(figsize=(6, 5))
        plt.scatter(self.pts_src[:, 0], self.pts_src[:, 1], marker='o', s=20, c='blue')
        plt.scatter(self.pts_dst[:, 0], self.pts_dst[:, 1], marker='o', s=20, c='red')
        for i, j in edge:
            plt.plot([self.pts_src[i, 0], self.pts_dst[j - self.num_src, 0]],
                     [self.pts_src[i, 1], self.pts_dst[j - self.num_src, 1]],
                     'k-')

        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'#Chains={len(edge)}_'+path_name)
        plt.close()

def edge2mat(edges: np.ndarray, num_nodes: int):
    """
    :param edges: a matrix with size (K, 2) for K directed edges, each row (v, u) indicates an edge v->u
    :param num_nodes: the number of nodes in a graph
    :return:
        a matrix with size (#nodes, #edges)
    """
    edge_cost = np.zeros((num_nodes, edges.shape[0]))
    for n in range(num_nodes):
        edge_cost[n, edges[:, 0] == n] = 1
        edge_cost[n, edges[:, 1] == n] = 1
    return edge_cost


def max_flow(price: np.ndarray,
             edges: np.ndarray,
             supply: np.ndarray,
             demand: np.ndarray,
             integer: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Minimum flow algorithm given a bipartite graph
    :param price: the cost matrix with size (ns, nd)
    :param edges: an array with size (K, 2), each row represents an edge u->v
    :param supply: (ns, ) supply histogram
    :param demand: (nt, ) demand histogram
    :param integer: require integer variable or not
    :return:
        result: the optimum objective
        weights: the weights on the edges, with size (K, )
        flow: the flows on the edges, with size (K, )
    """
    num_src = supply.shape[0]
    weights = np.array([price[edges[i, 0], edges[i, 1] - num_src] for i in range(edges.shape[0])])
    edge_topo = edge2mat(edges, num_nodes=supply.shape[0] + demand.shape[0])
    b = np.concatenate((supply, demand), axis=0)
    if integer:
        x = cp.Variable(edges.shape[0], nonneg=True, integer=True)
    else:
        x = cp.Variable(edges.shape[0], nonneg=True)

    objective = cp.Maximize(cp.sum(weights * x))

    constraints = [edge_topo @ x <= b, x >= np.zeros((edges.shape[0],))]
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    return result, weights, x.value