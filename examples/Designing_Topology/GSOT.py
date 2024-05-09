import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 8})
from ADMM import ADMM
from utils import *

def get_edges_from_plan(trans_plan:np.ndarray, max_num_edges):
    '''
    convert the transport plan  into a set of sorted edges
    :param trans: input array of computed transport plan
    :param max_num_edges: the maximum number of edges in the proposed chain
    :return:
     edges: the designed edges, a list of supply-demand node pairs with length max_num_edges
    '''
    num_src = trans_plan.shape[0]
    chain = np.sum(trans_plan, axis=2)
    ind = np.unravel_index(np.argsort(chain, axis=None)[::-1], chain.shape)
    edges = [[ind[0][k], ind[1][k] + num_src] for k in range(max_num_edges)]

    return np.array(edges)

def get_transport_plan(price: np.ndarray,
                       supply:  np.ndarray,
                       demand:  np.ndarray,
                       balance: bool=False,
                       alpha:   float=1,
                       rho:     float=1,
                       iterations:int = 300) -> Tuple[np.ndarray, np.ndarray, float]:
    """
       :param price:    a cost matrix with size (ns, nd)
       :param supply:   N supply histograms with size (ns, N)
       :param demand:   N demand histograms with size (nd, N)
       :param balance:  True: balanced case
                        False: unbalance case
       :param alpha:    hyperparameter
       :param rho:      hyperparameter
        recommend (alpha,rho) = { (0.1,0.1) , (1,1) }
       :param iterations: the number of iterations

       :return:
           transports: a tensor with size (ns, nd, N)
       """

    num_src, num_dst = price.shape
    num_pairs = supply.shape[1]
    admm = ADMM(num_src, num_dst, num_pairs, price, supply, demand, alpha, rho)

    #normalize supply vector and demand vector
    admm.Normalization()
    if(balance):
        #Balanced case

        for i in range(0, iterations):
            admm.Update_BalanceT()
            admm.UpdateZ()
            admm.UpdateU()
            print(i)

    else:
        # UnBalanced case
        for i in range(0, iterations):
            admm.Update_UnbalanceT_CG()
            #admm.Update_UnbalanceT_semi()
            admm.UpdateZ()
            admm.UpdateU()
            print(i)

    return admm.T


def evaluate_net_profit(sorted_edges: np.ndarray,
                        price: np.ndarray,
                        supply: np.ndarray,
                        demand: np.ndarray,
                        integer: bool = False) -> np.ndarray:
    """
    Evaluation the profit of designed chains
    :param sorted_edges: (K, 2) the chains we derived, the chains are sorted according to their significance
    :param price: a cost with size (ns, nd)
    :param supply: (ns, N) supply histograms
    :param demand: (nd, N) demand histograms
    :param integer: require integer variables or not
    :param max_num_edges: the predefined maximum number of edges
    :return:
        values: the profit achieved by the network as edges gradually being added
                to the network ,with size (K, N)

        net_profit: the total profit achieved by the network (all edges include)
    """

    max_num_edges=len(sorted_edges)

    num_pairs = supply.shape[1]
    values = np.zeros((max_num_edges, num_pairs))
    for k in range(sorted_edges.shape[0]):
        edges = sorted_edges[:k + 1, :]
        for n in range(supply.shape[1]):
            values[k, n], _, _ = max_flow(price=price,
                                          edges=edges,
                                          supply=supply[:, n],
                                          demand=demand[:, n],
                                          integer=integer)

    net_profit=np.mean(values, axis=1).max()
    return values,net_profit



def plot_network(path_name: str, num_src: int, num_dst: int, edge: list ) :
    """
    Plot the supply chain
    :param path_name: the path with image name
    :param num_src: M (the number of supply nodes)
    :param num_dst: N (the number of demand nodes)
    :param edge: the designed edges, a list of supply-demand node pairs with length max_num_edges
    :return:
    """
    pts_src = (np.array(list(range(num_src))).reshape(num_src, 1) + 0.5) / num_src
    pts_src = np.concatenate((np.zeros_like(pts_src), pts_src), axis=1)
    pts_dst = (np.array(list(range(num_dst))).reshape(num_dst, 1) + 0.5) / num_dst
    pts_dst = np.concatenate((np.ones_like(pts_dst), pts_dst), axis=1)

    plt.figure(figsize=(6, 5))
    plt.scatter(pts_src[:, 0], pts_src[:, 1], marker='o', s=20, c='blue')
    plt.scatter(pts_dst[:, 0], pts_dst[:, 1], marker='o', s=20, c='red')
    for i, j in edge:
        plt.plot([pts_src[i, 0], pts_dst[j - num_src, 0]],
                 [pts_src[i, 1], pts_dst[j - num_src, 1]],
                 'k-')

    plt.title('#Chains={}'.format(len(edge)))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(path_name)
    plt.close()

def visualize_map(data: np.ndarray,
                  dst: str,
                  xticklabels: List[str],
                  yticklabels: List[str],
                  xlabel: str = None,
                  ylabel: str = None):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(data,
                     linewidth=0,
                     square=True,
                     annot=False,
                     cmap="YlGnBu_r",
                     xticklabels=xticklabels,
                     yticklabels=yticklabels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    if ylabel is not None:
        plt.ylabel(ylabel, fontdict={'size': 24})
    if xlabel is not None:
        plt.xlabel(xlabel, fontdict={'size': 24})
    plt.tight_layout()
    plt.savefig(dst)
    plt.close()