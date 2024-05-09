from GSOT import get_transport_plan, get_edges_from_plan
from GSOT import evaluate_net_profit, plot_network
import numpy as np
import math
from utils import *
import argparse
from sparse_ot.utils import createLogHandler
import os, pickle
from sparse_ot.utils import get_G, postprocess_gamma
from sparse_ot.sparse_repr import get_gamma
import torch

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GSOT')
parser.add_argument('--M', type=int, default=100, help='number of source nodes')
parser.add_argument('--N', type=int, default=100, help='number of destination nodes')
parser.add_argument('--K', type=int, default=20, help='number of OT pairs')
parser.add_argument('--s_eps', type=float, default=0)
parser.add_argument('--L', type=int, default=25, help='max no. edges')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bal', type=bool, default=True)
parser.add_argument('--data', default='synth')
parser.add_argument('--method', default="prp")
parser.add_argument('--lda', type=float, default=10)
parser.add_argument('--lda3', type=float, default=0.1)
parser.add_argument('--khp', type=float, default=-1)
parser.add_argument('--ktype', type=str, default="rbf")
parser.add_argument('--save_as', default="")
parser.add_argument('--val', default=1, type=int)
args = parser.parse_args()
M = args.M
N = args.N
K = args.K
K_hlf = int(K/2)
bal = args.bal
method = args.method
seed = args.seed
data = args.data
s_eps = args.s_eps
lda = args.lda
lda3 = args.lda3
khp = args.khp
ktype = args.ktype

L = args.L
our_K = int(L/K_hlf)

mx = max(M, N)
ub = int(2.5*mx)
if L < mx:
    L = mx
elif L > ub:
    L = ub

save_as = f"res_seps_{L}/{method}_{data}_{bal}/{ktype}_{khp}_{lda}_{lda3}_{s_eps}_{M}_{N}_{K}"
os.makedirs(save_as, exist_ok=True)

def get_t(a):
    return torch.from_numpy(a)

simulator = SystheticData(num_src=M, num_dst=N, var=0.3)
supply, demand = simulator.generate_pairs(num_pairs=K, seed=12)
train_demand, test_demand = simulator.generate_data(num_pairs=K, balan=bal)

if args.val:
    dem = train_demand.copy()
else:
    dem = test_demand.copy()

log_path = f"{save_as}/log_{args.val}.csv"
new = 0
if not os.path.exists(log_path):
    new = 1
logger = createLogHandler(log_path)
if new:
    logger.info("Profit, M, N, K, L, s_eps, khp, ktype, lda, lda3, seed")

assert "prp" in method

supp = get_t(simulator.pts_src)
dst = get_t(simulator.pts_dst)
C = get_t(simulator.cost)

G1 = get_G(x=supp, y=supp, khp=khp, ktype=ktype)
G2 = get_G(x=dst, y=dst, khp=khp, ktype=ktype)

s = int(np.ceil((M*N)/our_K*math.log(1/s_eps)))

plans = np.zeros((K_hlf, M, N))
for ix in range(K_hlf):
    v1 = get_t(supply[:, ix]).view(-1)
    v2 = get_t(dem[:, ix]).view(-1)
    gamma, S_i, S_j = get_gamma(C, G1, G2, v1, v2, max_itr=1000, K=our_K, lda=lda, lda3=lda3, ws=1, s=s)
    plans[ix] = postprocess_gamma(gamma, S_i, S_j, M, N).numpy()/K_hlf
plans = np.moveaxis(plans, 0, 2)
edges = get_edges_from_plan(trans_plan=plans, max_num_edges=L)
    
# with open(f"{save_as}/{seed}.pickle", "wb") as fp:
#     pickle.dump(plans, fp)

values, net_profit = evaluate_net_profit(sorted_edges=edges, price=simulator.cost,
                                         supply=supply, demand=dem)
# plot_network(f"{save_as}/{seed}.pdf", M, N, edges)
logger.info(f"{net_profit}, {M}, {N}, {K}, {L}, {s_eps}, {khp}, {ktype}, {lda}, {lda3}, {seed}")
