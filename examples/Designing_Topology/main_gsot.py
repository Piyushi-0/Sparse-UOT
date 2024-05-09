from GSOT import get_transport_plan, get_edges_from_plan
from GSOT import evaluate_net_profit, plot_network
import numpy as np
from utils import *
import argparse
from sparse_ot.utils import createLogHandler
import os, pickle

parser = argparse.ArgumentParser(description='GSOT')
parser.add_argument('--M', type=int, default=100, help='number of source nodes')
parser.add_argument('--N', type=int, default=100, help='number of destination nodes')
parser.add_argument('--K', type=int, default=20, help='number of OT pairs')
parser.add_argument('--L', type=int, default=25, help='max no. edges')
parser.add_argument('--method', default='gsot')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bal', type=bool, default=True)
parser.add_argument('--data', default='synth')
parser.add_argument('--save_as', default="")
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--val', type=int, default=1)

args = parser.parse_args()
M = args.M
N = args.N
K = args.K
bal = args.bal
method = args.method
seed = args.seed
data = args.data
L = args.L

mx = max(M, N)
ub = int(2.5*mx)
if L < mx:
    L = mx
elif L > ub:
    L = ub

save_as = f"{args.save_as}_{method}_{data}_{bal}_{args.alpha}, {args.rho}"
os.makedirs(f"results/{save_as}", exist_ok=True)
log_path = f"results/{save_as}/log_{args.val}.csv"

new = 0
if not os.path.exists(log_path):
    new = 1
logger = createLogHandler(log_path)
if new:
    logger.info("Profit, M, N, K, L, seed")

if "synth" in data:
    simulator = SystheticData(num_src=M, num_dst=N, var=0.3)
    supply, demand = simulator.generate_pairs(num_pairs=K, seed=12)
    train_demand, test_demand = simulator.generate_data(num_pairs=K, balan=bal)
else:
    simulator = RealData(num_pairs=K)
    supply = simulator.supply[:, :(K//2)]
    train_demand, test_demand = simulator.generate_data()

if args.val:
    dem = train_demand.copy()
else:
    dem = test_demand.copy()

assert "gsot" in method
plan = get_transport_plan(price=simulator.cost, supply=supply,
                        demand=dem, alpha=args.alpha, rho=args.rho, balance=bal)

edges = get_edges_from_plan(trans_plan=plan, max_num_edges=L)
with open(f"results/{save_as}/{M}_{N}_{K}_{L}_{seed}.pickle", "wb") as fp:
    pickle.dump(plan, fp)

values, net_profit = evaluate_net_profit(sorted_edges=edges, price=simulator.cost,
                                         supply=supply, demand=dem)
# plot_network(f"results/{save_as}/{method}/{M}_{N}_{K}_{L}_{seed}.pdf", M, N, edges)
logger.info(f"{net_profit}, {M}, {N}, {K}, {L}, {seed}")
