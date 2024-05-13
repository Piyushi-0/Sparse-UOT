import os
import torch
from torch import nn
import argparse
from sparse_ot.utils import createLogHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import GELU_

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Sparse MoE')
parser.add_argument('--lda3', type=float, default=0.1, help='lda3')
parser.add_argument('--max_itr', type=int, default=100, help='max_itr')
parser.add_argument('--lda', type=float, default=0.1, help='lda')
parser.add_argument('--ws', type=int, default=1, help='ws')
parser.add_argument('--ktype', type=str, default='I', help='ktype')
parser.add_argument('--khp', type=float, default=1.0, help='khp')
parser.add_argument('--num_experts', type=int, default=3, help='num_experts')
parser.add_argument('--num_epochs', type=int, default=100, help='num_epochs')
parser.add_argument('--alpha', type=float, default=1, help='alpha')
parser.add_argument('--method', type=str, default='single', help='method')
parser.add_argument('--data_x', default='xor')
parser.add_argument('--num_pts', default=220, type=int, help='num_pts')
args = parser.parse_args()

device = torch.device("cuda")
dtype = torch.float
num_epochs = args.num_epochs
alpha = args.alpha
method = args.method
num_pts = args.num_pts
data_x = args.data_x + str(num_pts)

if "topk" in method:
    from smoe_k import *
else:
    from smoe import *

G_e = torch.eye(args.num_experts, device=device, dtype=dtype)

os.makedirs("logs", exist_ok=True)
save_as = f"logs/{data_x}_{method}_{num_epochs}_{alpha}_{args.num_experts}"

logger_opti = None
save_as += f"_{args.lda3}_{args.max_itr}"
if "scot" in method or "prp" in method:
    logger_opti = createLogHandler(save_as+"_opti.csv", str(os.getpid())+str(os.getpid()))
if "prp" in method:
    save_as += f"_{args.lda}_{args.khp}_{args.ktype}_{args.ws}"
if "uotmmd" in method:
    save_as += f"_{args.lda}_{args.khp}_{args.ktype}"
if "uotkl" in method:
    save_as += f"_{args.lda}"
logger = createLogHandler(save_as+".csv", str(os.getpid()))

data_all, label_all = generate_data(num_pts)
data_all = data_all/data_all.max()

data_train_all, data_test, label_train_all, label_test = train_test_split(data_all, label_all, random_state=SEED, stratify=label_all, train_size=120)
data, data_val, label, label_val = train_test_split(data_train_all, label_train_all, random_state=SEED, stratify=label_train_all, train_size=100)

data= data.to(device)
data_val = data_val.to(device)
data_test = data_test.to(device)
label = label.to(device)
label_val = label_val.to(device)
label_test = label_test.to(device)

n, d = data.shape
c = torch.unique(label).shape[0]

moe = MoE(
    dim = d,
    dim_out=c,
    args = args,
    G_e = G_e,
    hidden_dim = d * 4,
    activation = nn.GELU if hasattr(nn, 'GELU') else GELU_,
    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
    loss_coef = alpha,              # multiplier on the auxiliary expert balancing auxiliary loss
    method = method,
    logger=logger_opti
)
moe.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(moe.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

label_oh = F.one_hot(label, num_classes=c).to(device)

best_acc = -torch.inf

train_tot = data.shape[0]
test_tot = data_test.shape[0]
val_tot = data_val.shape[0]

for epoch in range(num_epochs):
    out, loss_1, _, _ = moe(data.view(1, n, d))
    out = out.squeeze(0)
    loss_2 = criterion(out, label_oh.to(dtype))
    loss = loss_1 + alpha*loss_2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    with torch.no_grad():
        moe.eval()
        acc = (torch.sum(torch.argmax(out, dim=1)==label)/train_tot).item()
        # print(loss.item(), acc)
        logger.info(f"{epoch}, {loss_1.item()}, {loss_2.item()}, {acc}")
        
        out_val, _, cnt1, cnt2 = moe(data_val.view(1, data_val.shape[0], data_val.shape[1]))
        out_val = out_val.squeeze(0)
        acc = torch.sum(torch.argmax(out_val, dim=1)==label_val)/val_tot
        if acc > best_acc:
            best_acc = acc
            torch.save(moe.state_dict(), save_as+".pt")
        cnt1 = cnt1.squeeze(0)
        cnt2 = cnt2.squeeze(0)
        logger.info(f"VAL, {epoch}, {acc.item()}, {cnt1[0]}, {cnt1[1]}, {cnt1[2]}, {cnt2[0]}, {cnt2[1]}, {cnt2[2]}")
    moe.train()
logger.info("Best Val Acc: "+str(best_acc.item()))
# moe.eval()
test_tot = data_test.shape[0]

moe.load_state_dict(torch.load(save_as+".pt"))
moe.eval()
with torch.no_grad():
    out_test, _, cnt1, cnt2 = moe(data_test.view(1, data_test.shape[0], data_test.shape[1]))
    out_test = out_test.squeeze(0)
    pred = torch.argmax(out_test, dim=1)
    acc = torch.sum(pred==label_test)/test_tot
    cnt1 = cnt1.squeeze(0)
    cnt2 = cnt2.squeeze(0)
    auc = roc_auc_score(label_test.cpu().numpy(), pred.cpu().numpy())
    logger.info(f"Test, {acc.item()}, {auc.item()}, {cnt1[0]}, {cnt1[1]}, {cnt1[2]}, {cnt2[0]}, {cnt2[1]}, {cnt2[2]}")
