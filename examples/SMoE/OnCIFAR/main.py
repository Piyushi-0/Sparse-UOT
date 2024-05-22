'''
implementation of resnet18 and mobilenet follows https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as torchmodels

import os
import argparse

from utils import progress_bar
from utils import get_config
import moe_prp as moe
import resnet
from PIL import Image
from utils import entropy
import numpy as np
import random
import supported
from sparse_ot.utils import createLogHandler
from tqdm import tqdm

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

config = get_config()
EXPERT_NUM = config['experts']
CLUSTER_NUM = config['clusters']
strategy = config['strategy']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model', default='resnet18')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--mixture', default=True)
parser.add_argument('--ktype', default="imq")
parser.add_argument('--khp', default=-1, type=float)
parser.add_argument('--lda3', default=1, type=float)
parser.add_argument('--v', default=0)
parser.add_argument('--test',  action='store_true')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint', default=False)
parser.add_argument('--save_as', default='prp')
args = parser.parse_args()

save_as = args.save_as + str(args.gamma) + str(args.model)

if 'scot' in save_as:
    import moe_scot as moe
elif 'prp' in save_as:
    import moe_prp as moe
    save_as += args.ktype + str(args.khp) + str(args.lda3)
# elif 'ot' in save_as:
#     import moe_ot as moe
# elif 'uotkl' in save_as:
#     import moe_uotkl as moe
#     save_as += str(args.lda3)
# elif 'uotmmd' in save_as:
#     import moe_uotmmd as moe
#     save_as += args.ktype + str(args.khp) + str(args.lda3)
# elif 'ssot' in save_as:
#     import moe_ssot as moe
elif 'topk' in save_as or 'single' in save_as:
    import moe

log_fldr = "LOGS"

if args.test:
    log_fldr += "_test_"

os.makedirs(f"{log_fldr}/"+save_as, exist_ok=True)
fname = f"{log_fldr}/"+save_as+"/log.csv"

logger = createLogHandler(fname, str(os.getpid()))

device = "cuda"
best_acc = 0  # best test accuracy
best_acc_list = []
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_rotate_train = transforms.Compose([
    torchvision.transforms.RandomRotation((30,30)),
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_rotate_test = transforms.Compose([
    torchvision.transforms.RandomRotation((30,30)),
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Create trainset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

# Create cluster and targets
trainset.targets = torch.tensor(trainset.targets)
trainset.cluster = trainset.targets
trainset.targets = torch.zeros_like(trainset.targets)

# trainset negative examples
trainset_flip = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_rotate_train)
# Cluster and targets
trainset_flip.targets = torch.tensor(trainset_flip.targets)
trainset_flip.cluster = trainset_flip.targets
trainset_flip.targets = torch.ones_like(trainset_flip.targets)

trainset = torch.utils.data.ConcatDataset([trainset,trainset_flip])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         shuffle=True, num_workers=2,
                                         worker_init_fn=seed_worker,generator=g,)

# Testset cluster and targets
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testset.targets = torch.tensor(testset.targets)
testset.cluster = testset.targets
testset.targets = torch.zeros_like(testset.targets)

# Testset negative
testset_flip = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_rotate_test)
testset_flip.targets = torch.tensor(testset_flip.targets)
testset_flip.cluster = testset_flip.targets
testset_flip.targets = torch.ones_like(testset_flip.targets)

testset = torch.utils.data.ConcatDataset([testset,testset_flip])
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2,
                                         worker_init_fn=seed_worker,generator=g,)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/2layer_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Training
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for optim in optimizers:
            optim.zero_grad()

        if args.mixture:
            outputs,_,loss,_, _ = net(inputs)
            loss = criterion(outputs, targets) + 0.01*loss
        else:
            if args.model == 'resnet18':
                outputs,_ = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)
        loss.backward()

        for optim in optimizers:
            optim.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    mask_cnt = torch.zeros(EXPERT_NUM)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.mixture:
                outputs,select0, _, _, mask_count = net(inputs)
            else:
                outputs = net(inputs)

            loss = criterion(outputs, targets)
            
            mask_cnt += mask_count.view(-1).cpu()
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
    # Save checkpoint.
    acc = 100.*correct/total
    logger.info(f"{acc}, {best_acc, {mask_cnt[0].item()}, {mask_cnt[1].item()}, {mask_cnt[2].item()}}")
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer0': optimizers[0].state_dict(),
            'optimizer1': optimizers[1].state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if args.mixture:
    if 'topk' not in save_as:
        if 'resnet' in args.model:
            net = moe.NonlinearMixtureMobile(EXPERT_NUM, gamma=args.gamma, strategy=strategy, v=args.v, ktype=args.ktype, khp=args.khp, lda3=args.lda3).to(device)
        else:
            net = moe.NonlinearMixtureRes(EXPERT_NUM, gamma=args.gamma, strategy=strategy, v=args.v, ktype=args.ktype, khp=args.khp, lda3=args.lda3).to(device)
    else: # NOTE: due to the above typo, we report MobileNet's result for topk.
        if 'resnet' in args.model:
            net = moe.NonlinearMixtureRes(EXPERT_NUM, strategy=strategy).to(device)
        else:
            net = moe.NonlinearMixtureMobile(EXPERT_NUM, strategy=strategy).to(device)

    optimizer = moe.NormalizedGD(net.models.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net.router.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=5e-4)
    optimizers = [optimizer,optimizer2]
else:
    if 'resnet' in args.model:
        net = resnet.ResNet18().to(device)
    else:
        net = mobilenet.MobileNetV2().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    optimizers = [optimizer]
EPOCHS=args.epochs

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

ent_list, acc_list = [], []

for epoch in range(start_epoch, start_epoch+EPOCHS):
    train(epoch)
    test(epoch)
    scheduler.step()
