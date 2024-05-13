import torch
import math
from inspect import isfunction
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs

def generate_d2():
    n_samples = [66, 225, 225]
    n_features = 2
    cluster_std = [0.9, 0.9, 0.5]
    centers = [(0,1), (3,3), (1,4)]
    data, labels = make_blobs(n_samples, n_features = n_features, centers=centers, cluster_std=cluster_std)
    return data, labels
    

def generate_data(n_samples=100):
    X = torch.zeros(n_samples, 2)
    y = torch.zeros(n_samples, dtype=torch.long)

    # Generate samples from two Gaussian distributions
    X[:n_samples//2] = torch.randn(n_samples//2, 2) + torch.Tensor([3, 2])
    X[n_samples//2:] = torch.randn(n_samples//2, 2) + torch.Tensor([-3, 2])

    # Labels
    for i in range(X.shape[0]):
        if X[i].norm() > math.sqrt(13):
            y[i] = 1

    X[:, 1] = X[:, 1] - 2

    return X, y

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

