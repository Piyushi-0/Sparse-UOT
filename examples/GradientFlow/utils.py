import imageio.v2 as imageio
import numpy as np
import torch
from random import choices
from sparse_ot.utils import seed


def load_image(fname):
    img = imageio.imread(fname, mode='L')  # Grayscale
    img = (img[::-1, :]) / 255.
    return 1 - img

def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]),
                         np.linspace(0, 1, A.shape[1]))

    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    seed()
    dots = np.array(choices(grid, dens, k=n))
    dots += (.5 / A.shape[0]) * np.random.standard_normal(dots.shape)

    return torch.from_numpy(dots).type(dtype)

def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")
