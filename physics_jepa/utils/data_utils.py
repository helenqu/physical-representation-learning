from einops import rearrange
import torch
from skimage.transform import resize
import numpy as np
from PIL import Image
import torch.nn.functional as F

def normalize_labels(x, stats={}):
    if 'mins' in stats and 'maxes' in stats:
        mins = torch.tensor(stats['mins'])
        maxes = torch.tensor(stats['maxes'])
        return (x - mins) / (maxes - mins)
    elif 'means' in stats and 'stds' in stats:
        if 'compression' in stats:
            for i, compression_type in enumerate(stats['compression']):
                if compression_type == 'log':
                    x[:, i] = torch.log10(x[:, i])
                elif compression_type == None:
                    continue
        means = torch.tensor(stats['means'])
        stds = torch.tensor(stats['stds'])
        return (x - means) / stds
    else:
        return x

def subsample(x, reso):
    """ Subsample a numpy array (or cpu tensor) to a given resolution 
    using antialiasing Gaussian filter. """
    ndim = len(reso)
    output_shape = x.shape[:-ndim] + tuple(reso)
    if any(output_shape[d] > x.shape[d] for d in range(-ndim, 0)):
        return x
    if output_shape == x.shape:
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(resize(x, output_shape, anti_aliasing=True))
    elif isinstance(x, torch.Tensor):
        x = x.numpy()
        x = resize(x, output_shape, anti_aliasing=True)
        return torch.tensor(x)

def mse(x, y):
    loss = (x - y).pow(2).mean()
    return {"loss": loss}

def mae(x, y):
    loss = (x - y).abs().mean()
    return {"loss": loss}