"""
This file is part of the "From Hogg To Gordon" project.
Copyright 2013 the authors.
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','size':12})
rc('text', usetex=True)
import pylab as plt
import numpy as np

max_val = 6.

def prior_x(xs):
    ps = np.zeros_like(xs).astype(float)
    I = (xs > 1.) * (xs < max_val)
    ps[I] = 1. / (xs[I])
    return ps

def prior_y(ys):
    ps = np.zeros_like(ys).astype(float)
    I = (ys > 1.) * (ys < max_val)
    ps[I] = np.exp(-0.5 * (ys[I] - 0.5 * max_val) ** 2 / 2. ** 2)
    return ps

def likelihood(xs, ys):
    return np.exp(-0.5 * ((xs - max_val + 2.5) ** 2 + (ys - max_val + 3.1) ** 2) / 1. ** 2)

def grid_likelihood(xs, ys, xgrid, ygrid):
    ms = proximity(xs, ys, xgrid, ygrid)
    return likelihood(xgrid[ms], ygrid[ms])

def proximity(xs, ys, xgrid, ygrid):
    """
    stupidly written to be extra slow!
    """
    xsf = xs.flatten()
    ysf = ys.flatten()
    msf = np.zeros_like(xsf).astype(int) - 1
    for i, (x, y) in enumerate(zip(xsf, ysf)):
        msf[i] = np.argmin((xgrid - x) ** 2 + (ygrid - y) ** 2)
    assert np.all(msf > -1)
    return msf.reshape(xs.shape)

def draw_points():
    return (0.5 + max_val * np.random.uniform(size=64), 
            0.5 + max_val * np.random.uniform(size=64))

if __name__ == "__main__":
    np.random.seed(1)
    xgrid, ygrid = draw_points()
    nx = 512
    ys, xs = 0.5 + max_val * (np.mgrid[0:nx,0:nx] + 0.5) / float(nx)
    priors_x = prior_x(xs)
    priors_y = prior_y(ys)
    priors = priors_x * priors_y
    likes = likelihood(xs, ys)
    glikes = grid_likelihood(xs, ys, xgrid, ygrid)
    gposts = priors * glikes
    print priors.shape, likes.shape, glikes.shape, gposts.shape
    kwargs = {"interpolation": "nearest", "vmin": 0., "origin": "lower", "extent": (0.5, max_val + 0.5, 0.5, max_val + 0.5)}
    xlim = [0.5, max_val + 0.5]
    plt.figure(figsize=(8, 8))
    plt.gray()
    plt.subplot(221)
    plt.imshow(likes, vmax=np.max(likes), **kwargs)
    plt.title("true likelihood")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.subplot(222)
    plt.imshow(glikes, vmax=np.max(likes), **kwargs)
    plt.plot(xgrid, ygrid, "rx")
    plt.title("approximate likelihood")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.subplot(223)
    plt.imshow(priors, vmax=np.max(priors), **kwargs)
    plt.title("prior")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.subplot(224)
    plt.imshow(gposts, vmax=np.max(gposts), **kwargs)
    plt.title("approximate posterior")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.savefig("voxels.png")
