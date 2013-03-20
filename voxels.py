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
    return np.exp(-0.5 * ((xs - max_val + 2.5) ** 2 + (ys - max_val + 3.1) ** 2) / 1.5 ** 2)

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
    nx = 8
    amp = 1. / 6.
    ygrid, xgrid = 0.5 + max_val * (np.mgrid[0:nx,0:nx] + 0.5) / float(nx)
    xgrid = xgrid.flatten() + amp * np.random.normal(size=nx * nx)
    ygrid = ygrid.flatten() + amp * np.random.normal(size=nx * nx)
    return xgrid, ygrid

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
    marg_likes = np.sum(likes * priors_y, axis=0)
    marg_glikes = np.sum(glikes * priors_y, axis=0)
    marg_posts = np.sum(likes * priors_x * priors_y, axis=0)
    marg_gposts = np.sum(glikes * priors_x * priors_y, axis=0)
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
    plt.savefig("voxels.pdf")
    plt.savefig("voxels.eps")
    plt.figure(figsize=(8, 8))
    plt.gray()
    plt.subplot(221)
    plt.plot(xs[0], marg_likes, 'k-')
    plt.axhline(color="k", alpha=0.5)
    plt.title("true marginalized likelihood")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(-0.1 * np.max(marg_likes), 1.1 * np.max(marg_likes))
    plt.subplot(222)
    plt.plot(xs[0], marg_glikes, 'k-')
    plt.plot(xs[0], marg_likes, 'k-', alpha=0.5)
    plt.axhline(color="k", alpha=0.5)
    plt.title("approximate marginalized likelihood")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(-0.1 * np.max(marg_likes), 1.1 * np.max(marg_likes))
    plt.subplot(223)
    plt.plot(xs[0], marg_posts, 'k-')
    plt.axhline(color="k", alpha=0.5)
    plt.title("true marginalized posterior")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(-0.1 * np.max(marg_posts), 1.1 * np.max(marg_posts))
    plt.subplot(224)
    plt.plot(xs[0], marg_gposts, 'k-')
    plt.plot(xs[0], marg_posts, 'k-', alpha=0.5)
    plt.axhline(color="k", alpha=0.5)
    plt.title("approximate marginalized posterior")
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlim(xlim)
    plt.ylim(-0.1 * np.max(marg_posts), 1.1 * np.max(marg_posts))
    plt.savefig("voxels2.png")
    plt.savefig("voxels2.pdf")
    plt.savefig("voxels2.eps")
