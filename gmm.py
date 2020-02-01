import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from matplotlib.animation import FuncAnimation
np.random.seed(9000)

def step(X, dist1, dist2, **kwargs):
    mu1 = kwargs['mu1']
    sig1 = kwargs['sig1']
    mu2 = kwargs['mu2']
    sig2 = kwargs['sig2']

    Z = dist1.pdf(X) * 0.5 + dist2.pdf(X) * 0.5

    posterior_dist1 = dist1.pdf(X) * 0.5 / Z
    posterior_dist2 = dist2.pdf(X) * 0.5 / Z

    mu1_new = np.sum(X * posterior_dist1) / np.sum(posterior_dist1)
    mu2_new = np.sum(X * posterior_dist2) / np.sum(posterior_dist2)

    sig1_new = np.sum((X - mu1_new) ** 2 * posterior_dist1) / np.sum(posterior_dist1)
    sig2_new = np.sum((X - mu2_new) ** 2 * posterior_dist2) / np.sum(posterior_dist2)

    dist1_new = norm(mu1_new, sig1_new ** 0.5)
    dist2_new = norm(mu2_new, sig2_new ** 0.5)

    return mu1_new, sig1_new, mu2_new, sig2_new, dist1_new, dist2_new

def plot_me(X, gauss1, gauss2):
    global ax, plot1, plot2, sc1, sc2
    y = np.linspace(-10., 10., num=1000)
    normalize_ = gauss1.pdf(X) + gauss2.pdf(X)
    opacities_gauss1 = gauss1.pdf(X) / normalize_
    opacities_gauss2 = gauss2.pdf(X) / normalize_
    ax.clear()
    plot1, = ax.plot(y, gauss1.pdf(y), color='r', label=f'$N({mu1:.2f}, {sig1:.2f})$')
    plot2, = ax.plot(y, gauss2.pdf(y), color='g', label=f'$N({mu2:.2f}, {sig2:.2f})$')
    for i in range(len(X)):
        sc1 = plt.scatter(X[i], 0, color='r', alpha=opacities_gauss1[i])
        sc2 = plt.scatter(X[i], 0, color='g', alpha=opacities_gauss2[i])
    plt.legend()

def init():
    global X, mu1, sig1, mu2, sig2, gauss1, gauss2, ax
    X = np.concatenate((norm(-4.0, 3.0).rvs(10).reshape(-1,1), norm(5.0, 1.0).rvs(10).reshape(-1,1)), axis=0).reshape(-1,)
    mu1, sig1, mu2, sig2 = np.random.rand(), 0.5*np.random.rand(), np.random.rand(), 0.5*np.random.rand()
    gauss1, gauss2 = norm(mu1, sig1), norm(mu2, sig2)
    plot_me(X, gauss1, gauss2)
    return plot1, plot2, sc1, sc2

def animate(frame):
    global X, mu1, sig1, mu2, sig2, gauss1, gauss2
    mu1, sig1, mu2, sig2, gauss1, gauss2 = step(X, gauss1, gauss2, mu1=mu1, sig1=sig1, mu2=mu2, sig2=sig2)
    plot_me(X, gauss1, gauss2)
    return plot1, plot2, sc1, sc2

if __name__ == "__main__":
    global fig, ax
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, animate, init_func=init, blit=True, frames=50, interval=200)
    anim.save('/mnt/c/users/tirth/desktop/oop in python/gmm1d_animation_large_tmp.mp4')
