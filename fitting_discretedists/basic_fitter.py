import numpy as np
from scipy.stats import binom
from scipy.optimize import brute, differential_evolution
import warnings

def func(free_params, *args):
    dist, x = args
    # NLL
    nll = -dist.logpmf(x, *free_params).sum()
    if np.isnan(nll):
        nll = np.inf
    return nll

def fit_discrete(dist, x, bounds, optimizer=brute):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return optimizer(func, bounds, args=(dist, x))

n, p = 5, 0.4
x = binom.rvs(n, p, size=10000)
bounds = [(0, 100), (0, 1)]
u2, s2 = fit_discrete(binom, x, bounds)
res = fit_discrete(binom, x, bounds, optimizer=differential_evolution)
print(u2, s2)
print(res.x)
