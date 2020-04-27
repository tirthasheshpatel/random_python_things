import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
X = np.linspace(-5, 5, 100).reshape(-1, 1)

class SquareSystem(object):
    def __init__(self, p):
        self.p = p

    def _validate_x(self, x):
        if len(x.shape) > 1 and x.shape[0] != self.p:
            raise ValueError("expected x of shape ({}, ). got {}".format(self.p, x.shape))

    def __call__(self, x, multidim=False):
        if multidim:
            res = np.empty(x.shape[:-1] + (self.p, ), dtype=x.dtype)
            for i in range(self.p):
                res[..., i] = np.sum(x ** (i+2), axis=-1)
            return res
        self._validate_x(x)
        res = np.empty((self.p, 1), dtype=x.dtype)
        for i in range(self.p):
            res[i, :] = np.sum(x ** (i+2), axis=-1)
        return res

    def jacobian(self, x):
        self._validate_x(x)
        res = np.empty((self.p, x.shape[0]), dtype=x.dtype)
        for i in range(self.p):
            res[i, :] = (i+2) * x ** (i+1)
        return res

    def optimize(self, init_x=None, n_iter=5, *args, **kwargs):
        if init_x == None:
            init_x = np.random.randn(self.p)
        x = init_x
        # for i in range(n_iter):
        func = self.__call__(x)
        # ax.clear()
        if not hasattr(self, "plot1"):
            self.plot1 = ax.plot(X, self.__call__(X, multidim=True))
        self.sc1 = ax.scatter(x, func)
        jac  = self.jacobian(x)
        dx   = np.linalg.solve(jac, -func)
        x = x + dx.reshape(-1, )
        return self.plot1, self.sc1

class OverconstrainedSystem(object):
    def __init__(self, p):
        self.p = p

    def _validate_x(self, x):
        if len(x.shape) > 1:
            raise ValueError("expected x of shape (q, ). got {}".format(x.shape))

func = SquareSystem(1)
anim = FuncAnimation(fig, functools.partial(func.optimize, None, 50), frames=100, interval=500)
plt.show()
