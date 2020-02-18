import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def sample_x2(mu, sigma, x1):
    new_mu = mu[1] + sigma[1][0] / sigma[0][0] * (x1 - mu[0])
    new_sigma = sigma[1][1] + (sigma[0][1] ** 2) / sigma[0][0]
    return norm.rvs(new_mu, new_sigma ** 0.5, size=1)


def sample_x1(mu, sigma, x2):
    new_mu = mu[0] + sigma[0][1] / sigma[1][1] * (x2 - mu[1])
    new_sigma = sigma[0][0] + (sigma[1][0] ** 2) / sigma[1][1]
    return norm.rvs(new_mu, new_sigma ** 0.5, size=1)


class AnimateGibbs:
    def __init__(self):
        self.mu = np.array([35.0, 40.0])
        self.sigma = np.array([[4.0, 0.0], [0.0, 4.0]])
        self.fig, self.ax = plt.subplots()
        self.x1, self.x2 = 0.0, 0.0
        self.sample_history = []
        xs = np.linspace(30.0, 50.0, num=500)
        ys = np.linspace(30.0, 50.0, num=500)
        self.xm, self.ym = np.meshgrid(xs, ys)
        self.grid = np.empty((500, 500, 2))
        self.grid[:, :, 0] = self.xm
        self.grid[:, :, 1] = self.ym
        self.zm = multivariate_normal(self.mu, self.sigma).pdf(self.grid)
        self.cf = self.ax.contourf(self.xm, self.ym, self.zm, levels=20, alpha=0.6)
        self.burn(0)

    def burn(self, steps):
        for _ in range(steps):
            self.x1 = sample_x1(self.mu, self.sigma, self.x2)
            self.x2 = sample_x2(self.mu, self.sigma, self.x1)

    def animate(self, frame):
        self.x1 = sample_x1(self.mu, self.sigma, self.x2)
        self.x2 = sample_x2(self.mu, self.sigma, self.x1)
        # self.x1, self.x2 = tuple(multivariate_normal(self.mu, self.sigma).rvs(size=1))
        self.sc = self.ax.scatter(self.x1, self.x2, color="b")
        self.sample_history.append([self.x1, self.x2])
        plt.title(
            f"Gibb's Sampling\n$\\mu$ : {self.mu}, $\\hat \\mu$ : {np.round(np.mean(self.sample_history, axis=0), 2)}"
        )
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
        return (self.sc,)

    def run(self, frames, interval, save=False):
        anim = FuncAnimation(self.fig, self.animate, frames=frames, interval=interval)
        plt.show()


if __name__ == "__main__":
    gibb = AnimateGibbs()
    gibb.run(10000, 1)
