import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_mc(frame):
    global fig, ax
    rand_x = np.random.rand(frame * 1000 + 5)
    rand_y = np.random.rand(frame * 1000 + 5)
    labels = 1 * ((rand_x ** 2 + rand_y ** 2) < 1.0)
    colors = np.array([(0, 0, 255), (0, 255, 0)]) / 255.0
    ax.clear()
    sc = ax.scatter(rand_x, rand_y, 1 ** 2, marker=".", c=colors[labels])
    plt.title(
        f"$n$ : {frame*1000+5}, $\\pi$ : {(np.sum(labels)/(frame*1000+5))*4.:.4f}"
    )
    return sc


if __name__ == "__main__":
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, animate_mc, frames=50, interval=100)
    plt.show()
