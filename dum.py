import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

sc = ax.plot(x, y)


def init():
    sc = ax.plot(x, y)
    return sc


def update_scatter(i):
    sc.set_data(x, y)
    return sc


anim = FuncAnimation(fig, update_scatter, init_func=init, frames=100, interval=20)
plt.show()
