import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time

fig, ax = plt.subplots()

def loss(y_true, y_pred):
    return (1./y_true.size)*np.sum((y_pred-y_true)**2)

def loss_grad(X, y_true, y_pred):
    return (2./y_true.size)*(X.T @ (y_pred - y_true) )

def clac_r2(y_true, y_pred):
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - y_true.mean()) ** 2)
    return (1. - u/v)

theta = np.random.randn(2, 1)
X = np.random.randn(30, 1)
X_concat = np.concatenate((np.ones((X.shape[0],1)), X), axis=1) # dim =  (30, 2)
y_true = X_concat@theta + 0.1*np.random.randn(30,1)

theta_est_list = []

def init():
    theta_est_list.append(np.random.randn(2,1))

def animate(i):
    if i >= 1000:
        return
    theta_est = theta_est_list[-1]
    y_pred = X_concat@theta_est
    theta_est = theta_est - 0.1*loss_grad(X_concat, y_true, y_pred)
    theta_est_list.clear()
    theta_est_list.append(theta_est)
    ax.clear()
    ax.plot(X, y_pred, color='r')
    ax.scatter(X, y_true, color='g')
    plt.title("Live Regression!")
    plt.xlabel("X")
    plt.ylabel("y")
    ax.text(0.5, 0.5, f"loss: {loss(y_true, y_pred):.4f}; R-squared: {clac_r2(y_true, y_pred):.4f}")

ani = FuncAnimation(fig, animate, init_func = init, interval=10)
plt.show()