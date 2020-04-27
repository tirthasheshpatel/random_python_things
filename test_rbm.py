import numpy as np
import matplotlib.pyplot as plt
import rbm
from keras.datasets import mnist

(X_train, y), (_, _) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(60000, -1)
X_train = 1. * ((X_train[:10000] / 255.) >= 0.5)

# Plot some training isntances
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
ax[0, 0].imshow(X_train[10].reshape(28, 28))
ax[0, 1].imshow(X_train[11].reshape(28, 28))
ax[0, 2].imshow(X_train[12].reshape(28, 28))
ax[1, 0].imshow(X_train[13].reshape(28, 28))
ax[1, 1].imshow(X_train[14].reshape(28, 28))
ax[1, 2].imshow(X_train[15].reshape(28, 28))
ax[2, 0].imshow(X_train[16].reshape(28, 28))
ax[2, 1].imshow(X_train[17].reshape(28, 28))
ax[2, 2].imshow(X_train[18].reshape(28, 28))
fig.suptitle("Training instances")
plt.show()

# We will mainly experiment with different latent space
# dimensions. For this instance, i have a 30-D latent space.
hidden_dims = 4

# Define our model
model = rbm.BinaryRestrictedBoltzmannMachine(hidden_dims)

# Train the model on our dataset with learning rate 0.01
model.fit(X_train, lr=2., burn_in=None, tune=1, epochs=50, verbose=True)

# Use the `decode()` method to generate an image.
images = [model.decode() for _ in range(9)]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
ax[0, 0].imshow(images[0].reshape(28, 28))
ax[0, 1].imshow(images[1].reshape(28, 28))
ax[0, 2].imshow(images[2].reshape(28, 28))
ax[1, 0].imshow(images[3].reshape(28, 28))
ax[1, 1].imshow(images[4].reshape(28, 28))
ax[1, 2].imshow(images[5].reshape(28, 28))
ax[2, 0].imshow(images[6].reshape(28, 28))
ax[2, 1].imshow(images[7].reshape(28, 28))
ax[2, 2].imshow(images[8].reshape(28, 28))
fig.suptitle("Generated instances")
plt.show()