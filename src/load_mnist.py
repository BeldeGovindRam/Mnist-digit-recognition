import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# =========================
# LOAD MNIST DATASET
# =========================

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset loaded successfully!")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("First image shape before reshape:", X[0].shape)
print("First label:", y[0])


# =========================
# NORMALIZE PIXEL VALUES
# =========================

X = X / 255.0

print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())


# =========================
# VISUALIZE ONE IMAGE
# =========================

first_image = X[0].reshape(28, 28)

plt.imshow(first_image, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.axis("off")
plt.show()

image = X[0].reshape(1, 784)

print(image.shape)