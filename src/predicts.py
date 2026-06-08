import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# =========================
# FUNCTIONS
# =========================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =========================
# LOAD MNIST DATA
# =========================

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset loaded successfully!")


# =========================
# LOAD TRAINED WEIGHTS
# =========================

weights = np.load("models/mnist_nn_weights.npz")

W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]

print("Model weights loaded successfully!")


# =========================
# PREDICT MULTIPLE IMAGES
# =========================

num_images = 10

X_sample = X_test[:num_images]
y_sample = y_test[:num_images]

Z1 = X_sample @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

predictions = np.argmax(A2, axis=1)


# =========================
# PRINT RESULTS
# =========================

for i in range(num_images):
    print(
        f"Image {i}: "
        f"Predicted = {predictions[i]}, "
        f"Actual = {y_sample[i]}"
    )


# =========================
# SHOW IMAGES
# =========================

plt.figure(figsize=(12, 4))

for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_sample[i].reshape(28, 28), cmap="gray")

    title = f"Pred: {predictions[i]}\nActual: {y_sample[i]}"
    plt.title(title)

    plt.axis("off")

plt.tight_layout()
plt.show()