import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# LOAD MNIST DATASET
# print("Loading MNIST dataset...")
# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# X = mnist.data
# y = mnist.target.astype(int)
# print("Dataset loaded successfully!")
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("First image shape before reshape:", X[0].shape)
# print("First label:", y[0])

# # NORMALIZE PIXEL VALUES
# X = X / 255.0
# print("Minimum pixel value:", X.min())
# print("Maximum pixel value:", X.max())
# # VISUALIZE ONE IMAGE

# first_image = X[0].reshape(28, 28)
# plt.imshow(first_image, cmap="gray")
# plt.title(f"Label: {y[0]}")
# plt.axis("off")
# plt.show()
# image = X[0].reshape(1, 784)
# print(image.shape)


# LOAD MNIST DATASET
print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset loaded successfully!")

# Normalize pixel values
X = X / 255.0

print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# INITIALIZE PARAMETERS
# =========================

np.random.seed(42)

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)


# =========================
# ACTIVATION FUNCTIONS
# =========================

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))


def accuracy(y_true_labels, y_pred_probs):
    predictions = np.argmax(y_pred_probs, axis=1)
    return np.mean(predictions == y_true_labels)
# =========================
# FORWARD PROPAGATION
# =========================

Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

print("Z1 shape:", Z1.shape)
print("A1 shape:", A1.shape)
print("Z2 shape:", Z2.shape)
print("A2 shape:", A2.shape)

print("First image predicted probabilities:")
print(A2[0])

print("Sum of probabilities for first image:")
print(np.sum(A2[0]))

prediction = np.argmax(A2[0])

print("Prediction for first image:", prediction)
print("Actual label for first image:", y[0])

y_one_hot = one_hot_encode(y)

loss = cross_entropy_loss(y_one_hot, A2)
acc = accuracy(y, A2)

print("Initial loss:", loss)
print("Initial accuracy:", acc)