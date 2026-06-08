import numpy as np
from sklearn.datasets import fetch_openml


# =========================
# LOAD MNIST DATASET
# =========================

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

y_one_hot = np.zeros((y.shape[0], 10))
y_one_hot[np.arange(y.shape[0]), y] = 1

print("Dataset loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# PARAMETERS
# =========================

np.random.seed(42)

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

learning_rate = 0.1
epochs = 100


# =========================
# FUNCTIONS
# =========================

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return z > 0


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))


def accuracy(y_true_labels, y_pred_probs):
    predictions = np.argmax(y_pred_probs, axis=1)
    return np.mean(predictions == y_true_labels)


# =========================
# TRAINING LOOP
# =========================

m = X.shape[0]

for epoch in range(epochs):

    # Forward propagation
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss and accuracy
    loss = cross_entropy_loss(y_one_hot, A2)
    acc = accuracy(y, A2)

    # Backpropagation
    dZ2 = A2 - y_one_hot
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")


# =========================
# FINAL RESULT
# =========================

Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = softmax(Z2)

final_loss = cross_entropy_loss(y_one_hot, A2)
final_acc = accuracy(y, A2)

print("\nTraining complete!")
print("Final loss:", final_loss)
print("Final accuracy:", final_acc)