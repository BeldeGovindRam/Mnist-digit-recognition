import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# =========================
# LOAD MNIST DATASET
# =========================

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset loaded successfully!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# =========================
# ONE-HOT ENCODE TRAIN LABELS
# =========================

y_train_one_hot = np.zeros((y_train.shape[0], 10))
y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1


# =========================
# INITIALIZE PARAMETERS
# =========================

np.random.seed(42)

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

learning_rate = 0.1
epochs = 100


from model import (
    relu,
    relu_derivative,
    softmax,
    cross_entropy_loss,
    accuracy
)

# =========================
# TRAINING LOOP
# =========================

m = X_train.shape[0]

for epoch in range(epochs):

    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss and training accuracy
    loss = cross_entropy_loss(y_train_one_hot, A2)
    train_acc = accuracy(y_train, A2)

    # Backpropagation
    dZ2 = A2 - y_train_one_hot

    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = (X_train.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}")


# =========================
# FINAL TRAIN EVALUATION
# =========================

Z1_train = X_train @ W1 + b1
A1_train = relu(Z1_train)

Z2_train = A1_train @ W2 + b2
A2_train = softmax(Z2_train)

final_train_loss = cross_entropy_loss(y_train_one_hot, A2_train)
final_train_acc = accuracy(y_train, A2_train)


# =========================
# TEST EVALUATION
# =========================

Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)

Z2_test = A1_test @ W2 + b2
A2_test = softmax(Z2_test)

test_acc = accuracy(y_test, A2_test)


# =========================
# RESULTS
# =========================

print("\nTraining complete!")
print(f"Final training loss: {final_train_loss:.4f}")
print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")