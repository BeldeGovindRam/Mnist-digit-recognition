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
# FUNCTIONS
# =========================

def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


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
# INITIALIZE PARAMETERS
# =========================

np.random.seed(42)

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

learning_rate = 0.1
epochs = 10
batch_size = 64


# =========================
# TRAINING LOOP
# =========================

num_samples = X_train.shape[0]

for epoch in range(epochs):

    # Shuffle training data every epoch
    indices = np.random.permutation(num_samples)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        y_batch_one_hot = one_hot_encode(y_batch)

        m = X_batch.shape[0]

        # Forward propagation
        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)

        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Loss
        loss = cross_entropy_loss(y_batch_one_hot, A2)
        epoch_loss += loss

        # Backpropagation
        dZ2 = A2 - y_batch_one_hot

        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)

        dW1 = (X_batch.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

    # Evaluate after each epoch
    Z1_train = X_train @ W1 + b1
    A1_train = relu(Z1_train)
    Z2_train = A1_train @ W2 + b2
    A2_train = softmax(Z2_train)

    train_acc = accuracy(y_train, A2_train)

    average_loss = epoch_loss / (num_samples / batch_size)

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {average_loss:.4f}, "
        f"Training Accuracy: {train_acc:.4f}"
    )


# =========================
# TEST EVALUATION
# =========================

Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)

Z2_test = A1_test @ W2 + b2
A2_test = softmax(Z2_test)

test_acc = accuracy(y_test, A2_test)

print("\nTraining complete!")
print(f"Test accuracy: {test_acc:.4f}")

# =========================
# SAVE MODEL PARAMETERS
# =========================

np.savez(
    "models/mnist_nn_weights.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("Model weights saved to models/mnist_nn_weights.npz")