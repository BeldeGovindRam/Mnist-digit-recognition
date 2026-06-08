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
