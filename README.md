# MNIST Digit Recognition From Scratch Using NumPy

## Project Overview

This project implements a complete neural network from scratch using only NumPy to recognize handwritten digits from the MNIST dataset.

The goal of this project was not simply to achieve high accuracy, but to deeply understand how neural networks work internally by implementing every major component manually instead of using frameworks such as PyTorch or TensorFlow.

This project covers:

* Data loading and preprocessing
* Forward propagation
* ReLU activation
* Softmax activation
* Cross Entropy Loss
* Backpropagation
* Gradient Descent
* Mini-Batch Gradient Descent
* Model saving and loading
* Prediction and visualization

Final Test Accuracy:

```text
~97%
```

---

# Dataset

This project uses the MNIST handwritten digit dataset.

Dataset Characteristics:

```text
70,000 images
28 x 28 grayscale images
10 classes (0-9)
```

Each image contains:

```text
28 x 28 = 784 pixels
```

The original image is flattened into a vector:

```text
28 x 28
↓
784
```

Example:

```text
Image
↓
[0, 0, 0, 255, 125, ...]
```

Pixel values are normalized:

```text
0-255
↓
0-1
```

---

# Neural Network Architecture

The final model architecture is:

```text
Input Layer
784 neurons
(one neuron per pixel)

        ↓

Hidden Layer
128 neurons
ReLU Activation

        ↓

Output Layer
10 neurons
(one neuron per digit)

        ↓

Softmax
```

Output example:

```text
[0.01, 0.02, 0.90, 0.01, ...]
```

The highest probability becomes the prediction.

---

# Mathematical Flow

## Layer 1

Input to Hidden Layer:

Z1 = XW1 + b1

Shapes:

```text
X   = (batch_size, 784)
W1  = (784, 128)
b1  = (1, 128)

Z1  = (batch_size, 128)
```

---

## ReLU Activation

ReLU introduces non-linearity.

Formula:

```text
ReLU(x) = max(0, x)
```

Example:

```text
[-2, -1, 0, 3, 5]
↓
[0, 0, 0, 3, 5]
```

---

## Layer 2

Hidden Layer to Output Layer:

Z2 = A1W2 + b2

Shapes:

```text
A1 = (batch_size, 128)
W2 = (128, 10)
b2 = (1, 10)

Z2 = (batch_size, 10)
```

---

## Softmax

Softmax converts raw scores into probabilities.

Example:

```text
Scores:
[2.1, 5.0, 1.2]

↓

Probabilities:
[0.05, 0.90, 0.05]
```

All probabilities sum to:

```text
1.0
```

---

## Loss Function

Cross Entropy Loss was used.

Purpose:

```text
Measure how wrong the predictions are.
```

Lower loss means better predictions.

---

## Backpropagation

Backpropagation computes gradients for:

```text
W1
b1
W2
b2
```

The gradients indicate:

```text
How much each parameter contributed to the error.
```

---

## Gradient Descent

Parameters are updated using:

```text
Parameter = Parameter - Learning Rate × Gradient
```

This gradually reduces loss and improves accuracy.

---

# Training Versions

## Version 1: Batch Gradient Descent

File:

```text
src/train_nn.py
```

Process:

```text
All training images
↓
Forward Pass
↓
Backpropagation
↓
Single Weight Update
```

Result:

```text
~78% Test Accuracy
```

---

## Version 2: Mini-Batch Gradient Descent

File:

```text
src/train_nn_minibatch.py
```

Process:

```text
64 Images
↓
Forward Pass
↓
Backpropagation
↓
Weight Update

Repeat...
```

Advantages:

* Faster learning
* More frequent updates
* Better convergence
* Industry standard approach

Result:

```text
~97% Test Accuracy
```

---

# Project Structure

```text
Mnist-digit-recognition/

│
├── data/
│
├── models/
│   └── mnist_nn_weights.npz
│
├── notebooks/
│
├── src/
│   ├── load_mnist.py
│   ├── train_nn.py
│   ├── train_nn_minibatch.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# File Descriptions

## load_mnist.py

Purpose:

```text
Understand the dataset.
```

Tasks:

* Load MNIST
* Normalize images
* Inspect shapes
* Visualize digits

---

## train_nn.py

Purpose:

```text
First neural network implementation.
```

Uses:

```text
Batch Gradient Descent
```

Mainly created to understand the training process.

---

## train_nn_minibatch.py

Purpose:

```text
Final training implementation.
```

Uses:

```text
Mini-Batch Gradient Descent
```

This file trains the model and saves weights.

Output:

```text
models/mnist_nn_weights.npz
```

---

## predict.py

Purpose:

```text
Use trained model for inference.
```

Tasks:

* Load saved weights
* Predict digits
* Visualize predictions
* Compare predicted vs actual labels

---

# Key Learnings

Through this project I learned:

* Matrix multiplication in neural networks
* Shape calculations
* Weight initialization
* ReLU activation
* Softmax activation
* Cross Entropy Loss
* Forward propagation
* Backpropagation
* Gradient Descent
* Mini-Batch Gradient Descent
* Train/Test split
* Model persistence
* Prediction pipelines

Most importantly, I learned how neural networks actually learn rather than simply calling framework APIs.

---

# Future Improvements

Possible extensions:

* Multiple hidden layers
* Dropout
* L2 Regularization
* Adam Optimizer
* Learning Rate Scheduling
* Confusion Matrix
* CNN implementation
* PyTorch implementation
* TensorFlow implementation

---

# Final Notes

This project was built as a learning exercise to understand the mathematical foundations of neural networks before moving on to:

```text
CNNs
Attention
Transformers
LLMs
```

The objective was not merely to classify digits but to understand every major step involved in training a neural network from scratch.
