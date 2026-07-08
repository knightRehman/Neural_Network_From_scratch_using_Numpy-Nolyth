"""
Neural_Network_from_Scratch

Author: Wasi-Ur-Rehman
Description:
This file implements a simple feedforward neural network with:
- Weight Initialization
- Forward Propagation
- Backpropagation
- Gradient Descent
"""

import numpy as np
from activations import relu, relu_derivative, sigmoid


class NeuralNetwork:
    """Simple Neural Network for Binary Classification."""

    def __init__(self):
        """Initialize weights and biases."""

        np.random.seed(42)

        # Weights and biases between Input Layer and Hidden Layer
        self.W1 = np.random.randn(2, 4) * 0.01
        self.b1 = np.zeros((1, 4))

        # Weights and biases between Hidden Layer and Output Layer
        self.W2 = np.random.randn(4, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        """
        Perform Forward Propagation.

        Parameters:
            X : Input features

        Returns:
            Predicted probabilities
        """

        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y, learning_rate=0.01):
        """
        Perform Backpropagation and update weights.

        Parameters:
            X : Training data
            y : True labels
            learning_rate : Step size for gradient descent
        """

        m = X.shape[0]

        # Output layer gradients
        dZ2 = self.A2 - y.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)

        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1