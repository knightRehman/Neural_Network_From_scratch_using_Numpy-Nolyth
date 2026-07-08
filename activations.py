"""
This file Contains Activation Functions for Neural Network

Author: Wasi-Ur-Rehman

"""

import numpy as np


# ReLU Activation Function
def relu(Z):
    """Apply ReLU activation."""
    return np.maximum(0, Z)


# ReLU Derivative
def relu_derivative(Z):
    """Derivative of ReLU."""
    return (Z > 0).astype(float)


# Sigmoid Activation Function
def sigmoid(Z):
    """Apply Sigmoid activation."""
    return 1 / (1 + np.exp(-Z))


# Sigmoid Derivative
def sigmoid_derivative(A):
    """Derivative of Sigmoid."""
    return A * (1 - A)