"""
This file contains Binary Cross Entropy Loss Function.

Author: Wasi-Ur-Rehman
"""

import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Calculate Binary Cross Entropy Loss.
    """

    # Prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss