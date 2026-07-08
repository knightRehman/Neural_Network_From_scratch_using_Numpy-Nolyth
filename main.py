"""
This file Contains training and evaluating the neural network on a generated dataset.

Author: Wasi-Ur-Rehman
"""

import numpy as np
from utils import generate_data
from neural_network import NeuralNetwork
from loss import binary_cross_entropy

# Generate Dataset
X, y = generate_data()

# Shuffle Dataset
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Train-Test Split (80% Train, 20% Test)
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# Create Neural Network
model = NeuralNetwork()

# Number of training epochs
epochs = 10000

# Training Loop
for epoch in range(epochs):

    # Forward Propagation
    predictions = model.forward(X_train)

    # Calculate Loss
    loss = binary_cross_entropy(y_train.reshape(-1, 1), predictions)

    # Backpropagation and Weight Update
    model.backward(X_train, y_train, learning_rate=0.1)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss = {loss:.6f}")

# Evaluate on Test Data
test_predictions = model.forward(X_test)

# Convert probabilities to binary labels
predicted_labels = (test_predictions > 0.5).astype(int)

# Calculate Accuracy
accuracy = np.mean(predicted_labels == y_test.reshape(-1, 1))

print("\nTraining Complete!")
print(f"Test Accuracy: {accuracy*100:.2f}%")