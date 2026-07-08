# Neural Network From Scratch (Using NumPy)

<p align="center">
  <em>A fully vectorized feedforward neural network built entirely from scratch using only NumPy — no TensorFlow, no PyTorch, just math.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Vectorized-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Built%20With-Pure%20Math-orange?style=flat-square" alt="Pure Math">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>



## Overview

This project implements a **feedforward neural network** completely from scratch — forward propagation, backpropagation, gradient descent, and loss computation — without relying on any deep learning framework.

The task: generate random 2D points and classify whether each one lies **inside or outside a circle**. This is a **non-linearly separable** problem, meaning a simple linear model can't solve it — a hidden layer with a non-linear activation is required.



## Features

- **Pure NumPy implementation** — every gradient derived and applied manually
- **Single hidden layer architecture** — `Input (2) → ReLU (4) → Sigmoid Output (1)`
- **Binary Cross-Entropy loss** with numerical stability (epsilon clipping to avoid `log(0)`)
- **Custom synthetic dataset generator** — circular non-linear classification boundary
- **Modular codebase** — activations, loss, data generation, and model cleanly separated



## Project Structure

```
├── activations.py       # ReLU, Sigmoid, and their derivatives
├── loss.py               # Binary Cross-Entropy loss function
├── neural_network.py     # NeuralNetwork class (forward + backward pass)
├── utils.py               # Synthetic circular dataset generator
├── main.py                # Training loop, evaluation
├── Requirements.txt       # Project description / task summary
└── README.md
```



## How It Works

| Stage | Description |
|---|---|
| **Data** | 1000 random 2D points in `[-1, 1]`, labeled `1` if `x² + y² < 0.5`, else `0` |
| **Forward Pass** | `Z1 = X·W1 + b1` → `A1 = ReLU(Z1)` → `Z2 = A1·W2 + b2` → `A2 = Sigmoid(Z2)` |
| **Loss** | Binary Cross-Entropy between predictions and true labels |
| **Backward Pass** | Gradients computed layer-by-layer via the chain rule |
| **Update** | Weights and biases updated using vanilla gradient descent |

### Architecture

```
Input Layer (2 features)
        │
        ▼
Hidden Layer (4 neurons, ReLU)
        │
        ▼
Output Layer (1 neuron, Sigmoid)
```



## Getting Started

### Prerequisites

```bash
pip install numpy
```

### Run Training

```bash
python main.py
```

You'll see the loss decrease over training epochs, followed by a final test accuracy report:

```
Epoch 0    | Loss = 0.693147
Epoch 1000 | Loss = ...
...
Training Complete!
Test Accuracy: 97.05%
```



## Configuration

You can tune the network directly in `neural_network.py`:

```python
self.W1 = np.random.randn(2, 4) * 0.01   # input -> hidden
self.W2 = np.random.randn(4, 1) * 0.01   # hidden -> output
```

| Parameter | Location | Description |
|---|---|---|
| Hidden layer size (`4`) | `neural_network.py` | Increase for more capacity / smoother decision boundaries |
| `learning_rate` | Passed in `main.py` (`model.backward(...)`) | Controls gradient descent step size |
| `epochs` | `main.py` | Total training iterations |
| `samples` | `utils.py` (`generate_data`) | Number of data points generated |



## Author

**Wasi-Ur-Rehman**
Generative AI & ML Engineer
[GitHub @knightRehman](https://github.com/knightRehman)


<p align="center">
  <sub>Built with ❤️ and raw NumPy — no shortcuts, no frameworks.</sub>
</p>
