# Neural Network From Scratch

This project implements a neural network using Python. The implementation includes key features such as model architecture, training, evaluation, and optimization. The project is structured to be modular and easily extensible.

## Table of Contents

- [Features](#features)
- [Implementation](#implementation)
  - [Model Definition](#model-definition)
  - [Training Process](#training-process)
  - [Visualization](#visualization)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- **Customizable Model Architecture**: Supports multiple layers with configurable activation functions and layer sizes.
- **Training and Evaluation**: Implements training loops with backpropagation and optimizers.
- **Dataset**: Uses the `EMNIST` dataset. Loads and preprocesses datasets efficiently.
- **Performance Metrics**: Computes accuracy, precision, recall, and F1 score for classification tasks.
- **Hyperparameter Tuning**: Allows modification of learning rates, batch sizes, and optimization techniques.
- **Report**: Generates a detailed report with training and evaluation metrics.

## Implementation

The core implementation includes the following components:

### Model Definition

It consists of:

- Input Layer
- Hidden Dense Layers (with ReLU as activation function)
- Dropout Layer (for regularization)
- Output Layer (for classification or regression tasks)

### Training Process

The training process involves:

1. Forward propagation
2. Loss computation (cross-entropy for classification)
3. Backpropagation
4. Parameter updates using ADAM optimizer
5. Performance tracking

### Visualization

The implementation includes visualization tools to analyze:

- Training and validation loss
- Accuracy trends
- Model performance metrics

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

Run the `neural_net.ipynb` notebook step by step to train and evaluate the neural network.

## License

This project is licensed under the MIT License.
