# Handwritten_digit_recognition
Handwritten Digit Recognition Using Neural Networks
This project implements a neural network to recognize handwritten digits. The neural network is trained on the MNIST dataset, which contains 28x28 grayscale images of digits (0–9). The model uses supervised learning to classify these digits based on their pixel values.

Features
Dataset: MNIST, a standard dataset for handwritten digit recognition.
Model Architecture:
Input Layer: Processes 28x28 pixel images flattened into a 784-dimensional vector.
Hidden Layers: One or more dense layers with activation functions like ReLU.
Output Layer: A dense layer with 10 neurons (one for each digit) using softmax activation for classification.
Training: The model is trained using backpropagation and an optimization algorithm like Adam or SGD.
Evaluation: Accuracy and loss metrics are used to evaluate performance on test data.
Dependencies
Python
TensorFlow/Keras
Numpy
Matplotlib (for visualizing results)
Usage
Clone the repository.
Install the required dependencies.
Run the notebook/script to train the model.
Test the model with new handwritten digits or the MNIST test set.
