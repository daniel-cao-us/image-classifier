# image-classifier

## Image Classifier
This project implements a convolutional neural network (CNN) for classifying images into one of four classes. Built with PyTorch, the model processes input image data through convolutional, pooling, and fully connected layers, optimizing performance using gradient-based methods. It is designed to handle standardized input data and offers flexibility for training and evaluation.

## Features
Convolutional Layers: Three layers for feature extraction with increasing depth.
Pooling: Max-pooling layers for dimensionality reduction and parameter optimization.
Activation Functions: LeakyReLU and Tanh for non-linear transformations.
Fully Connected Layers: Dense layers for final classification.
Training Pipeline: End-to-end support for training with batch processing and loss tracking.
Preprocessing: Standardization of input image data for improved performance.

## Usage
#1. Dataset Preparation
Prepare your image dataset with the following requirements:

Input images should have three color channels (RGB).
Input images should be resized to 31x31 pixels.

#2. Training the Model
The model can be trained using the fit function, which takes the following parameters:

train_set: Tensor of training images.
train_labels: Tensor of training labels.
dev_set: Tensor of development/validation images.
epochs: Number of training epochs.
batch_size: Batch size for training (default: 100).

#3. Model Architecture
The neural network consists of:

3 Convolutional Layers: For feature extraction.
3 Max-Pooling Layers: For reducing spatial dimensions.
2 Fully Connected Layers: For classification.

#4. Evaluation
After training, the model evaluates the development set (dev_set) and outputs:

A list of losses per epoch.
Predicted class labels for the development set.
