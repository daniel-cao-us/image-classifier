# image-classifier

## Image Classifier
This code implements a convolutional neural network (CNN) for classifying images into one of four classes(ship, automobile, dog, or frog). Built with PyTorch, the model processes input image data through convolutional, pooling, and fully connected layers, optimizing performance using gradient-based methods. It is built to handle standardized input data and gives the user flexibility for training and evaluation. 

## Features of the CNN
Convolutional Layers: Three layers for feature extraction with increasing depth.
Pooling: Max-pooling layers for dimensionality reduction and parameter optimization.
Activation Functions: LeakyReLU and Tanh for non-linear transformations.
Fully Connected Layers: Dense layers for final classification.

## Usage
# 1. Dataset Preparation
Prepare the image dataset with the following requirements:

Input images should have three color channels (RGB).
Input images should be resized to 31x31 pixels.

# 2. Training the Model
The model can be trained using the fit function, which takes the following parameters:

train_set: Tensor of training images.
train_labels: Tensor of training labels.
dev_set: Tensor of development/validation images.
epochs: Number of training epochs.
batch_size: Batch size for training (default: 100).

# 3. Model Architecture
The neural network consists of:

3 Convolutional Layers: For feature extraction.
3 Max-Pooling Layers: For reducing spatial dimensions.
2 Fully Connected Layers: For classification.

# 4. Evaluation
After training, the model evaluates the development set (dev_set) and outputs:

A list of losses per epoch.
Predicted class labels for the development set.
