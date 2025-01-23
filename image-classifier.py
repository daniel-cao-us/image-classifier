import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # in_size -> h -> out_size, where 1 <= h <= 256
        #make neural net with two cnn layers and two linear layers + one activation layer in between
        h = 130
        cnn_layer_1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        cnn_layer_2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        cnn_layer_3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)

        #use pools to reduce overall parameters
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        output_size = (((31 + 2 * 1-1 * 2-1) // 1) + 1)
        output_size *= output_size * 128
        output_size = 128 *3*3 #after flattening to reduce the amt of parameters
        print(output_size)

        self.neural_net = nn.Sequential(cnn_layer_1, nn.LeakyReLU(), pool1, cnn_layer_2, nn.LeakyReLU(), pool2, cnn_layer_3,nn.LeakyReLU(),pool3,nn.Flatten(),
            nn.Linear(output_size, h), nn.Tanh(), nn.Linear(h, out_size))
        self.optimizer = optim.Adam(self.parameters(), lr=lrate) #create optimizer 
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        #reshape the data
        x = x.view(-1, 3, 31, 31)
        return self.neural_net(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
        predicted_labels = self.forward(x)
        loss_vals = self.loss_fn(predicted_labels, y) #calculate loss
        self.optimizer.zero_grad() #clear gradient buffer
        loss_vals.backward() #find gradients w/ backward pass
        self.optimizer.step() #do final step of updating weights

        return loss_vals.item()




def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. 

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method will work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    #standardize data for both train and dev set
    train_mean = train_set.mean(dim=0)
    train_std = train_set.std(dim=0)
    train_set = ((train_set - train_mean)) / train_std
    dev_set = (dev_set - train_mean) / train_std

    #create nueral net and loss function
    lrate = 0.000065
    loss_function = nn.CrossEntropyLoss()
    net = NeuralNet(lrate,loss_function,train_set.shape[1],4) #make neural net of corresponding insize and outsize(4 classes)

    training_set = get_dataset_from_arrays(train_set, train_labels)
    data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=False)


    total_losses = []
    #do training by going through epochs
    for i in range(epochs):
        total_loss = 0
        for batch in data_loader: #go through images and training set/dat
            x = batch['features'] #get features and labels and store them as tensors
            y = batch['labels']
            loss = net.step(x,y) #loss will be detached in step function
            total_loss += loss
        
        total_losses.append(total_loss) #keep track of list of total losses of epoch

    #do evaluation
    net.eval()
    with torch.no_grad():
        dev_probabilities = net(dev_set)
        _, predicted_labels = torch.max(dev_probabilities.data, 1)
        predicted_labels = predicted_labels.cpu().numpy() 
        predicted_labels = predicted_labels.astype(np.int64)
        
    return total_losses,predicted_labels,net

