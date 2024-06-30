# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: neural_network.py file of our SUR project. This file includes ModelNN class for training  png and wav files
# Topic: Recognition of speaker by images and short voice records.

import torch.nn as nn
from torch import flatten, stack
from solution import Solution


class ModelNN(nn.Module):
    """ Class for simple neural network model for our SUR project solution
    """

    def __init__(self, classes: int = 31, numChannels: int = 3):
        super(ModelNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=16,
                               kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(5, 5), stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.norm4 = nn.InstanceNorm2d(32)

        self.fc1 = nn.Linear(in_features=9248, out_features=500)
        self.relu3 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=500, out_features=classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward function for our neural network model

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Returns probabilities for each class created by soft max function.
        """
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the input through the second set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.norm4(x)
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size, 9248)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = self.Softmax(x)
        x = x.view(batch_size, 31)
        return x
