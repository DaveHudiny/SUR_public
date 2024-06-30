# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: solutionnnn.py file of our SUR project. This file includes SolutionNN class
#              for neural network implementation for photo recognition
# Topic: Recognition of speaker by images and short voice records.

# set the matplotlib backend so figures can be saved in the background
from loader import Loader
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from neural_network import ModelNN
from solution import Solution
import os
import numpy as np
import copy as copy

torch.backends.cudnn.enabled = True


class SolutionNN(Solution):
    """Implementation of interface for neural network model.

    Args:
        Solution (Solution): Abstract class for interface of used solutions.
    """

    def __init__(self, loader: Loader = None, batch_size: int = 1, device=torch.device("cuda"), model: ModelNN = None,
                 trainDataLoader: DataLoader = None, devDataLoader: DataLoader = None, fit_only: bool = False):
        """Init method of SolutionNN class.

        Args:
            loader (Loader, optional): Initialized Loader instance for loading datasets. Defaults to None.
            batch_size (int, optional): Size of batch for learning. Defaults to 1.
            device (torch.device(), optional): Torch device. Defaults to torch.device("cpu").
            model (ModelNN, optional): Pretrained model. Defaults to None.
            trainDataLoader (DataLoader, optional): DataLoader of custom train data. Defaults to None.
            devDataLoader (DataLoader, optional): DataLoader of custom test data. Defaults to None.
        """
        self.batch_size = batch_size
        self.device = device
        if not fit_only:
            self.loader = loader

            if trainDataLoader is None:
                self.setDefaultDataLoader()
            else:
                self.trainDataLoader = trainDataLoader

            if devDataLoader is None:
                self.devDataLoader = DataLoader(
                    loader.dev["photo"], batch_size=batch_size, shuffle=False)
            else:
                self.devDataLoader = devDataLoader

        if model is None:
            print("[INFO] Byl vytvořen model zcela nový.")
            self.model = ModelNN(
                numChannels=3,
                classes=31).to(self.device)
        else:
            print("[INFO] Byl načten již existující model.")
            self.model = model

    def changeDataset(self, newDataLoader: DataLoader):
        """Changes source of train dataset.

        Args:
            newDataLoader (DataLoader): New train data loader.
        """
        self.trainDataLoader = newDataLoader

    def setDefaultDataLoader(self):
        """Resets custom data loader to good ol' loader.
        """
        self.trainDataLoader = DataLoader(self.loader.train["photo"], shuffle=True,
                                          batch_size=self.batch_size)

    def saveModel(self, path):
        """Saves model to pickle.

        Args:
            path (str): Saves file to path.
        """
        with open(path, "wb") as handle:
            torch.save(self.model, handle)

    def loadModel(path):
        """Loads model from pickle.

        Args:
            path (str): Path to pickle.

        Returns:
            ModelNN: Returns trained model of neural network from path.
        """
        if os.path.exists(path):
            with open(path, "rb") as handle:
                model = torch.load(handle, map_location='cpu')
            return model
        else:
            return None

    def train(self, epochs: int = 100, init_lr: float = 1e-4, weight_decay: float = 0.02):
        """Train neural network

        Args:
            epochs (int, optional): Number of epochs of training. Defaults to 100.
            init_lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): L2 regularization value. Defaults to 0.02.
        """
        print("[INFO] Inicializace trénování...")

        # initialize our optimizer and loss function
        opt = Adam(self.model.parameters(), lr=init_lr,
                   weight_decay=weight_decay)
        lossFn = nn.CrossEntropyLoss()

        print("[INFO] Trénování započalo")
        for epoch in range(epochs):
            totalTrainLoss = 0
            trainCorrect = 0
            for (x, y, _) in self.trainDataLoader:
                x = x.permute(0, 3, 1, 2)

                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                y = y.long()
                pred = self.model(x)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
            if epoch % 2 == 0:
                print("[INFO] Epocha {}, Chybová funkce: {}, Správné trénovací výstupy: {}".format(
                    epoch, totalTrainLoss, trainCorrect))

    def fit(self, x):
        """Uses model to predict value of input.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Likelihoods of classes. 
        """
        self.model.to(self.device)
        x = x.permute(0, 3, 1, 2)
        (x) = (x.to(self.device))
        pred = self.model(x)
        return pred

    def testDev(self):
        """Tests dev dataset.

        Returns:
            (np.array, [str]): Returns tuple of likelihoods and filenames.
        """
        devCorrect = 0
        top3Correct = 0
        results = []
        results_keys = []
        self.model.to(self.device)
        old_dataset = copy.copy(self.devDataLoader.dataset)
        for x, y, z in self.devDataLoader:
            x = x.permute(0, 3, 1, 2)
            (x, y) = (x.to(self.device), y.to(self.device))
            pred = self.model(x)
            _, indices = torch.topk(pred, 3)
            top3Correct += (y == indices).type(torch.float).sum().item()
            devCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
            with torch.no_grad():
                results.append(pred.cpu().numpy()[0])
                results_keys.append(z[0])
            x = x.permute(0, 2, 3, 1)
        datasetSize = len(self.devDataLoader.dataset)
        print("[VÝSLEDKY TEST] Správné testovací výstupy: {} z {}, top 3 správně: {} z {}".format(
            devCorrect, datasetSize, top3Correct, datasetSize))
        return np.array(results), results_keys
