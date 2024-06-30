# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: loader.py file of our SUR project. This file includes Loader class for loading png and wav files
# Topic: Recognition of speaker by images and short voice records.

import os
import cv2
from scipy.io import wavfile
import pandas as pd
import pickle as pkl
import numpy as np


class Loader():
    """Class for loading photos and voice records from direstory
    """

    def __init__(self, path="./data", force=False):
        """Init method of Loader class instance.

        Args:
            path (str, optional): Path to data. Defaults to "./data".
            force (bool, optional): Force reload of data. Legacy. Defaults to False.
        """
        self.train = None
        self.dev = None
        # Initializes self.train and self.dev
        self.pickleLoadOrForce(path, force)

    def pickleLoadOrForce(self, path, force: bool = False):
        """Loads data from pickle or loads them from default directory.

        Args:
            path (str): Path to directories. 
            force (bool, optional): Force laod from path, or prefer pickle. Defaults to False.
        """
        if os.path.isfile("./train.pkl") and os.path.isfile("./dev.pkl") and not force:
            print("Načítám pickle!\n")
            with open("./train.pkl", "rb") as trainFile:
                self.train = pkl.load(trainFile)
            with open("./dev.pkl", "rb") as devFile:
                self.dev = pkl.load(devFile)
        else:
            self.train, self.dev = Loader.recursiveScan(path)

    def pickleSave(self):
        """Saves data to pickle.
        """
        with open("./train.pkl", "wb") as trainFile:
            pkl.dump(self.train, trainFile, protocol=pkl.HIGHEST_PROTOCOL)
        with open("./dev.pkl", "wb") as devFile:
            pkl.dump(self.dev, devFile, protocol=pkl.HIGHEST_PROTOCOL)

    def scanFolder(path: str):
        """Goes through folder and returns all folders and files.

        Args:
            path (str): Path to directory where to scan.

        Returns:
            ([str], [str]): Tuple of lists of folder and file names.
        """
        ls = os.scandir(path)
        folders, files = [], []
        for i in ls:
            if i.is_dir():
                folders.append(i)
            elif i.is_file():
                files.append(i.name)
            else:
                print(f"There is some mess in folder {path}")
        return folders, files

    def loadContents(prefix, files, speaker):
        """Loads content from files.

        Args:
            prefix (str): Prefix of filename (for example "./").
            files (list): Names of files, list.
            speaker (str): 

        Returns:
            np.array: Array with photo pixel values.
        """
        contents_photo = []
        for filename in files:
            if "png" in filename:
                loaded_photo = cv2.imread(
                    prefix + filename, flags=cv2.IMREAD_COLOR)
                contents_photo.append(
                    ((np.array(loaded_photo, dtype=np.float32), int(speaker) - 1, filename[:-4])))
        return contents_photo

    def recursiveScan(path):
        """Loads all data and returns train and dev dataset.

        Args:
            path (str): Path to root folder.

        Returns:
            (dict, dict): Returns dictionaries with train and dev datas. Photo data can be accessed by train["photo"].  
        """
        files = {"dev": {}, "train": {}}
        train = {"voice": [], "photo": []}
        dev = {"voice": [], "photo": []}
        folders, _ = Loader.scanFolder(path)
        for i in folders:
            speakerFolders, _ = Loader.scanFolder(f"./data/{i.name}")
            for subFolder in speakerFolders:
                _, files[i.name][subFolder.name] = Loader.scanFolder(
                    f"./data/{i.name}/{subFolder.name}")
        myKeys = list(files["dev"].keys())
        myKeys = list(map(int, myKeys))
        myKeys.sort()
        myKeys = list(map(str, myKeys))
        files["dev"] = {i: files["dev"][i] for i in myKeys}
        for speaker in files["dev"]:
            photo = Loader.loadContents(
                prefix=f"./data/dev/{speaker}/", files=files["dev"][speaker], speaker=speaker)
            dev["photo"].extend(photo)
        for speaker in files["train"]:
            photo = Loader.loadContents(
                prefix=f"./data/train/{speaker}/", files=files["train"][speaker], speaker=speaker)
            train["photo"].extend(photo)
        return train, dev

    def selectPrenoisedOrTrain(self, prenoised=None):
        """Auxilary function for slection of train or custom data.

        Args:
            prenoised ([tuple], optional): List of tuples of values and speaker. Defaults to None.

        Returns:
            [tuple]: Returns selected list of data. 
        """
        if prenoised is None:
            images = self.train["photo"]
        else:
            images = prenoised
        return images

    def createNoisedTrain(self, prenoised=None, result_size=1000):
        """Add Gaussian noise to images.

        Args:
            prenoised ([tuple], optional): Prenoised data. Defaults to None.
            result_size (int, optional): How many images you want. Defaults to 1000.

        Returns:
            [tuple]: New images.
        """
        noisy = []
        images = self.selectPrenoisedOrTrain(prenoised)
        i = 0
        while True:
            for (image, speaker, filename) in images:
                row, col, ch = image.shape
                mean = 0
                sigma = 30
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                image += gauss
                noisy.append((image, speaker, filename))
                i += 1
                if i >= result_size:
                    break
            if i >= result_size:
                break

        return noisy

    def createRotatedTrain(self, prenoised=None):
        """Creates rotated data.

        Args:
            prenoised ([tuple], optional): Prenoised data. Defaults to None.

        Returns:
            [tuple]: Returns rotated data.
        """
        rotations = []
        images = self.selectPrenoisedOrTrain(prenoised)
        for (image, speaker, filename) in images:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            angle = np.random.normal(0, 5)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            rotations.append((cv2.warpAffine(
                image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR), speaker, filename))
        return rotations
