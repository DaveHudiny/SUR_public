# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: Main file of our KNN project. There we load data, call training and testing etc.
# Topic: Recognition of speaker by images and short voice records.

from loader import Loader
import torch
from torch.utils.data import DataLoader
from neural_network import ModelNN
from solutionnn import SolutionNN
from sound_classif import SolutionGMM
import cv2
import numpy as np
import os
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainAndTestSound(epochs=10):
    """Train several GMM models for sound recognition and then save best one and returns best likelihoods.

    Args:
        epochs (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    print("[INFO] Začátek trénování na zvuku")
    gmm = SolutionGMM("./data/train", "./data/dev")
    old_success_rate = -1.0
    for i in range(0, epochs):
        print("epocha "+str(i+1))
        gmm_input = gmm.train(iters=100, components=13)
        test_likelihoods, success_rate = gmm.fit_and_verify_test(gmm_input)
        if (success_rate > old_success_rate):
            old_success_rate = success_rate
            max_gmm = gmm_input
    with open("models/res.pkl", "wb") as handle:
        pickle.dump(max_gmm, handle)
    train_likelihoods = gmm.fit_and_verify_train(gmm_input)
    test_likelihoods, _ = gmm.fit_and_verify_test(max_gmm)
    # print(train_likelihoods)
    # print(test_likelihoods.shape)
    test_likelihoods_voice = np.array(
        [likes["likelihoods"] for likes in test_likelihoods])
    test_likelihoods_labels = np.array(
        [likes["correct"] for likes in test_likelihoods])
    print("Konec trenovani na zvuku")
    return test_likelihoods_voice * 10, test_likelihoods_labels - 1, gmm.test_files


def trainAndTestNeuralNetwork(metapochs=10, epochs=51):
    """Trains and tests neural network for speaker recognition by photos.

    Args:
        metapochs (int, optional): How much sets of augmented data train with. Defaults to 10.
        epochs (int, optional): Epochs of each augmented data set. Defaults to 51.

    Returns:
        (np.ndarray, [str]): Returns tuple of likelihoods and strings of file names.
    """
    loader = Loader("./data", True)
    model = SolutionNN.loadModel("./models/cnn.pkl")
    sol = SolutionNN(loader, 1, DEVICE, model)
    for i in range(metapochs):
        print(f"[INFO] Začíná metapocha {i}")
        sol.testDev()
        noises = loader.createNoisedTrain(result_size=186*2)
        noise_rotated = loader.createRotatedTrain(noises)
        noisy_rotated_dataset = noise_rotated
        noisy = DataLoader(noisy_rotated_dataset, shuffle=True, batch_size=1)
        sol.changeDataset(noisy)

	# Here you can change learning rate and regularization rate.
        sol.train(epochs, 6e-6, 1e-4)
        sol.saveModel(f"./models/cnn{i}.pkl")

    sol.saveModel("./models/cnn.pkl")
    likelihoods_nn, files = sol.testDev()
    return likelihoods_nn, files


if __name__ == "__main__":
    # model = None
    # Here you can change number of metapochs (generations of new augmented data) and epochs.
    test_likelihoods_nn_unsorted, files2 = trainAndTestNeuralNetwork(40, 15) 
    test_likelihoods_voice, test_likelihoods_labels, files = trainAndTestSound()
    test_likelihoods_nn = []
    files_real = []
    for file in files:
        index = files2.index(file)
        test_likelihoods_nn.append(test_likelihoods_nn_unsorted[index])
        files_real.append(files2[index])
    averages = np.add(test_likelihoods_nn, test_likelihoods_voice) / 2
    pointproduct = test_likelihoods_nn * test_likelihoods_voice
    correctscmp = 0
    photodec = 0
    voicedec = 0
    corrects = 0
    correctspoint = 0
    for photo, label in zip(test_likelihoods_nn, test_likelihoods_labels):
        if np.argmax(label) == label:
            correctscmp += 1
    for photo, voice, avg, label, point in zip(test_likelihoods_nn, test_likelihoods_voice, averages, test_likelihoods_labels, pointproduct):
        if np.max(voice) > np.max(photo):
            if np.argmax(voice) == label:
                correctscmp += 1
            voicedec += 1
        else:
            if np.argmax(photo) == label:
                correctscmp += 1
            photodec += 1
        if np.argmax(avg) == label:
            corrects += 1
        if np.argmax(point) == label:
            correctspoint += 1

    print(
        f"Počet korektních výsledků po průměrování: {corrects} z {len(averages)}")
    print(
        f"Počet korektních výsledků po porovnání: {correctscmp} z {len(averages)}. Fotka rozhodla v {photodec} případech a v {voicedec} rozhodl zvuk.")
    print(
        f"Počet korektních výsledků po násobení: {correctspoint} z {len(averages)}")

    # loader.pickleSave()
