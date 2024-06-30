from solution import Solution
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm
import numpy as np


class SolutionGMM(Solution):
    # Initialization of the object -- loads sound samples and extracts features from them to be worked with in other functions
    # train_path -- Path where training data is saved
    # dev_path -- Path where testing data is saved
    def __init__(self, train_path, dev_path):
        self.trains = []
        self.tests = []
        self.orig_trains = []
        test_files = []
        self.test_files = []
        # Loads and extracts features from sound samples for each speaker
        for i in range(1, 32):
            self.trains.append(wav16khz2mfcc(train_path+"/"+str(i)).values())
            self.tests.append(wav16khz2mfcc(dev_path+"/"+str(i)).values())
            self.orig_trains.append(wav16khz2mfcc(
                train_path+"/"+str(i)).values())
            test_files.extend(wav16khz2mfcc(dev_path+"/"+str(i)).keys())
        for f in test_files:
            splitor = f.split("/")
            self.test_files.append(splitor[-1][:-4])

    def redefine_tests(self, path):
        self.tests = []
        self.tests.append(wav16khz2mfcc(path).values())

    # Trains a GMM with the EM algorithm 
    # iters -- Amount of iterations of the EM algorithm
    # components -- Amount of component parts of the GMM
    def train(self, iters, components):
        # Transformation of extracted features to the appropriate form
        for i in range(0, 31):
            self.trains[i] = np.vstack(self.trains[i])
        gaussian_components = components
        mu_vector = []
        sigma = []
        weights = []
        gmm_input = []
        # Initialization of a GMM model
        for i in range(0, 31):
            # Mean vector -- picks a random element from a list of extracted features
            mu_vector.append(self.trains[i][np.random.randint(
                1, len(self.trains[i]), gaussian_components)])
            # Covariance matrix -- is initialized as variance of extracted features for individual classes times the amount of gaussian components
            sigma.append([np.var(self.trains[i], axis=0)]*gaussian_components)
            # Weights of individual components -- initialized to be the same for all components
            weights.append(np.ones(gaussian_components)/gaussian_components)
            # Creation of a initialized GMM for a specific classs
            gmm_input.append(
                [self.trains[i], weights[i], mu_vector[i], sigma[i]])
        # Training of the GMM
        for i in range(iters):
            for j in range(0, 31):
                res = train_gmm(
                    gmm_input[j][0], gmm_input[j][1], gmm_input[j][2], gmm_input[j][3])
                gmm_input[j] = [self.trains[j], res[0], res[1], res[2]]
        return gmm_input

    # Calculates likelihoods of classes for a training dataset, classifies the files in the training dataset, and determines accuracy of the model
    # gmm_input -- The trained GMM model
    def fit_and_verify_train(self, gmm_input):
        corrects = 0
        overall = 0
        ref_likelihoods = []
        # Across all training recordings
        for enum, train in enumerate(self.orig_trains):
            for i in train:
                overall = overall+1
                log_likelihood = []
                likelihoods = []
                # Gets log-likelihoods for each class
                for j in range(0, 31):
                    log_likelihood.append(
                        np.sum(logpdf_gmm(i, gmm_input[j][1], gmm_input[j][2], gmm_input[j][3])))
                log_likelihood = np.array(log_likelihood)
                # Transforms likelihoods by applying an inverse value of the division of a likelihood of a classification divided by all likelihoods
                for j in range(0, 31):
                    likelihoods.append(
                        1/(log_likelihood[j]/np.sum(log_likelihood)))
                likelihoods = np.array(likelihoods)
                # Subtracts a minimum likelihood value and normalizes results (so that the sum of all likelihoods is one)
                likelihoods = likelihoods-np.min(likelihoods)
                likelihoods = likelihoods/np.sum(likelihoods)
                # Picks the class that corresponds to the highest likelihood
                if (likelihoods.argmax() == enum):
                    corrects = corrects+1
                ref_likelihoods.append(
                    {"likelihoods": likelihoods, "correct": enum+1})
        # Prints out the accuracy of the trained model
        print("Uspesnost na trenovacim datasetu: "+str(corrects/overall))
        return ref_likelihoods
    
    # Calculates likelihoods of classes for a testing dataset, classifies the files in the training dataset, and determines accuracy of the model
    # gmm_input -- The trained GMM model
    def fit_and_verify_test(self, gmm_input, suppress_print=False):
        corrects = 0
        overall = 0
        ref_likelihoods = []
        for enum, test in enumerate(self.tests):
            for i in test:
                overall = overall+1
                log_likelihood = []
                likelihoods = []
                for j in range(0, 31):
                    log_likelihood.append(
                        np.sum(logpdf_gmm(i, gmm_input[j][1], gmm_input[j][2], gmm_input[j][3])))
                log_likelihood = np.array(log_likelihood)
                for j in range(0, 31):
                    likelihoods.append(
                        1/(log_likelihood[j]/np.sum(log_likelihood)))
                likelihoods = np.array(likelihoods)
                likelihoods = likelihoods-np.min(likelihoods)
                likelihoods = likelihoods/np.sum(likelihoods)
                if (likelihoods.argmax() == enum):
                    corrects = corrects+1
                ref_likelihoods.append(
                    {"likelihoods": likelihoods, "correct": enum+1})
        if suppress_print == False:
            print("Uspesnost na testovacim datasetu: "+str(corrects/overall))

        return ref_likelihoods, corrects/overall

    def fit():
        pass
