# Team: xhudak03
# Members: xhudak03, xmracn00, xpleva07
# Subject: SUR
# Description: final.py file of our SUR project. There we generate final results of project.
# Topic: File for final recognition.

import os
import argparse
import cv2
import solutionnn
import torch
import numpy as np
import pickle
from sound_classif import SolutionGMM


def print_decisions(decisions, path_name):
    """Goes through tuples of filenames and decisions and prints them to the folder path.

    Args:
        decisions ([tuple]): Array of tuples of filenames and likelihood arrays.
        path_name (str): Path to folder where to print results. 
    """
    with open(path_name, "w") as handle:
        for name, results in decisions:
            handle.write(name + " ")
            handle.write(str(np.argmax(results) + 1))
            for result in results:
                handle.write(" " + str(np.log(result)))
            handle.write("\n")


def fit_final_neural(path):
    """Loads network from hardcoded path and then creates likelihoods.

    Args:
        path (str): Source of files.

    Returns:
        [(str, np.ndarray)]: Tuple of filename and likelihoods.
    """
    model = solutionnn.SolutionNN.loadModel("./models/cnn.pkl")
    sol = solutionnn.SolutionNN(model=model, fit_only=True)
    decisions = []
    files = []
    for mypath, dirs, filenames in os.walk(path):
        for file in filenames:
            if "png" in file:
                loaded_photo = cv2.imread(
                    os.path.join(mypath, file), flags=cv2.IMREAD_COLOR)
                fit = sol.fit(torch.Tensor(np.array([loaded_photo])))
                fit = fit.cpu()
                with torch.no_grad():
                    decisions.append((file[:-4], fit.numpy()[0]))
                    files.append(file[:-4])
    return decisions, files


def fit_final_gmm(path):
    """Loads GMM from hardcoded path and then creates likelihoods.

    Args:
        path (str): Source of files.

    Returns:
        [(str, np.ndarray)]: Tuple of filename and likelihoods.
    """
    with open("./models/res.pkl", "rb") as handle:
        gmm_model = pickle.load(handle)
    results = []
    solution = SolutionGMM("./None", path)
    solution.redefine_tests(path)
    test_likelihoods, _ = solution.fit_and_verify_test(
        gmm_model, suppress_print=True)
    results = results + test_likelihoods
    for mypath, dirs, _ in os.walk(path):
        for dir in dirs:
            solution = SolutionGMM("./None", os.path.join(mypath, dir))
            solution.redefine_tests(os.path.join(mypath, dir))
            test_likelihoods, _ = solution.fit_and_verify_test(
                gmm_model, suppress_print=True)
            results = results + test_likelihoods
    files = []
    for mypath, _, myfiles in os.walk(path):
        for file in myfiles:
            if "wav" in file:
                files.append(file[:-4])
    res = [x["likelihoods"] for x in results]
    sound_decisions = [(file, result) for file, result in zip(files, res)]

    return sound_decisions, files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program pro finální rozpoznávání v předmětu SUR.")
    parser.add_argument("--path", default="./prefinal")
    args = parser.parse_args()
    # Single model prints
    print("[INFO] Dělání rozhodnutí nad soubory.")
    photos, photo_files = fit_final_neural(args.path)
    print_decisions(photos, "photo_results.txt")
    voices, voice_files = fit_final_gmm(args.path)
    print("[INFO] Printing decisions for single models to photo_results.txt and voice_results.txt")
    print_decisions(photos, "photo_results.txt")
    print_decisions(voices, "voice_results.txt")

    # Multiple models prints
    voices_correct_order = []
    files_real = []
    photo_vals = np.array([x[1] for x in photos])
    voices_vals = np.array([x[1] for x in voices])
    for file in photo_files:
        index = voice_files.index(file)
        voices_correct_order.append(voices_vals[index])
        files_real.append(voice_files[index])

    averages = np.add(photo_vals, voices_correct_order) / 2
    pointproduct = photo_vals * voices_correct_order
    maxes = np.maximum(photo_vals, voices_correct_order)

    print("[INFO] Printing decisions for model combinations to combined_averages.txt, combined_products.txt and combined_compares.txt")
    print_decisions(zip(files_real, averages), "combined_averages.txt")
    print_decisions(zip(files_real, pointproduct), "combined_products.txt")
    print_decisions(zip(files_real, maxes), "combined_compares.txt")
