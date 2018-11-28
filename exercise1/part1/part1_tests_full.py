# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:14:27 2018

@author: arnau
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import mnist
import itertools
from exercise1.part1.PerceptronOnlineTraining import perceptron_online_training, perc
import exercise1.part1.mnist_subset_and_feature_utils as utils
import exercise1.part1.Perceptron as Perceptron


def main():
    digits = [0, 1]
    training_set, training_targets = utils.get_digits(digits, 500)
    
    features = []
    for image in training_set:
        feature_pixel = []  # image as a straight ligne
        n = np.size(image, 0)
        m = np.size(image, 1)
        for i in range(0, n):
            for j in range(0, m):
                feature_pixel.append(image[i, j])
        features.append(feature_pixel)  # add the image as a straigh list of pixel

    features = np.transpose(np.array(features))

    weights = Perceptron.percTrain(features, targets=training_targets, online =False, maxIts=5000)[0]
    perc_result = Perceptron.perc(weights, features)
    correct = np.equal(perc_result, training_targets)
    correct_nr = sum(correct)
    print(str(correct_nr))
    correct_precentage = np.sum(correct) / np.size(features, 1)
    print("Correct percentage:" + str(correct_precentage))
    
       """part on the test set"""
    
    test_set,test_targets = utils.get_test_digits(digits, 200)
    
    test_features = []
    for image in test_set:
        feature_pixel = []  # image as a straight ligne
        n = np.size(image, 0)
        m = np.size(image, 1)
        for i in range(0, n):
            for j in range(0, m):
                feature_pixel.append(image[i, j])
        test_features.append(feature_pixel)  # add the image as a straigh list of pixel

    test_features = np.transpose(np.array(test_features))

    test_result = Perceptron.perc(weights, test_features)
    test_correct = np.equal(test_result, test_targets)
    test_correct_nr = sum(test_correct)
    print(str(test_correct_nr))
    test_correct_precentage = np.sum(test_correct) / np.size(test_features, 1)
    print("Correct percentage of test ser:" + str(test_correct_precentage))
    
    

    return


if __name__ == '__main__':
    main()