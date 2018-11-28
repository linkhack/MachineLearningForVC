# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:27:43 2018

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
    digits = [0, 7]
    training_set, training_targets = utils.get_digits(digits, 500)

    features = utils.calculate_features(training_set)
    properties = utils.collect_regionprops(training_set)

    features = utils.transform_features(features)
    print(np.shape(features))

    weights = Perceptron.percTrain(features, targets=training_targets, online =True, maxIts=200000)
    perc_result = Perceptron.perc(weights, features)
    correct = np.equal(perc_result, training_targets)
    correct_nr = sum(correct)
    print(str(correct_nr))
    correct_precentage = np.sum(correct) / np.size(features, 1)
    print("Correct percentage:" + str(correct_precentage))

#    plt.scatter(features[1, :], features[2, :], c=training_targets)
#    # fig = scatter_matrix_from_dict(properties, training_targets)
#    plt.show()

    fig = Perceptron.plot_decision_boundary(weights, features, training_targets,True)
#    fig.show()
    
    """part on the test set"""
    
    test_set,test_targets = utils.get_test_digits(digits, 200)
    
    test_features = utils.calculate_features(test_set)
    test_properties = utils.collect_regionprops(test_set)

    test_features = utils.transform_features(test_features)
    
    test_result = Perceptron.perc(weights, test_features)
    test_correct = np.equal(test_result, test_targets)
    test_correct_nr = sum(test_correct)
    print(str(test_correct_nr))
    test_correct_precentage = np.sum(test_correct) / np.size(test_features, 1)
    print("Correct percentage of test ser:" + str(test_correct_precentage))
    
    return


if __name__ == '__main__':
    main()
