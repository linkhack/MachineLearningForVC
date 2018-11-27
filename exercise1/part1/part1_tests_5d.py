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
    digits = [0, 1]
    training_set, training_targets = utils.get_digits(digits, 500)

    features = utils.calculate_features(training_set)
    properties = utils.collect_regionprops(training_set)

    features = utils.transform_features(features)

    weights = Perceptron.percTrain(features, targets=training_targets, online =False, maxIts=500000)[0]
    perc_result = Perceptron.perc(weights, features)
    correct = np.equal(perc_result, training_targets)
    correct_nr = sum(correct)
    print(str(correct_nr))
    correct_precentage = np.sum(correct) / np.size(features, 1)
    print("Correct percentage:" + str(correct_precentage))

    plt.scatter(features[1, :], features[2, :], c=training_targets)
    # fig = scatter_matrix_from_dict(properties, training_targets)
    plt.show()

    fig = Perceptron.plot_decision_boundary(weights, training_set, training_targets,False)
    fig.show()
    return


if __name__ == '__main__':
    main()
