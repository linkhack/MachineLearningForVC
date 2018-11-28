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

    features = utils.augment_data(features)

    weights = perceptron_online_training(features, targets=training_targets, max_iterations=500000)
    perc_result = perc(weights, features)
    correct = np.equal(perc_result, training_targets)
    correct_nr = sum(correct)
    print(str(correct_nr))
    correct_precentage = np.sum(correct) / np.size(features, 1)
    print("Correct percentage:" + str(correct_precentage))

#    plt.scatter(features[0, :], features[1, :], c=training_targets)
#    # fig = scatter_matrix_from_dict(properties, training_targets)
#    plt.show()

    fig = Perceptron.plot_decision_boundary(weights, features, training_targets,False)
#    fig.show()
    
    """part on the test set"""
    
    test_set,test_targets = utils.get_test_digits(digits, 200)
    
    test_features = utils.calculate_features(test_set)
    test_properties = utils.collect_regionprops(test_set)

    test_features = utils.augment_data(test_features)
    
    test_result = Perceptron.perc(weights, test_features)
    test_correct = np.equal(test_result, test_targets)
    test_correct_nr = sum(test_correct)
    print(str(test_correct_nr))
    test_correct_precentage = np.sum(test_correct) / np.size(test_features, 1)
    print("Correct percentage of test ser:" + str(test_correct_precentage))
    return


if __name__ == '__main__':
    main()
