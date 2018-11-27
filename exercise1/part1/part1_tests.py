import numpy as np
import matplotlib.pyplot as plt
import cv2
import mnist
import itertools
from exercise1.part1.PerceptronOnlineTraining import perceptron_online_training, perc
import exercise1.part1.mnist_subset_and_feature_utils as utils
import exercise1.part1.Perceptron as Perceptron


def main():
    digits = [1, 4]
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

    plt.scatter(features[0, :], features[1, :], c=training_targets)
    # fig = scatter_matrix_from_dict(properties, training_targets)
    plt.show()

    fig = Perceptron.plot_decision_boundary(weights, training_set, training_targets,False)
    fig.show()
    return


if __name__ == '__main__':
    main()
