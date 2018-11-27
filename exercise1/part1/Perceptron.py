import exercise1.part1.PerceptronOnlineTraining as online_perceptron
import exercise1.part1.Perceptron_batch_test as batch_perceptron

import numpy as np


def perc(weights, data):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    result = np.sign(np.dot(weights, data))  # @ is matrix multiplication
    return result


def percTrain(data, targets, maxIts, online):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    if online:
        return online_perceptron.perceptron_online_training(data, targets, max_iterations=maxIts)
    else:
        return batch_perceptron.percTrain(data, targets, maxIts)


def plot_decision_boundary(weights,data,targets):
    dim_data = np.size(data, 0)
    features = data[1:2, :]
    min_x = np.min(features[1,:])
    max_x = np.max(features[1,:])

    min_y = np.min(features[2,:])
    max_y = np.max(features[2,:])

    x_axis = np.linspace(min_x,max_x,500)
    y_axis = np.linspace(min_x,max_x,500)