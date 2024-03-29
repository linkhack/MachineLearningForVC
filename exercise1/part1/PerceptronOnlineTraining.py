import numpy as np

"""
 Author: Link
"""


def perceptron_online_training(training_set, targets, max_iterations):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    data_dimension = np.size(training_set, 0)
    nr_of_datapoints = np.size(training_set, 1)
    weights = np.zeros(data_dimension)

    for i in range(max_iterations):
        weights = update_weights(training_set[:, i % nr_of_datapoints], targets[i % nr_of_datapoints], weights)
        if np.all(perc(weights, training_set) == targets):
            return weights
    return weights


def update_weights(data_point, target, old_weights):
    value = np.dot(data_point,old_weights)*target
    if value <= 0:
        return old_weights + target * data_point
    else:
        return old_weights


def perc(weights, data):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    result = np.sign(np.dot(weights, data))  
    return result
