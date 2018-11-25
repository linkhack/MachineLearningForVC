import numpy as np


# Assumes the data was already augmented (homogeneous coordinates)
def perceptron_online_training(training_set, targets, max_iterations):
    data_dimension = np.size(training_set, 0)
    nr_of_datapoints = np.size(training_set, 1)
    weights = np.zeros(data_dimension)
    avg_weight = np.zeros(data_dimension)

    for i in range(max_iterations):
        weights = update_weights(training_set[:, i % nr_of_datapoints], targets[i % nr_of_datapoints], weights)
        avg_weight = i/(i+1) * avg_weight + weights/(i+1)
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
    result = np.sign(np.dot(weights, data))  # @ is matrix multiplication
    return result
