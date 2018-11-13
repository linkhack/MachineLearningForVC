import numpy as np

def perceptron_online_training(training_set, targets, max_iterations):
    data_dimension = np.size(training_set, 0)
    nr_of_datapoints = np.size(training_set, 1)
    weights = np.zeros(data_dimension)
    augmented_data = np.ones([data_dimension + 1, nr_of_datapoints])
    augmented_data[:-1, :] = training_set

    for i in range(max_iterations):
        weights = update_weights(augmented_data[:, i % nr_of_datapoints],targets[i % nr_of_datapoints],weights)
        if perc(weights, augmented_data) == targets:
            return weights
        
    return weights


def update_weights(data_point, target, old_weights):
    new_weight = np.zeros(np.shape(old_weights))
    if (np.dot(data_point, old_weights) * target < 0):
        new_weight = old_weights + target * data_point
    return new_weight

def perc(weights,data):
    result = np.sign(data.T * weights)
    return result
