import exercise1.part1.PerceptronOnlineTraining as online_perceptron
import exercise1.part1.Perceptron_batch_test as batch_perceptron

import numpy as np


# data has to be augmented, i.e. data[-1,:] = 1
def perc(weights, data):
    result = np.sign(np.dot(weights, data))  # @ is matrix multiplication
    return result


# data has to be augmented, i.e. data[-1,:] = 1
def percTrain(data, targets, maxIts, online):
    if online:
        return online_perceptron.perceptron_online_training(data, targets, max_iterations=maxIts)
    else:
        return batch_perceptron.percTrain(data, targets, maxIts)
