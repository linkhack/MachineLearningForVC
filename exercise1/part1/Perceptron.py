import exercise1.part1.PerceptronOnlineTraining as online_perceptron
import exercise1.part1.Perceptron_batch_test as batch_perceptron
import matplotlib.pyplot as plt
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


def plot_decision_boundary(weights,data,targets, transformed):
    dim_data = np.size(data, 0)
    features = data[1:2, :]
    min_x = np.min(data[1,:])
    max_x = np.max(data[1,:])

    min_y = np.min(data[2,:])
    max_y = np.max(data[2,:])

    x_axis = np.linspace(min_x,max_x,500)
    y_axis = np.linspace(min_y,max_y,500)

    X_grid, Y_grid = np.meshgrid(x_axis,y_axis);
    if transformed:
        transformed_features = 0
    else:
        # transformed_features = np.array([[[1,X_grid[i,j], Y_grid[i,j]] for j in range(500)] for i in range(500)])
        transformed_features = np.array(([np.ones((500,500)),X_grid,Y_grid]))
    result = np.transpose(weights)@transformed_features
    fig = plt.contour(result,0);
    fig.plot(data[1,:],data[2,:])
    fig.show()

    return fig


