# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:20:00 2018

@author: arnau
"""

import numpy as np

import matplotlib.pyplot as plt

from skimage import feature
from skimage import measure

import cv2


# %% Algorithms


def perc(w, X):
    """simulate a perceptron , where w is the weight vector and X is a matrix witn inputs vector in its Columns, return the target values"""
    n = np.size(X, 0)
    m = np.size(X, 1)
    y = []
    for i in range(0, m):
        vect = X[:, i]
        product = np.vdot(w, vect)  # scalar product between the perceptron and each vector
        if product > 0:
            y.append(1)
        else:
            y.append(-1)
    return (y)


def percTrain(X, t, maxIts):
    """returns a weight vector w corresponding to the decision boundary separating the input vectors in X according to
        their target values t.
        Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1
    """
    n = np.size(X, 0)
    m = np.size(X, 1)
    w = np.zeros(n)
    state = True
    Its = 0
    perc0 = perc(w, X)

    while state and Its < maxIts:  # we continue while the vectors are missclassified (state = True) and while the number of iterations is below the maximum
        delta = 0.0
        for i in range(0, m):  # adding to the perceptron all vectors that are missclassified
            if perc0[i] != t[i]:
                delta += t[i] * X[:, i]
        w += delta/m
        perc0 = perc(w, X)  # classification of vectors with the new perceptron
        nb_missed = 0
        for i in range(0, m):  # consider the number of vectors missclassified
            if perc0[i] == -t[i]:
                nb_missed += 1

        if nb_missed == 0:  # if there is none, the perceptron is correct and the algorithm can end
            state = False
        Its += 1

    return (w,
            state)  # we also return the state of the perceptron in case the algorithm ended cause the maximum of iterations was reached


def convertion(image):
    """ transcribe an image in 0 and 1 instead of 256"""
    image_01 = np.copy(image)
    for i in range(0, 28):
        for j in range(0, 28):
            if image[i, j] == 0:
                image_01[i, j] = 0
            else:
                image_01[i, j] = 1
    return (image_01)


def sum_pixel(image):
    n = np.size(image, 0)
    m = np.size(image, 1)
    sum_pixel = 0
    for i in range(0, n):
        for j in range(0, m):
            if image[i, j] > 0:
                sum_pixel += 1
    return (sum_pixel)


def resolv_6d(x, weights):
    """to print the decision boundary for the 6_d perceptron, corresponds to a 2nd order equation"""
    a = weights[4]  # a*Y²
    b = weights[2] + weights[5] * x  # b*Y with b= w2+w5*x
    c = weights[0] + weights[1] * x + weights[3] * (x ** 2)
    delta = b ** 2 - 4 * a * c
    if a == 0:  # finally not a 2nd order equation
        sol = -b / c
        return ([sol, sol, 0])
    if delta < 0:
        return ([0, 0, -1])
    elif delta == 0:
        sol0 = -b / (2 * a)
        return ([sol0, sol0, delta])
    else:
        sol1 = (-b - np.sqrt(delta)) / (2 * a)
        sol2 = (-b + np.sqrt(delta)) / (2 * a)
        return ([sol1, sol2, 1])


# %% Importation of Datas + selection
import mnist as mn
def main():
    data_set = mn.train_images()  # array contening all the images

    data_label = mn.train_labels()  # array contening the corresponding labels

    nb_0 = 0
    nb_1 = 0
    liste_0_1 = []

    number = np.size(data_label)

    train_set_label = []
    train_set_value = []

    i = 0
    search_0 = True
    search_1 = True

    while i < 60000 and (search_1 or search_0):
        """way to create a set of images contening only 0 and 1 images from  the initial data set"""
        label = data_label[i]
        if label == 1 and nb_1 < 500:
            train_set_label.append([i, 1])
            train_set_value.append(1)
            nb_1 += 1
        elif label == 0 and nb_0 < 500:
            train_set_label.append([i, 0])
            train_set_value.append(-1)
            nb_0 += 1
        if nb_1 == 500:
            search_1 = False
        if nb_0 == 500:
            search_0 = False
        i += 1

    train_set_label = np.array(train_set_label)

    # %% Initiation of the features
    train_set_features = []
    for y in train_set_label:
        image = data_set[y[0]]
        image_center = image[5:20, 5:20]
        image_01 = convertion(image)

        # feature 1: 3 part images
        part_1 = image_01[:, 0:12]
        part_2 = image_01[:, 12:16]
        part_3 = image_01[:, 16:28]
        a = sum_pixel(part_1)
        b = sum_pixel(part_2)
        c = sum_pixel(part_3)
        d = sum_pixel(image_01)
        #    print(a)
        #    print(b)
        #    print(c)
        #    print(d)
        feature1 = (a - c + b) / (d + 1)
        #    print(feature1)
        #    print('ok')

        # feature 2: line profile horizontal
        line_profile2 = measure.profile_line(image, [14, 1], [14, 27])
        i = 0
        j = 26
        while line_profile2[i] == 0 and i < j:
            i += 1
        while line_profile2[j] == 0 and j > i:
            j += -1
        feature2 = j - i  # distance between the 2 exterior sides of the number

        train_set_features.append([1, feature1, feature2])

    train_set_features = np.transpose(np.array(train_set_features))

    # %% others features


    # %% training of the perceptron of 2 features


    percep, state = percTrain(train_set_features, train_set_value, 500)

    print(percep)
    print(state)

    # %% print of the result
    X1 = []
    Y1 = []

    X0 = []
    Y0 = []
    k = 0
    for y in train_set_label:
        if y[1] == 0:
            X0.append(train_set_features[0, k])
            Y0.append(train_set_features[1, k])
        #        plt.scatter(train_set_features[k,0],train_set_features[k,1],"r")
        if y[1] == 1:
            X1.append(train_set_features[0, k])
            Y1.append(train_set_features[1, k])
        #        plt.scatter(train_set_features[k,0],train_set_features[k,1],"b")
        k += 1

    X = np.arange(0, 1, 0.05)
    Y = [-(percep[0] + percep[1] * x) / percep[2] for x in X]

    plt.plot(X0, Y0, "r")
    plt.plot(X1, Y1, "b")
    plt.plot(X, Y, "r--")

    # %% training of the 5_d perceptron
    train_set_features_5d = []
    k = 0
    for k in range(0, 1000):
        x_1 = train_set_features[0, k]
        x_2 = train_set_features[1, k]
        train_set_features_5d.append(
            [1, x_1, x_2, x_1 ** 2, x_2 ** 2, x_1 * x_2])  # features of each image : 1 x1 x2 x1² x2² x1 x2

    train_set_features_5d = np.transpose(
        np.array(train_set_features_5d))  # reshape to have a matrix where each colonn represent the 6_features of an image

    percep_5d, state_5 = percTrain(train_set_features_5d, train_set_value, 500)

    # %% Print of the result fo the 5_d perceptron
    X1 = []
    Y1 = []

    X0 = []
    Y0 = []
    k = 0
    for y in train_set_label:
        if y[1] == 0:
            X0.append(train_set_features[0, k])
            Y0.append(train_set_features[1, k])
        #
        if y[1] == 1:
            X1.append(train_set_features[0, k])
            Y1.append(train_set_features[1, k])
        #
        k += 1

    X = np.arange(0, 1, 0.05)
    X_6d = []
    Y_6d = []
    for x in X:  # calculations of the boundaries
        solv = resolv_6d(x, percep_5d)
        if solv[2] == 0:  # delta = 0 => double root
            X_6d.append(x)
            Y_6d.append(solv[0])
        elif solv[2] == 1:  # delta > 0 => 2 solutions for one values of x
            X_6d.append(x)
            X_6d.append(x)
            Y_6d.append(solv[0])
            Y_6d.append(solv[1])

    plt.plot(X0, Y0, "r")
    plt.plot(X1, Y1, "b")
    plt.plot(X_6d, Y_6d, "r--")
    plt.show()

    # %% training on the full image
    train_set_images = []
    for y in train_set_label:
        imag = data_set[y[0]]
        feature_pixel = []  # image as a straight ligne
        n = np.size(imag, 0)
        m = np.size(imag, 1)
        for i in range(0, n):
            for j in range(0, m):
                feature_pixel.append(imag[i, j])
        train_set_images.append(feature_pixel)  # add the image as a straigh list of pixel

    train_set_images = np.transpose(
        np.array(train_set_images))  # reshape to have a matrix where each colonn represent an image of the data_set

    percep_full, state_full = percTrain(train_set_images, train_set_value, 500)

    # %% implementation of the test set
    test_set = mn.test_images()  # array contening all the images

    test_label = mn.test_labels()  # array contening the corresponding labels

    nb_0 = 0
    nb_1 = 0
    liste_0_1 = []

    number = np.size(test_label)

    test_set_label = []
    test_set_value = []

    i = 0
    search_0 = True
    search_1 = True

    while i < 60000 and (search_1 or search_0):
        """way to create a set of images contening only 0 and 1 images from  the initial test set"""
        label = data_label[i]
        if label == 1 and nb_1 < 200:
            test_set_label.append([i, 1])
            test_set_value.append(1)
            nb_1 += 1
        elif label == 0 and nb_0 < 200:
            test_set_label.append([i, 0])
            test_set_value.append(-1)
            nb_0 += 1
        if nb_1 == 200:
            search_1 = False
        if nb_0 == 200:
            search_0 = False
        i += 1

    test_set_label = np.array(test_set_label)
# %% features of the test set

# %% result on the test set
if __name__ == '__main__':
    main()