# -*- coding: utf-8 -*-

import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import exercise2.tools_Plot as t_pl
import matplotlib.pyplot as plt
import exercise1.part1.mnist_subset_and_feature_utils as utils

##importants datas
digits = [0, 7]
sigma = 0.5
C = 0.50
c_range = [5, 10, 100]
sigma_range = [0.1, 1, 5]
## Importation of the set from the first exercice

training_set, training_targets = utils.get_digits(digits, 30)
test_set, test_targets = utils.get_test_digits(digits, 10000)

features = utils.calculate_features(training_set)
test_features = utils.calculate_features(test_set)

## Creation of the SVM

svm = SVM()
svm.setSigma(sigma)
[alpha, w0, sv_index] = svm.trainSVM(features, training_targets, kernel.rbfkernel, c=C)

# position index of support vectors in alpha array

##Draw data points:
# firstClass
ind = training_targets > 0
data = features[ind, :]
plt.scatter(data[:, 0], data[:, 1], s=70, c="r", marker='.')
# secondClass
ind = training_targets < 0
data = features[ind, :]
plt.scatter(data[:, 0], data[:, 1], s=70, c="b", marker='.')

# draw margin
t_pl.plot(features, training_targets, w0, alpha, sv_index, svm)
# t_pl.plot_SVM(alpha,w0,positions,features,training_targets,features,training_targets,kernel=kernel.rbfkernel,sigma=sigma)
index = 1
fig = plt.figure(figsize=(12, 12))
error_rate = np.zeros((3, 3))
error_rate_test = np.zeros((3, 3))
i = 0
j = 0
for sigma in sigma_range:
    for c in c_range:
        plt.subplot(3, 3, index)
        svm.setSigma(sigma)
        [alpha, w0, sv_index] = svm.trainSVM(features, training_targets, kernel.rbfkernel, c=c)

        ##Draw data points:
        # firstClass
        ind = training_targets > 0
        data = features[ind, :]
        plt.scatter(data[:, 0], data[:, 1], s=70, c="r", marker='.')
        # secondClass
        ind = training_targets < 0
        data = features[ind, :]
        plt.scatter(data[:, 0], data[:, 1], s=70, c="b", marker='.')

        # draw margin
        t_pl.plot(features, training_targets, w0, alpha, sv_index, svm)

        pred_test = np.sign(svm.discriminant(alpha, w0, sv_index, features, training_targets, test_features))
        pred_train = np.sign(svm.discriminant(alpha, w0, sv_index, features, training_targets, features))
        error_rate[i, j] = t_pl.error_rate(pred_train, training_targets)
        error_rate_test[i, j] = t_pl.error_rate(pred_test, test_targets)
        index += 1
        j += 1
    i += 1
    j = 0
plt.show()
print(np.array_str(error_rate))
print(np.array_str(error_rate_test))
