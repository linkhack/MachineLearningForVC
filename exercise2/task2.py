# -*- coding: utf-8 -*-

import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import exercise2.tools_Plot as t_pl
import matplotlib.pyplot as plt
import exercise1.part1.mnist_subset_and_feature_utils as utils

##importants datas
digits = [0, 7]
sigma = 1.5
C = 100

## Importation of the set from the first exercice

training_set, training_targets = utils.get_digits(digits, 500)
features = utils.calculate_features(training_set)

## Creation of the SVM

svm = SVM()
svm.setSigma(sigma)
[alpha, w0, sv_index] = svm.trainSVM(features, training_targets, kernel.rbfkernel, c=C)

# position index of support vectors in alpha array
ind = np.arange(len(alpha))[sv_index]


features = np.transpose(features)

##Draw data points:
# firstClass
ind = training_targets > 0
data = features[ind, :]
plt.scatter(data[:, 0], data[:, 1],s=70, c="r", marker='.')
# secondClass
ind = training_targets < 0
data = features[ind, :]
plt.scatter(data[:, 0], data[:, 1], s=70, c="b", marker='.')

#draw margin
t_pl.plot(features, training_targets, w0, alpha, sv_index, svm)
# t_pl.plot_SVM(alpha,w0,positions,features,training_targets,features,training_targets,kernel=kernel.rbfkernel,sigma=sigma)
