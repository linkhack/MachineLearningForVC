# -*- coding: utf-8 -*-

import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import tools_Plot as t_pl
import exercise1.part1.mnist_subset_and_feature_utils as utils

##importants datas
digits = [0, 7]
sigma = 1
C = 10


## Importation of the set from the first exercice

training_set, training_targets = utils.get_digits(digits, 500)
features = utils.calculate_features(training_set)


## Creation of the SVM

svm = SVM()

[alpha, w0] = svm.trainSVM(features, training_targets, kernel.rbfkernel, sigma =sigma, C=C)
