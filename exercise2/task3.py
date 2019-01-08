import exercise2.mnist_subset_and_feature_utils as mnf
import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import tools_Plot as t_pl
import exercise1.part1.Perceptron as perc


## useful values
nr_sets = 150
nr_sample = 35
digits = [0,7]

## value to change
sigma = 2
C = 100

""" initialisation of the data:
    150 sets of mnist data of 70 elements from 2 classes
"""

data_sets, target_sets = mnf.get_digits(digits,nr_sample,nr_sets)

test_data,test_target = mnf.get_test_digits(digits,100000) # high value to get all the digits of the two classes
test_data = mnf.transform_images(test_data) #transform into a matrix where images are in column


## calculation of the error rate on test data for the SVM trained on each data_sets and the perceptron trained on each data set:


SVM_error_rate =[]
Perc_error_rate=[]

for i in range(0,nr_sets):
    
    ## SVM

   svm = SVM()
   data_set = mnf.transform_images(data_sets[i])
   target_set = target_sets[i]
   
   [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.linearkernel)

   predicted_targets_svm = svm.discriminant(alpha=alpha,w0=w0,X=data_set,t=target_set,Xnew=test_data)
   SVM_error_rate.append(t_pl.error_rate(predicted_targets_svm,test_target)) 
   
   ## perceptron
   data_set_perc = mnf.augment_data(data_set)
   test_data_perc = mnf.augment_data(test_data)
   
   weights = perc.percTrain(data_set_perc, target_set,1000,True)
   
   predicted_targets_perc = perc.perc_multi(weights, test_data_perc)
   Perc_error_rate.append(t_pl.error_rate(predicted_targets_perc,test_target)) 
   

   