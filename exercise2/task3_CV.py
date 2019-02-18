# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:08:03 2019

"""

import exercise2.mnist_subset_and_feature_utils as mnf
import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import exercise2.tools_Plot as t_pl


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

#%%Implementation of a secondary function

def cross_validation(database,targetbase,sigma, C_range):
    """ calculate the error rate by cross-validation of a SVM on the given database for a RBF kernel with sigma.
    For C, a range of values can be given.
    it returns an array with the same size than the C array, which gives the error rate for each C
    """
    nbr_C = C_range.shape
    Error_rates = np.zeros(nbr_C)
    
    nSet, nFeature, nSample = database.shape
    
    # Implementation of the SVM
    
    svm_CV = SVM()
    svm_CV.setSigma(sigma)
    
    svm_CV.setCV(database,targetbase)
    svm_CV.setGram(kernel.rbfkernel)
    
    for i in range(0,nbr_C):
        C = C_range[i]
        error_rate = np.zeros((nSet,1))
        for k in range(0,nSet):
            [alpha, w0, positions] = svm_CV.trainSVM_CV(k, c=C)
            
            predictions = svm_CV.discriminant_CV(alpha, w0, k)
            
            error = t_pl.error_rate(predictions, svm_CV.targetbasecv[k])
            
            error_rate[k] = error
        
        Error_rates[i] = np.sum(error_rate)/nSet
        
    return Error_rates

#%%Test of the cross validation
    
C_range= np.array([0.02*i for i in range(1,10000,100)])
Sigma_range=np.array([0.02*i for i in range(1,5000,100)])
nbr_C = C_range.size
nbr_sigma = Sigma_range.size

result = np.zeros((nbr_C,nbr_sigma)) #matrix  of the result depending of C and sigma

for i in range(0,nbr_sigma):
    sigma = Sigma_range[i]
    error_rate = cross_validation(data_sets,target_sets,sigma,C_range)
    result[:,i]=error_rate

        
pos_min_C,pos_min_sigma= np.argmin(result) #get the position of the minimum of the matrix

[C_opti,sigma_opti] = [C_range[pos_min_C],Sigma_range[pos_min_sigma]]

print("sigma optimum =")
print(sigma_opti)

print("C optimum =")
print(C_opti)
        

