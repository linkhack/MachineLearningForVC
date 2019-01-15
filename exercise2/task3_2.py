import exercise2.mnist_subset_and_feature_utils as mnf
import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import exercise2.tools_Plot as t_pl
import exercise1.part1.Perceptron as perc
import matplotlib.pyplot as plt


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
SVM_soft_error_rate =[]
Perc_error_rate=[]

for i in range(0,nr_sets):
    
    ## SVM

   svm = SVM()
   data_set = mnf.transform_images(data_sets[i])
   target_set = target_sets[i]
   
   [alpha, w0, positions] = svm.trainSVM(data_set, target_set, kernel.linearkernel)

   predicted_targets_svm = np.sign(svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=test_data.T)) # calculation of predicted target for swm
   SVM_error_rate.append(t_pl.error_rate(predicted_targets_svm,test_target))

   svm = SVM()
   [alpha, w0, positions] = svm.trainSVM(data_set, target_set, kernel.linearkernel, c=0.1 )

   predicted_targets_svm = np.sign(svm.discriminant(alpha=alpha, w0=w0, X=data_set.T, t=target_set,
                                             Xnew=test_data.T))  # calculation of predicted target for swm

   SVM_soft_error_rate.append(t_pl.error_rate(predicted_targets_svm, test_target))


    ## perceptron
   data_set_perc = mnf.augment_data(data_set)
   test_data_perc = mnf.augment_data(test_data)
   
   weights = perc.percTrain(data_set_perc, target_set,1000,True)
   
   predicted_targets_perc = perc.perc_multi(weights, test_data_perc) # calculation of predicted target for perceptron
   Perc_error_rate.append(t_pl.error_rate(predicted_targets_perc,test_target)) 
   
SVM_error =sum(SVM_error_rate)/150
SVM_soft_error =sum(SVM_soft_error_rate)/150
Perc_error = sum(Perc_error_rate)/150


print("SVM_error =")
print(SVM_error)
print("SVM_soft_error =")
print(SVM_soft_error)
print("Perc_error =")
print(Perc_error)

#show error rate over sets
x= range(0,nr_sets )
plt.scatter(x, SVM_error_rate, c="b", marker='.')
plt.scatter(x, Perc_error_rate, c="r", marker='.')
#plt.scatter(x, SVM_soft_error_rate, c="y", marker='+')
plt.show()


#%% Analysing the effect of changing C and sigma on the average test error rate:

C_range=[1,10,100,1000]
Sigma_range=[0.5,1,1.5,3,6]

C_error_rate= []
Sigma_error_rate=[]

average_nbre_SV_1 = []
average_nbre_SV_2 = []

sigma = 1
C = 100

for C in C_range:
    SVM_error_rate_0 =[]
    nbre_SV = []
    
    for i in range(0,nr_sets):
    
        ## SVM
        svm = SVM()
        svm.setSigma(sigma)
        data_set = mnf.transform_images(data_sets[i])
        target_set = target_sets[i]
        
        [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c=C)
        predicted_targets_svm = np.sign( svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=test_data.T)) # calculation of predicted target for swm
        SVM_error_rate_0.append(t_pl.error_rate(predicted_targets_svm,test_target))
        nbre_SV.append(np.size(positions)) #number of support vector
    average_nbre_SV_1.append(sum(nbre_SV)/150) #average of number of support vector
    C_error_rate.append(sum(SVM_error_rate_0)/150)

C = 100

for sigma in Sigma_range:
    SVM_error_rate_1 =[]
    nbre_SV=[]
    
    for i in range(0,nr_sets):
    
        ## SVM
        svm = SVM()
        svm.setSigma(sigma)
        data_set = mnf.transform_images(data_sets[i])
        target_set = target_sets[i]
        

        [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c=C)
        predicted_targets_svm =np.sign( svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=test_data.T )) # calculation of predicted target for swm

        SVM_error_rate_1.append(t_pl.error_rate(predicted_targets_svm,test_target))
        nbre_SV.append(np.size(positions))
    average_nbre_SV_2.append(sum(nbre_SV)/150)
    Sigma_error_rate.append(sum(SVM_error_rate_1)/150)

plt.figure(11)
# printing of the result    
plt.subplot(2,2,1)
plt.plot(C_range,C_error_rate)
plt.title("average test error rate depending on C")

plt.subplot(2,2,2)
plt.plot(Sigma_range,Sigma_error_rate)
plt.title("average test error rate depending on sigma")

plt.subplot(2,2,3)
plt.plot(C_error_rate,average_nbre_SV_1)
plt.title("average number of SV depending on th error rate(for C variation)")

plt.subplot(2,2,4)
plt.plot(Sigma_error_rate,average_nbre_SV_2)
plt.title("average number of SV depending on th error rate(for sigma variation)")
plt.show()
#%%analysing of the effect changing set on average training error:
C_range=[1,10,100,1000]
Sigma_range=[0.5,1,1.5,3,6]
 
C_error_rate_2= []
Sigma_error_rate_2=[]

sigma = 1
C = 100

for C in C_range:
    SVM_error_rate_2 =[]
    
    for i in range(0,nr_sets):
    
        ## SVM
        svm = SVM()
        svm.setSigma(sigma)
        data_set = mnf.transform_images(data_sets[i])
        target_set = target_sets[i]
        


        [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel,c=C)

        predicted_targets_svm = np.sign( svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=data_set.T)) # calculation of predicted target for swm
        SVM_error_rate_0.append(t_pl.error_rate(predicted_targets_svm,target_set))
    C_error_rate_2.append(sum(SVM_error_rate_2)/150)

C = 100

for sigma in Sigma_range:
    SVM_error_rate_3 =[]
    
    for i in range(0,nr_sets):
    
        ## SVM
        svm = SVM()
        svm.setSigma(sigma)
        data_set = mnf.transform_images(data_sets[i])
        target_set = target_sets[i]
        

        [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c=C)

        predicted_targets_svm = np.sign(svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=data_set.T)) # calculation of predicted target for swm
        SVM_error_rate_1.append(t_pl.error_rate(predicted_targets_svm,target_set))
    Sigma_error_rate_2.append(sum(SVM_error_rate_3)/150)
    
#some plot
plt.figure(13)
plt.subplot(2,2,1)
plt.plot(C_range,C_error_rate_2)
plt.title("average training error rate depending on C")

plt.subplot(2,2,2)
plt.plot(Sigma_range,Sigma_error_rate_2)
plt.title("average training error rate depending on sigma")
plt.show()
#%%function for M-fold cross validation

def cross_validation(data_sets,target_sets,sigma,C):
    "return the average test error rate of the SVM with rbfkernel, sigma and C with cross-validation "
    error_rate = []
    [nbre_sets,nbre_images,m,n] = data_sets.shape
    print(nbre_sets)
    for k in range(0,nbre_sets):
        data_set = []
        target_set = []
        test_set = []
        for j in range(0,nbre_sets): #creation of the data set with all the images of the 150 datas except the number k
            if j != k:
                print(np.size(data_sets[j]))
                for l in range(nbre_images):
                    data_set.append(data_sets[j,l,:,:])
                    target_set.append(target_sets[j][l])
        data_set = np.array(data_set)
        target_set = np.array(target_set)
        data_set = mnf.transform_images(data_set) # get the matrix data
        
        test_set = mnf.transform_images(data_sets[k])
        test_target = target_sets[k]
        
        #initialisation the SVM
        svm = SVM()
        svm.setSigma(sigma)
        
        #trained the SVM on the data set        
        [alpha, w0,positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c=C)
        predicted_targets_svm = np.sign(svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=test_set.T))
        error_rate.append(t_pl.error_rate(predicted_targets_svm,test_target))
        
    return (sum(error_rate)/nbre_sets)

#%% cross-validation to get the best sigma and C
    
C_range=[0.02*i for i in range(1,10000,100)]
Sigma_range=[0.02*i for i in range(1,5000,100)]

result = []
sigma_C = []

for C in C_range:
    for sigma in Sigma_range:
        error_rate = cross_validation(data_sets,target_sets,sigma,C)
        result.append(error_rate)
        sigma_C.append([sigma,C])
        
position_minimum= np.argmin(np.array(result)) #get the position of the minimum

[sigma_opti, C_opti] = sigma_C(position_minimum)

print("sigma optimum =")
print(sigma_opti)

print("C optimum =")
print(C_opti)
        

#%% calculation of rbf kernel with optimum sigma and C on the MNIST test set
SVM_error_rate_opti =[]


for i in range(0,nr_sets):
    
    ## SVM

   svm = SVM()
   svm.setSigma(sigma_opti)
   data_set = mnf.transform_images(data_sets[i])
   target_set = target_sets[i]
   
   
   [alpha, w0, positions] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c= C_opti)

   predicted_targets_svm = np.sign(svm.discriminant(alpha=alpha,w0=w0,X=data_set.T,t=target_set,Xnew=test_data.T)) # calculation of predicted target for swm
   SVM_error_rate_opti.append(t_pl.error_rate(predicted_targets_svm,test_target))
   
SVM_error_opti = sum(SVM_error_rate_opti)/150

print("average error rate on test set for SVM with RBF kernel for optimum sigma and C = ")
print(SVM_error_opti)
