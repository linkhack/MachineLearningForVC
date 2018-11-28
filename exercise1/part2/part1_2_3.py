import LMS as lms
import ClosedForm as cf
import Setup as setup
import numpy as np
import matplotlib.pyplot as plt


########## 1.2.3

#setup
################
setup_training = setup.ImageSetup([0,5]) # setting the training_set images
################# 
#setup.plotImages()

t = cf.ClosedForm(setup_training.getAugmentedData(), setup_training.getInputData(), setup_training.getOutputData()) #implemention of closed form
imgData = setup_training.getAugmentedData()

result_c = t.calcOptimalImageW( imgData[0], imgData[1] ) #calculation of w*


error = result_c[0]
weights = result_c[1]

print(error)


label = imgData[1]
predicted_label = np.transpose(imgData[0]).dot(weights)



############# test
full_setup = setup.ImageSetup([0,5], case = 1) # computing the images of the all set of data

imgfull_set = full_setup.getAugmentedData()

full_label = imgfull_set[1] #yi
full_predicted_label = np.transpose(imgfull_set[0]).dot(weights)#yi*


############ Figures

plt.subplot(1,2,1)

x_axis = np.arange(0,5,0.8)

plt.scatter(x_axis,label)
plt.scatter(x_axis,predicted_label)

plt.subplot(1,2,2)
x_axisfull = np.arange(0,5.1,0.1)

plt.plot(x_axisfull,full_label,'r', label='target values')
plt.plot(x_axisfull, full_predicted_label,'b',label='predicted values')
plt.legend()