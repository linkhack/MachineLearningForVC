import LMS as lms
import ClosedForm as cf
import Setup as setup
import numpy as np
import matplotlib.pyplot as plt


########## 1.2.3

#setup
################
setup_training = setup.ImageSetup([0,5])
################# 
#setup.plotImages()

t = cf.ClosedForm(setup_training.getAugmentedData(), setup_training.getInputData(), setup_training.getOutputData())
imgData = setup_training.getAugmentedData()
print(imgData[1])
result_c = t.calcOptimalImageW( imgData[0], imgData[1] )


error = result_c[0]
weights = result_c[1]



label = imgData[1]
predicted_label = np.transpose(imgData[0]).dot(weights)

x_axis = np.arange(0,5,0.8)

plt.plot(x_axis,label)
plt.plot(x_axis,predicted_label)

############# test
full_setup = setup.ImageSetup([0,5], case = 1)

imgfull_set = full_setup.getAugmentedData()

full_label = imgfull_set[1]
full_predicted_label = np.transpose(imgfull_set[0]).dot(weights)

x_axisfull = np.arange(0,5.1,0.1)

plt.plot(x_axisfull,full_label)
plt.plot(x_axisfull, full_predicted_label)