import LMS as lms
import ClosedForm as cf
import Setup as setup


########## 1.2.3

#setup
################
setup = setup.ImageSetup([0,5])
################# 
#setup.plotImages()

t = cf.ClosedForm(setup.getAugmentedData(), setup.getInputData(), setup.getOutputData())
imgData = setup.getAugmentedData()
print(imgData[1])
result_c = t.calcOptimalImageW( imgData[0], imgData[1] )
