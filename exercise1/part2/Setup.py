import numpy as np
import matplotlib.pyplot as plt

class Setup:    
    def __init__(self, range_vector=[0, 5]):
        self.mu, self.sigma = 0, 4 # mean and standard deviation
        self.input_vector = self.setup(range_vector)
        self.output_vector = self.setupOutput(self.input_vector)
        self.training_set = self.setupTraingingSet()   

    def getSetup(self):
        return [ self.mu, self.sigma, self.training_set]

    def setSetup(self, setupData):
        self.mu = setupData[0]
        self.sigma = setupData[1]
        self.training_set = setupData[2]

    def getTrainingData(self):
        return self.training_set        

    def getInputData(self):
        return self.input_vector

    def getOutputData(self):
        return self.output_vector
            

    def my_range(self, start, end, step):        
        while start <= end:
            yield start
            start += step

    def setup(self, range_vector=[0, 5], step=0.1):
        input_vector = []

        for x in self.my_range(range_vector[0], range_vector[1], step):
            input_vector.append(x)

        return np.array(input_vector)

    def setupOutput(self, input_vector):        
        return np.array([(2 * (x ** 2) - (6 * x) +1) for x in input_vector])

    def setupTraingingSet(self):
        select_index = []
        for x in self.my_range(0, len(self.input_vector), 8):
            select_index.append(x)

        return np.array([self.input_vector[select_index], self.drawRandomValue(self.output_vector[select_index])])

    def drawRandomValue(self, value ):
        return np.array([x +  np.random.normal(self.mu, self.sigma) for x in value])

    def plotSetup(self, axis=[-1, 6, -5, 25]):
        plt.plot(self.input_vector, self.output_vector, 'r-')
        plt.plot(self.training_set[0], self.training_set[1], 'bo')
        plt.axis(axis)
        plt.show()

class ImageSetup:
    def __init__(self, range_vector=[0, 5], case = 0):      
        self.input_vector = self.setup(range_vector)
        self.output_vector = self.setupOutput(self.input_vector)
        if(case == 0 ):
            self.training_set = self.setupTraingingSet()
        else:   
            self.training_set = self.generateImages(self.input_vector)

    def setup(self, range_vector=[0, 5], step=0.1):
        input_vector = []

        for x in self.my_range(range_vector[0], range_vector[1], step):
            input_vector.append(x)

        return np.array(input_vector)

    def setupOutput(self, input_vector):        
        return np.array([(2 * (x ** 2) - (6 * x) +1) for x in input_vector])

    def getTrainingData(self):
        return self.training_set        

    def getInputData(self):
        return self.input_vector

    def getOutputData(self):
        return self.output_vector
    

    def my_range(self, start, end, step):        
        while start <= end:
            yield start
            start += step

    def setupTraingingSet(self ):
        select_index = []
        for x in self.my_range(0, len(input), 8):
            select_index.append(x)

        return self.generateImages(input[select_index])        

    def generateImages(self,input):
        images = []
        for x in input:
            images.append(self.generateImage(x))

        return images



    def generateImage(self, x):
        d = 29
        m_1 = np.random.normal(15, 2)
        m_2 = np.random.normal(15, 2)

        image = np.zeros([d,d])
        value = 0
        for i in range(d):
            for j in range(d):
                value = ( (i - m_1) ** 2 + (j - m_2 ) ** 2 -(3 * x) ** 2 )
                if(value > 0 ):
                    value_t = 1 
                else:
                    value_t = 0
                image[i][j] =  value_t

        return image

    def plotImages(self):
        n = len(self.training_set)
        fig=plt.figure()
        for i in range(n):
            fig.add_subplot(1, n, i+1)   # subplot one
            plt.imshow(self.training_set[i])

        plt.show()
    