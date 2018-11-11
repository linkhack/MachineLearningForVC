import numpy as np
import matplotlib.pyplot as plt

class LinearBasisFunction(object):

    # Constructor
    def __init__(self, range_vector=[0, 5], step=0.1):
        self.mu, self.sigma = 0, 4 # mean and standard deviation
        self.input_vector = self.setup(range_vector, step)
        self.output_vector = self.setupOutput(self.input_vector)
        self.training_set = self.setupTraingingSet()


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
        return np.array([(2 * x ** 2 - 6 * x +1) for x in input_vector])

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




range_v = [0, 5]
Model = LinearBasisFunction(range_v, 0.1)
#print(Model.output_vector)
Model.plotSetup()

#print(len(model.getinput_vector()))
