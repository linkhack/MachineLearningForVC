import numpy as np
import matplotlib.pyplot as plt

class LMS:

    # Constructor
    def __init__(self, range_vector=[0, 5], step=0.1):
        self.mu, self.sigma = 0, 4 # mean and standard deviation
        self.input_vector = self.setup(range_vector, step)
        self.output_vector = self.setupOutput(self.input_vector)
        self.training_set = self.setupTraingingSet()    
        print(self.training_set)

    def getSetup(self):
        return [ self.mu, self.sigma, self.training_set]

    def setSetup(self, setupData):
        self.mu = setupData[0]
        self.sigma = setupData[1]
        self.training_set = setupData[2]
        

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

    def augment(self, x, d):
        phi = []        
        for i in range(d):
            phi.append(x ** i)
        return np.array(phi)
        

    def learn(self, w, maxIterations = 500, gamma = 0.001, plot = 0):
        d= len(w)
        axis=[-1, 6, -100, 100]
        x = self.training_set[0]         
        t = self.training_set[1]
        n = len(x)
        if(plot == 1):
            plt.plot(self.input_vector, self.output_vector, 'r-')
            plt.plot(self.training_set[0], self.training_set[1], 'b*')
            plt.axis(axis)

        iterations = 0
        while ( iterations <= maxIterations):
            error = 0
            for i in range(n):
                o = np.dot(np.array(w).transpose(), self.augment(x[i],d))                                            
                error = error + np.power( t[i] - o , 2 )
                w = w + np.dot( gamma * ( t[i] - o), self.augment(x[i],d))

            #print(error)
            #print(w)        
            if(plot == 1 and iterations % 75 == 0):
                plt.plot(x, [np.dot(w , self.augment(x_i,d)) for x_i in x ], 'g-' )
                plt.pause(0.25)            
            iterations = iterations + 1
    
        if(plot == 1):
            plt.show()            

        return [error, w]    

        
