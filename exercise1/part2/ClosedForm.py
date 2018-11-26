import numpy as np
import matplotlib.pyplot as plt

class ClosedForm:
    # Constructor
    def __init__(self, range_vector=[0, 5]):
        self.mu, self.sigma = 0, 4 # mean and standard deviation
        self.input_vector = self.setup(range_vector)
        self.output_vector = self.setupOutput(self.input_vector)
        self.training_set = self.setupTraingingSet()    


    def setup(self, range_vector=[0, 5], step=0.1):
        input_vector = []

        for x in self.my_range(range_vector[0], range_vector[1], step):
            input_vector.append(x)

        return np.array(input_vector)

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

    def setupOutput(self, input_vector):
        return np.array([(2 * x ** 2 - 6 * x +1) for x in input_vector])

    def setupTraingingSet(self):
        select_index = []
        for x in range(0, len(self.input_vector), 8):
            select_index.append(x)

        return np.array([self.input_vector[select_index], self.drawRandomValue(self.output_vector[select_index])])

    def drawRandomValue(self, value ):
        return np.array([x +  np.random.normal(self.mu, self.sigma) for x in value])


    def calcOptimalW(self,j): 
        j= j+1       
        a = np.zeros([j,j])
        B = np.zeros(j)
        x = self.training_set[0]         
        t = self.training_set[1]

        for i in range(j):
            for k in range(j):               
                a[i][k] = self.afit(x, k+i)

            B[i] = self.bfit(x,self.training_set[1],i)
    
        w_ = np.linalg.solve(a, B)    
        
        w_phi = np.array( [ self.fit(x_value, w_) for x_value in x] )            
        error = np.sum((t - w_phi) ** 2 )

        return [error, w_]        
    
    def presentMode(self, max):
        ws = []
        for i in range(max):
            ws.append( self.calcOptimalW(i))

        self.calculateErrorAndPlot(ws)


    def afit(self, x, j):
        phi = []        
        for i in x:
            phi.append(i ** j)

        return np.sum(phi)
        

    def bfit(self, x,y,j):        
        phi = []        
        for i in range(len(x)):
            phi.append((x[i] ** j) * y[i])

        return np.sum(phi)


    def calculateErrorAndPlot(self, ws):
        axis=[-1, 6, -100, 100]
        
        x_ = np.linspace(-1.0,6.0, num=50)
    
        plt.plot(self.input_vector, self.output_vector, 'r-')
        plt.plot(self.training_set[0], self.training_set[1], 'b*')
        plt.axis(axis)

        for d_w in ws:
            w = d_w[1]
            print('error:'+str(d_w[0]))

            y_ = np.array( [ self.fit(x_value, w) for x_value in x_] )                    
            plt.plot(x_, y_ , 'g-' )
            plt.pause(0.25)

    
        plt.show()  

    def fit(self, x, w):
        phi = []        
        for i in range(len(w)):
            phi.append(x ** i)

        phi = np.array(phi)
        return (phi.dot(w))
