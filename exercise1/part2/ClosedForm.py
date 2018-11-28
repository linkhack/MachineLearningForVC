import numpy as np
import matplotlib.pyplot as plt

class ClosedForm:
    # Constructor
    # Constructor
    def __init__(self, training_set, input_vector,output_vector):        
        self.training_set = training_set 
        self.input_vector = input_vector
        self.output_vector = output_vector


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
        
        plt.figure()
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
