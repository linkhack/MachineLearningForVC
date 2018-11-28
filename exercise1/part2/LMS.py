import numpy as np
import matplotlib.pyplot as plt

class LMS:

    # Constructor
    def __init__(self, training_set, input_vector,output_vector):        
        self.training_set = training_set 
        self.input_vector = input_vector
        self.output_vector = output_vector


    def augment(self, x, d):
        phi = []        
        for i in range(d):
            phi.append(x ** i)
        return np.array(phi)
        

    def learn(self, w, maxIterations = 500, gamma = 0.001, plot = 0):
        d= len(w)
        axis=[-1, 6, -15, 30]
        x = self.training_set[0]         
        t = self.training_set[1]
        n = len(x)
        if(plot == 1):
            plt.figure()
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

        
