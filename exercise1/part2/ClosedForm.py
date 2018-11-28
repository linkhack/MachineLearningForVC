import numpy as np
import matplotlib.pyplot as plt

class ClosedForm:
    # Constructor
    def __init__(self, training_set, input_vector,output_vector):        
        self.training_set = training_set 
        self.input_vector = input_vector
        self.output_vector = output_vector

    def calcOptimalImageW(self, x , t):    
        
        X = x                         
        a = np.zeros([29*30*len(X),1])
        print(a)
        i= 1
        for x in X:
            for k in range(1,29):
                for l in range(1,30):     
                    print(x[k-1][l-1])          
                    print(i*k*l -1)
                    a[i*k*l -1 ] = x[k-1][l-1]                                  
            #img done        
            i = i+1        
    
        inv= np.linalg.pinv(a)

        w_ = inv.dot(t)    
        w_phi = np.array( [ self.fit(x_value, w_) for x_value in X] )            
        error = np.sum((t - w_phi) ** 2 )


        return [error, w_]  

    def calcOptimalW(self,j): 
        j= j+1
        
        X = self.training_set[0]         
        t = self.training_set[1]
        a = np.zeros([len(t),j])
        
        i= 0
        for x in X:
            for k in range(j):
                a[i][k] = x ** k                                  
            i = i+1        
    
        inv= np.linalg.pinv(a)

        w_ = inv.dot(t)    
        w_phi = np.array( [ self.fit(x_value, w_) for x_value in X] )            
        error = np.sum((t - w_phi) ** 2 )

        return [error, w_]        
    
    def presentMode(self, max):
        ws = []
        for i in range(max):
            ws.append( self.calcOptimalW(i))

        self.calculateErrorAndPlot(ws)


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
