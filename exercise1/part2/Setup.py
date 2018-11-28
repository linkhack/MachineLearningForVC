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
        if(case == 0 ):
            self.input_vector = self.setup(range_vector)
            self.output_vector = self.setupOutput(self.input_vector)
            self.training_set = self.setupTraingingSet(self.input_vector)
        else:   
            self.input_vector = self.setup(range_vector)
            self.output_vector = self.setupOutput(self.input_vector)
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
    
    def getAugmentedData(self):
        
        img = []
        augmentedData = []
        ts = []
            
        for training_set in self.training_set:
            img  = training_set[0]               
            y  = training_set[1]                     
            x  = training_set[2]                     
            m_1 = training_set[3]                     
            m_2 = training_set[4] 
        
            tData= []
            data= np.zeros(29*29)
            ts = []
            for i in range(29):
                for j in range(29):                    
                    #imgData.append( [1 , (i-m_1)**2, (j-m_2)**2, - (3*x) **2  ])                                    
                    #augmentedData.append(img[i][j] )        
                    data[ (i*29 + j) ] =img[i][j]
                    ts.append(np.array([y]) )
        
            augmentedData.append([data, np.array(ts)])
            #ts.append(np.array(tData).transpose())
            
        return  augmentedData

        print('---------------------------')
        print(np.shape(augmentedData))
        #print(augmentedData)
        print(np.shape(ts))
        #print(ts)
        print('---------------------------')
        
    
        return [augmentedData, np.array(ts)]            

    def augment_data(self, data):
        data_dimension = np.size(data, 0)
        nr_of_datapoints = np.size(data, 1)
        augmented_data = np.ones([data_dimension + 1, nr_of_datapoints])
        augmented_data[:-1, :] = data
        return augmented_data

    def my_range(self, start, end, step):        
        while start <= end:
            yield start
            start += step

    def setupTraingingSet(self,input ):
        select_index = []
        for x in self.my_range(0, len(input), 8):
            select_index.append(x)

        return self.generateImages(input[select_index])        

    def generateImages(self,input):
        images = []
        for x in input:
            y_i = (2 * (x ** 2) - (6 * x) +1)
            imgD = self.generateImage(x)
            img = np.array([ imgD[0] , y_i , x , imgD[1], imgD[2]])

            images.append(img )

        
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

        return [image, m_1, m_2]

    def plotImages(self):
        n = len(self.training_set)
        fig=plt.figure()
        for i in range(n):
            fig.add_subplot(1, n, i+1)   # subplot one
            plt.imshow(self.training_set[i][0])

        plt.show()
    