import numpy as np


def linearkernel(x1, x2):
    """
    linear kernel function
    """
    return np.dot(x1, x2)

def rbfkernel(x1,x2,sigma):
    """
    Radial basis function kernel of x1 and x2 with sigma as parameter

    """
    vector = x1-x2
    
    value = np.exp(-np.dot(vector,vector)/(2*sigma**2))
    
    return value
