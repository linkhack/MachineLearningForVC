import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel


def linearkernel(x1, x2):
    """
    linear kernel function
    """
    return np.inner(x1.T, x2.T)


def rbfkernel(x1, x2, sigma):
    """
    Radial basis function kernel of x1 and x2 with sigma as parameter

    """
    #vector_norms = cdist(np.atleast_2d(x1.T), np.atleast_2d(x2.T),'sqeuclidean')  # all distances for x_1 and x_2
    #value = np.exp(-vector_norms / (2 * sigma ** 2))  # here is anyway all pointwise

    value = rbf_kernel(x1.T,x2.T,1/(2*sigma**2))

    return value
