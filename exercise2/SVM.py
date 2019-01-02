import cvxopt
import numpy as np
import exercise2.Kernel as kernel


class SVM:

    def __init__(self):
        """contructor"""

    def trainSVM(self, x, t, kernel=kernel.linearkernel):
        """
        Keyword arguments:
        X -- input vector
        t -- labels
        kernel -- kernel function. see Kernel class,

        returns [alpha, w0]
        The vector alpha holds the optimal dual parameter values, i.e., the lagrange multipliers αi for all N input vectors
        w0 is the offset of the decision plane, which can be computed using alpha and one support vector (The data points for which the (dual variables) αi > 0 are called support vectors.)
        """
        # store for later?
        self.x = x
        self.t = t
        self.kernel = kernel

        # get some sizes
        nFeatures, nSamples = x.shape

        K = np.zeros((nSamples, nSamples))
        for i in range(nSamples):
            for j in range(nSamples):
                K[i, j] = self.kernel(x[:,i], x[:,j])

        # FYI tc='d' specifies double as matrix content type!

        # prepare arguments for solver:
        # we need to minify a constrained quadratic program
        # min x : 1/2 xT P x + qT x
        P = cvxopt.matrix(np.outer(t, t) * K, tc='d')
        q = cvxopt.matrix(np.ones(nSamples) * -1, tc='d')
        # equivalent constraints: Ax = b
        A = cvxopt.matrix(t, (1, nSamples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        # unequivalent constraints: Gx <= h
        G = cvxopt.matrix(np.diag(np.ones(nSamples) * -1), tc='d')
        h = cvxopt.matrix(np.zeros(nSamples), tc='d')

        # call the solver with our arguments
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        threshold = 1e-5 # some small threshold
        sv = alpha > threshold
        positions = np.where(alpha>threshold)[0]# indices of actives data set point
        


        #TODO: extract support vectors etc.
        sv_position = positions[0] # we choose the first vector of the list of sv
       
        sum_w = 0
        for i in range(nSamples):
            sum_w += alpha[i]*t[i]*K[i,sv_position]
        
        print(sum_w)

        w0 = t[sv_position] -sum_w
        


        return [alpha, w0,positions]
    
    def discriminant(alpha,w0,X,t,Xnew,kernel=kernel.linearkernel):
        """discriminent function applied on a set of vector in colomn represented by Xnew using the trained on X,t SVM"""
        nFeatures, nSamples= X.shape
        nFeatures, nNew = Xnew.shape # not sure about the shape of Xnew: vector are said to be in column
        
        K = np.zeros([nSamples,nNew])
        for i in range(nSamples):
            for j in range(nNew):
                K[i, j] = kernel(X[:,i], Xnew[:,j])
                
        y = np.zeros(nNew)
        for i in range(nNew):
            target = np.sign(w0 +np.sum(np.array([alpha[j]*t[j]*K[j,i] for j in range(0,nSamples)])))
            y[i] = target
            
        return y
