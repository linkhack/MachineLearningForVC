import cvxopt
import numpy as np
import exercise2.Kernel as kernel


class SVM:

    def __init__(self):
        """contructor"""

    def trainSVM(self, x, t, kernel=kernel.linearkernel, sigma = -1, C = -1):
        """
        Keyword arguments:
        X -- input vector
        t -- labels
        kernel -- kernel function. see Kernel class,

        returns [sv, sv_x, sv_t, w0]
        The vector alpha holds the optimal dual parameter values, i.e., the lagrange multipliers αi for all N input vectors
        w0 is the offset of the decision plane, which can be computed using alpha and one support vector (The data points for which the (dual variables) αi > 0 are called support vectors.)
        """
        self.kernel = kernel
        self.sigma  = sigma

        # get some dimensions
        nSamples, nFeatures = x.shape

        # Given a set 𝑉 of 𝑚 vectors (points in ℝ𝑛), the Gram matrix 𝐺 is the matrix of all possible inner products of 𝑉,
        # i.e., 𝑔𝑖𝑗=𝑣𝑇𝑖𝑣𝑗, where 𝐴𝑇 denotes the transpose.
        # kernel functions can be represented as Gram matrices for the dual formulation of SVM:
        # 𝑚𝑎𝑥𝛼𝑖≥0∑𝑖𝛼𝑖−12∑𝑗𝑘𝛼𝑗𝛼𝑘𝑦𝑗𝑦𝑘(𝑥𝑇𝑗𝑥𝑘)
        #
        gram_matrix = np.zeros((nSamples, nSamples))
        if sigma == -1:
            for i in range(nSamples):
                for j in range(nSamples):
                    gram_matrix[i, j] = self.kernel(x[i], x[j])
        else:
            for i in range(nSamples):
                for j in range(nSamples):
                    gram_matrix[i, j] = self.kernel(x[i], x[j], sigma)

        # FYI tc='d' specifies double as matrix content type!

        # prepare arguments for solver:
        # we need to minify a constrained quadratic program
        # min x : 1/2 xT P x + qT x
        P = cvxopt.matrix(np.outer(t, t) * gram_matrix, tc='d')
        q = cvxopt.matrix(np.ones(nSamples) * -1, tc='d')
        # equivalent constraints: Ax = b
        A = cvxopt.matrix(t, (1, nSamples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        # unequivalent constraints: Gx <= h
        if C== -1:
            G = cvxopt.matrix(np.diag(np.ones(nSamples) * -1), tc='d')
            h = cvxopt.matrix(np.zeros(nSamples), tc='d')
        else:
            D1= cvxopt.matrix(np.diag(np.ones(nSamples) * -1), tc='d')
            D2= cvxopt.matrix(np.diag(np.ones(nSamples) ), tc='d')
            G = cvxopt.matrix([D1,D2])
            h1 = cvxopt.matrix(np.zeros(nSamples), tc='d')
            h2 = cvxopt.matrix(np.ones(nSamples) * C, tc='d')
            h = cvxopt.matrix([h1,h2])
            
        # call the solver with our arguments
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

        # position index of support vectors in alpha array
        ind = np.arange(len(alpha))[sv_index]

        # get support vectors and corresponding x and label values
        sv = alpha[sv_index]
        sv_X = x[sv_index]
        sv_T = t[sv_index]

        # calculate w0
        w0 = 0
        for n in range(len(sv)):
            w0 = w0 + sv_T[n]
            w0 = w0 - np.sum(sv * sv_T * gram_matrix[ind[n], sv_index])
        w0 = w0 / len(sv)

        return [alpha, w0]

    def discriminant(self, alpha, w0, X, t, Xnew):
        """
        d (x) = (SUM αi K(xi , x)) + w0
        whereas K is kernel function


        """
        nSamples, nFeatures = X.shape
        gram_matrix = np.zeros((nSamples, nSamples))

        for i in range(nSamples):
            for j in range(nSamples):
                gram_matrix[i, j] = self.kernel(X[i], X[j])

        # Support vectors have non zero lagrange multipliers
        sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

        # get support vectors and corresponding x and label values
        sv = alpha[sv_index]
        sv_X = X[sv_index]
        sv_T = t[sv_index]

        # pre allocate result label vector
        y_predict = np.zeros(len(Xnew))

        # label all xNew values
        for i in range(len(Xnew)):
            s = 0
            for a, sv_t, sv_x in zip(sv, sv_T, sv_X):
                s += a * sv_t * self.kernel(Xnew[i], sv_x)
            y_predict[i] = s

        # return np.sign(y_predict + w0)
        # just return d(x)
        return y_predict + w0
