import cvxopt
import numpy as np
import exercise2.Kernel as k


class SVM:

    def __init__(self):
        """contructor"""
        # to be used with rbfKernels see exercise2.Kernel for function
        self.sigma = 5.0

    def setSigma(self, sigma):
        self.sigma = sigma

    def trainSVM(self, x, t, kernel=k.linearkernel, c=None):
        """
        Keyword arguments:
        :param x: ndarray
        :param t: ndarray
        :param kernel:
        :param c: float

        :return [alpha, w0]:
        The vector alpha holds the optimal dual parameter values, i.e., the lagrange multipliers αi for all N input vectors
        w0 is the offset of the decision plane, which can be computed using alpha and one support vector (The data points for which the (dual variables) αi > 0 are called support vectors.)
        """
        self.c = c
        self.kernel = kernel
        # self.sigma  = sigma

        # get some dimensions
        nFeatures, nSamples = x.shape

        # Given a set 𝑉 of 𝑚 vectors (points in ℝ𝑛), the Gram matrix 𝐺 is the matrix of all possible inner products of 𝑉,
        # i.e., 𝑔𝑖𝑗=𝑣𝑇𝑖𝑣𝑗, where 𝐴𝑇 denotes the transpose.
        # kernel functions can be represented as Gram matrices for the dual formulation of SVM:
        # 𝑚𝑎𝑥𝛼𝑖≥0∑𝑖𝛼𝑖−12∑𝑗𝑘𝛼𝑗𝛼𝑘𝑦𝑗𝑦𝑘(𝑥𝑇𝑗𝑥𝑘)
        #
        gram_matrix = np.zeros((nSamples, nSamples))
        for i in range(nSamples):
            for j in range(nSamples):
                if self.kernel == k.linearkernel:
                    gram_matrix[i, j] = self.kernel(x[:, i], x[:, j])
                elif self.kernel == k.rbfkernel:
                    gram_matrix[i, j] = self.kernel(x[:, i], x[:, j], self.sigma)
                else:
                    return 0
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
        if self.c is None:
            # hard margin
            G = cvxopt.matrix(np.diag(np.ones(nSamples) * -1), tc='d')
            h = cvxopt.matrix(np.zeros(nSamples), tc='d')
        else:
            ##with slack variable, a.k.a. soft margin!
            partA = np.diag(np.ones(nSamples) * -1)
            partB = np.identity(nSamples)
            G = cvxopt.matrix(np.vstack((partA, partB)), tc='d')

            partA = np.zeros(nSamples)
            partB = np.ones(nSamples) * self.c
            h = cvxopt.matrix(np.hstack((partA, partB)), tc='d')

        # call the solver with our arguments
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        alpha = np.ravel(solution['x'])
        slack = np.ravel(solution['z'])


        # Support vectors have non zero lagrange multipliers
        sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

        # position index of support vectors in alpha array
        ind = np.arange(len(alpha))[sv_index]

        # get support vectors and corresponding x and label values
        sv = alpha[sv_index]
        # sv_X = x[:,sv_index]
        sv_T = t[sv_index]

        # calculate w0
        main_sv_index = np.argmax(alpha)
        if self.c is None:
            w0 = t[main_sv_index] - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])
        else:
            w0 = t[main_sv_index]*(1 - slack[main_sv_index+len(t)])  # interested in associated slack variable
            w0 = w0 - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])

        return [alpha, w0, sv_index]

    def discriminant(self, alpha, w0, X, t, Xnew):
        """
        d (x) = (SUM αi K(xi , x)) + w0
        whereas K is kernel function
        :type Xnew: ndarray
        :type t: ndarray
        :type X: ndarray
        :type w0: ndarray
        :type alpha: ndarray


        """

        # Support vectors have non zero lagrange multipliers
        sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

        # get support vectors and corresponding x and label values
        sv = alpha[sv_index]
        sv_X = X[sv_index, :]
        sv_T = t[sv_index]

        # pre allocate result label vector
        y_predict = np.zeros(len(Xnew))

        # label all xNew values
        # first * is componentwise, second is matrix multiplication
        # self.kernel is a nr_of_data * nr_of_sv matrix
        # gives nr_of_data many values
        y_predict = sv * sv_T @ self.kernel(Xnew, sv_X).T

        #statemant above written our

        # for i in range(len(Xnew)):
        #     s = 0
        #     s2 = np.dot(sv*sv_T,self.kernel(Xnew[i],sv_X))
        #     for a, sv_t, sv_x in zip(sv, sv_T, sv_X):
        #         if self.kernel == k.linearkernel:

        #             s += a * sv_t * self.kernel(Xnew[i], sv_x)
        #         elif self.kernel == k.rbfkernel:
        #             s += a * sv_t * self.kernel(Xnew[i], sv_x, self.sigma)
        #         else:
        #             return 0
        #
        #     y_predict[i] = s

        # return np.sign(y_predict + w0)
        # just return d(x)
        return y_predict + w0
