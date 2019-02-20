import cvxopt
import numpy as np
import exercise2.Kernel as k


class SVM:

    def __init__(self):
        """contructor"""
        # to be used with rbfKernels see exercise2.Kernel for function
        self.sigma = None
        # for cross-validation
        self.databasecv = np.zeros(2)
        self.targetbasecv = np.zeros(2)
        self.complete_gram = None

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
        nSamples, nFeatures = x.shape

        # Given a set 𝑉 of 𝑚 vectors (points in ℝ𝑛), the Gram matrix 𝐺 is the matrix of all possible inner products of 𝑉,
        # i.e., 𝑔𝑖𝑗=𝑣𝑇𝑖𝑣𝑗, where 𝐴𝑇 denotes the transpose.
        # kernel functions can be represented as Gram matrices for the dual formulation of SVM:
        # 𝑚𝑎𝑥𝛼𝑖≥0∑𝑖𝛼𝑖−12∑𝑗𝑘𝛼𝑗𝛼𝑘𝑦𝑗𝑦𝑘(𝑥𝑇𝑗𝑥𝑘)
        #
        gram_matrix = np.zeros((nSamples, nSamples))
        if self.kernel == k.linearkernel:
            gram_matrix = self.kernel(x, x)
        elif self.kernel == k.rbfkernel:
            gram_matrix = self.kernel(x, x, self.sigma)
        else:
            return 0
        #
        # for i in range(nSamples):
        #     for j in range(nSamples):
        #         if self.kernel == k.linearkernel:
        #             gram_matrix[i, j] = self.kernel(x[:, i], x[:, j])
        #         elif self.kernel == k.rbfkernel:
        #             gram_matrix[i, j] = self.kernel(x[:, i], x[:, j], self.sigma)
        #         else:
        #             return 0
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
            # with slack variable, a.k.a. soft margin!
            partA = np.diag(np.ones(nSamples) * -1)
            partB = np.identity(nSamples)
            G = cvxopt.matrix(np.vstack((partA, partB)), tc='d')

            partA = np.zeros(nSamples)
            partB = np.ones(nSamples) * self.c
            h = cvxopt.matrix(np.hstack((partA, partB)), tc='d')

        # call the solver with our arguments
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        alpha = np.ravel(solution['x'])
        slack = np.ravel(solution['z'])

        if self.c is None:
            # Support vectors have non zero lagrange multipliers
            sv_index = alpha > 1e-4  # some small threshold a little bit greater than 0, [> 0  was too crowded]

            # position index of support vectors in alpha array
            ind = np.arange(len(alpha))[sv_index]

            # get support vectors and corresponding x and label values
            sv = alpha[sv_index]
            # sv_X = x[:,sv_index]
            sv_T = t[sv_index]

            # calculate w0
            main_sv_index = np.argmax(alpha)
            w0 = t[main_sv_index] - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])
        else:
            sv_index = (alpha > 1e-4)
            small_index = (alpha < self.c - 1e-5)
            alpha_support = np.zeros_like(alpha)
            alpha_support[small_index & sv_index] = alpha[small_index & sv_index]
            if np.all(~(sv_index & small_index)):
                print("No support vectors found")
            # return [alpha, 0, sv_index]
            # get support vectors and corresponding x and label values

            sv = alpha[sv_index]
            # sv_X = x[:,sv_index]
            sv_T = t[sv_index]

            # calculate w0
            main_sv_index = np.argmax(alpha_support)

            # w0 = t[main_sv_index] - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])
            w0 = t[main_sv_index] * (1 - slack[main_sv_index + len(t)])  # interested in associated slack variable
            w0 = w0 - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])

        return [alpha, w0, sv_index]

    def discriminant(self, alpha, w0, sv_index, X, t, Xnew):
        """
        d (x) = (SUM αi K(xi , x)) + w0
        whereas K is kernel function
        :param sv_index: logical indices of support vectors
        :type Xnew: ndarray
        :type t: ndarray
        :type X: ndarray
        :type w0: ndarray
        :type alpha: ndarray
        """

        # get support vectors and corresponding x and label values
        sv = alpha[sv_index]
        sv_X = X[sv_index, :]
        sv_T = t[sv_index]

        if self.kernel == k.linearkernel:

            y_predict = sv * sv_T @ self.kernel(sv_X, Xnew)
        elif self.kernel == k.rbfkernel:
            y_predict = sv * sv_T @ self.kernel(sv_X, Xnew, self.sigma)
        else:
            return 0

        return y_predict + w0

    def cross_validation(self, x, t, kernel, sigma, c, nr_sets):
        print(c)
        print(sigma)
        error_rate = []
        # calculate gram matrix only if sigma changes
        # gram matrix needed for all nr_sets subsets stays the same and by indexing the right positions
        # we can use this matrix for constructing the training matrices and also for the evaluation of the discriminant.
        if self.sigma != sigma:
            if kernel == k.linearkernel:
                self.complete_gram = kernel(x, x)
            elif kernel == k.rbfkernel:
                self.complete_gram = kernel(x, x, sigma)
            else:
                return 0
        self.setSigma(sigma)
        for i in range(nr_sets):
            # train_slice = slice(i, None, nr_sets)
            train_index = np.zeros_like(t, dtype=bool)
            train_index[i::nr_sets] = True
            test_index = ~train_index
            train_targets = t[train_index]

            nSamples = np.sum(train_index)
            #################################
            # Training
            #################################
            gram_matrix = self.complete_gram[train_index, :][:, train_index]

            P = cvxopt.matrix(np.outer(train_targets, train_targets) * gram_matrix, tc='d')
            q = cvxopt.matrix(np.ones(nSamples) * -1, tc='d')
            # equivalent constraints: Ax = b
            A = cvxopt.matrix(train_targets, (1, nSamples), tc='d')
            b = cvxopt.matrix(0.0, tc='d')
            # unequivalent constraints: Gx <= h
            if c is None:
                # hard margin
                G = cvxopt.matrix(np.diag(np.ones(nSamples) * -1), tc='d')
                h = cvxopt.matrix(np.zeros(nSamples), tc='d')
            else:
                # with slack variable, a.k.a. soft margin!
                partA = np.diag(np.ones(nSamples) * -1)
                partB = np.identity(nSamples)
                G = cvxopt.matrix(np.vstack((partA, partB)), tc='d')

                partA = np.zeros(nSamples)
                partB = np.ones(nSamples) * c
                h = cvxopt.matrix(np.hstack((partA, partB)), tc='d')

            # call the solver with our arguments
            cvxopt.solvers.options['show_progress'] = False
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            # Lagrange multipliers
            alpha = np.ravel(solution['x'])
            slack = np.ravel(solution['z'])

            # Calulating w0
            if c is None:
                # Support vectors have non zero lagrange multipliers
                sv_index = alpha > 1e-4  # some small threshold a little bit greater than 0, [> 0  was too crowded]

                # position index of support vectors in alpha array
                ind = np.arange(len(alpha))[sv_index]

                # get support vectors and corresponding x and label values
                sv = alpha[sv_index]
                # sv_X = x[:,sv_index]
                sv_T = train_targets[sv_index]

                # calculate w0
                main_sv_index = np.argmax(alpha)
                w0 = train_targets[main_sv_index] - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])
            else:
                sv_index = (alpha > 1e-4)
                small_index = (alpha < c - 1e-5)
                alpha_support = np.zeros_like(alpha)
                alpha_support[small_index & sv_index] = alpha[small_index & sv_index]
                if np.all(~(sv_index & small_index)):
                    print("No support vectors found")

                # get support vectors and corresponding x and label values

                sv = alpha[sv_index]
                # sv_X = x[:,sv_index]
                sv_T = train_targets[sv_index]

                # calculate w0
                main_sv_index = np.argmax(alpha_support)

                # w0 = t[main_sv_index] - np.sum(sv * sv_T * gram_matrix[sv_index, main_sv_index])
                w0 = train_targets[main_sv_index] * (
                        1 - slack[main_sv_index + nSamples])  # interested in associated slack variable
                w0 = w0 - np.sum(sv * sv_T * gram_matrix[sv_index, :][:, main_sv_index])

            #################################
            # Testing
            #################################

            # lift indices
            sv_index_whole = np.zeros_like(train_index, dtype=bool)
            sv_index_whole[i::nr_sets] = sv_index

            y_predict = np.sign(w0 + sv * sv_T @ self.complete_gram[sv_index_whole, :][:, test_index])

            # calculate error rate
            correct = (y_predict == t[test_index])
            error_rate.append(1 - np.mean(correct))

        return np.mean(error_rate)
