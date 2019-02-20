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
        self.completeGram = np.zeros(0)

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

    def setCV(self, databasecv, targetbasecv):
        """ give the database of 150 datasets of 75 samples as an attribut of the SVM"""
        self.databasecv = databasecv
        self.targetbasecv = targetbasecv

    def setGram(self, kernel=k.linearkernel):
        """calculation of the whole gram matrixfor the 150 sets of 75 in case of the CV"""
        nSet, nFeature, nSample = self.databasecv.shape
        Dim = nSet * nSample
        self.kernel = kernel
        gram_matrix = np.zeros((Dim, Dim))

        for i in range(0, nSet):
            for j in range(0, nSample):
                for p in range(0, i - 1):
                    for q in range(0, nSample):
                        pos_x = i * nSample + j
                        pos_y = p * nSample + q
                        if kernel == k.linearkernel:
                            value = self.kernel(self.databasecv[i, :, j], self.databasecv[p, :,
                                                                          q])  # calculation of the kernel between the jth sample of the ith dataset and the qth sample of the jth dataset
                        elif kernel == k.rbfkernel:
                            value = self.kernel(self.databasecv[i, :, j], self.databasecv[p, :, q], self.sigma)
                        else:
                            value = 0
                        gram_matrix[pos_x, pos_y] = value
                        gram_matrix[pos_y, pos_x] = value  # gram matrix is symetrical
                p = i
                for q in range(0, j + 1):
                    pos_x = i * nSample + j
                    pos_y = p * nSample + q
                    if kernel == k.linearkernel:
                        value = self.kernel(self.databasecv[i, :, j], self.databasecv[p, :,
                                                                      q])  # calculation of the kernel between the jth sample of the ith dataset and the qth sample of the jth dataset
                    elif kernel == k.rbfkernel:
                        value = self.kernel(self.databasecv[i, :, j], self.databasecv[p, :, q], self.sigma)
                    else:
                        value = 0
                    gram_matrix[pos_x, pos_y] = value
                    gram_matrix[pos_y, pos_x] = value  # gram matrix is symetrical

        self.gram_matrix = gram_matrix

    def trainSVM_CV(self, k, c=None):
        """ k corresponds to the rank of the dataset in the  self.databasecv which will be used as the test set"""
        self.c = c
        # self.sigma  = sigma

        # get some dimensions
        nSet, nFeature, nSample = self.databasecv.shape

        Dim = (nSet - 1) * nSample

        gram_matrix_2 = np.zeros((Dim, Dim))

        """ i don't know if there is a numpy function that can extract a sub matrix from a matrix 
        getting rid of specific columns and lines, if yes the following part can be replaced"""

        for i in range(0, k):  # we mustn't take into account the kth dataset
            for j in range(0, nSample):
                for p in range(0, i - 1):
                    for q in range(0, nSample):
                        pos_x = i * nSample + j
                        pos_y = p * nSample + q
                        value = self.gram_matrix[pos_x, pos_y]  #
                        gram_matrix_2[pos_x, pos_y] = value
                        gram_matrix_2[pos_y, pos_x] = value
                p = i
                for q in range(0, j + 1):
                    pos_x = i * nSample + j
                    pos_y = p * nSample + q
                    value = self.gram_matrix[pos_x, pos_y]
                    gram_matrix_2[pos_x, pos_y] = value
                    gram_matrix_2[pos_y, pos_x] = value  # gram matrix is symetrical

        for i in range(k + 1, nSet):  # we skip the k th dataset
            for j in range(0, nSample):
                for p in range(0, k):
                    for q in range(0, nSample):
                        pos_x = (i - 1) * nSample + j  # i-1 as we skip the kth value
                        pos_y = p * nSample + q
                        value = self.gram_matrix[pos_x + nSample, pos_y]  #
                        gram_matrix_2[pos_x, pos_y] = value
                        gram_matrix_2[pos_y, pos_x] = value

                for p in range(k + 1, i):
                    for q in range(0, nSample):
                        pos_x = (i - 1) * nSample + j  # i-1 as we skip the kth value
                        pos_y = (p - 1) * nSample + q
                        value = self.gram_matrix[pos_x + nSample, pos_y + nSample]  #
                        gram_matrix_2[pos_x, pos_y] = value
                        gram_matrix_2[pos_y, pos_x] = value

                p = i
                for q in range(0, j + 1):
                    pos_x = (i - 1) * nSample + j
                    pos_y = (p - 1) * nSample + q
                    value = self.gram_matrix[pos_x + nSample, pos_y + nSample]
                    gram_matrix_2[pos_x, pos_y] = value
                    gram_matrix_2[pos_y, pos_x] = value  # gram matrix is symetrical

        """End of calculation of Gram Matrix , we then use the same algorithm as for Train_SVM """

        # calculation of the target set of (nSet-1)*nSample value

        t = np.zeros((Dim, 1))
        compt = 0

        for i in range(0, nSet):
            if i != k:
                for j in range(0, nSample):
                    t[nSample * compt + j, 1] = self.targetbasecv[i, j]
                compt += 1  # take into account if we have skipped k or not

        # prepare arguments for solver:
        nSamples = (nSet - 1) * nSample  # total number of training exemples

        # we need to minify a constrained quadratic program
        # min x : 1/2 xT P x + qT x
        P = cvxopt.matrix(np.outer(t, t) * gram_matrix_2, tc='d')
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
            small_index = (alpha < self.c - 1e-4)
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

    def discriminant_CV(self, alpha, w0, k):
        """ calculation of the predicted values for the  kth dataset for a SVM trained on all the databases except the kth"""
        # Support vectors have non zero lagrange multipliers
        sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

        # now we want to calculate the index in the whole gram matrix as for the training we skipped the kth dataset

        nSet, nFeature, nSample = self.databasecv.shape

        nbr_sv = np.size(sv_index)

        index = np.zeros((nbr_sv, 1))

        for i in range(0, nbr_sv):
            pos = sv_index[i]
            if pos < k * nSample:
                index[i] = sv_index[i]
            else:
                index[i] = sv_index[
                               i] + nSample  # if the SV belongs to a dataset  after the kth position, we have to add the number of datasample in the kth data to have its position in the big gram matrix

        predict = np.zeros((nSample, 1))

        for i in range(0, nSample):
            value = w0
            for j in range(0, nbr_sv):
                value += alpha[j] * self.gram_matrix[k * nSample + i, index[j]]
            predict[i] = value

        return predict

    def cross_validation(self, x, t, kernel, sigma, c, nr_sets):
        print(c)
        print(sigma)
        error_rate = []
        if self.sigma != sigma:
            if kernel == k.linearkernel:
                self.completeGram = kernel(x, x)
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
