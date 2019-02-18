import exercise2.mnist_subset_and_feature_utils as mnist
import numpy as np
from exercise2.SVM import SVM
import exercise2.Kernel as kernel


def main():
    print("start")
    # useful values
    nr_sets = 150
    nr_sample = 35
    digits = [0, 7]
    sigma_range = range(5, 66, 5)
    c_range = range(30, 131, 10)

    # variables for results
    cv_error = {}

    # data sets

    [data, targets] = mnist.get_digits_compact(digits, nr_sample, nr_sets)

    # test set
    test_data, test_target = mnist.get_test_digits(digits,
                                                   100000)  # high value to get all the digits of the two classes
    test_data = mnist.transform_images(test_data).T  # transform into a matrix where images are in column

    svm = SVM()

    for c in c_range:
        for sigma in sigma_range:
            avg_error = svm.cross_validation(data, targets, kernel.rbfkernel, sigma, c, 150)
            cv_error[(c, sigma)] = avg_error

    print(cv_error)

    best_params = min(cv_error, key=cv_error.get)
    print(f"Best Parameters (c, sigma) = {best_params} with cv_error = {cv_error[best_params]}")

    # train final svm on whole set ???
    svm.setSigma(best_params[1])
    [alpha, w0, sv_index] = svm.trainSVM(data, targets, kernel.rbfkernel, best_params[0])
    predict = np.sign(svm.discriminant(alpha, w0, sv_index, data, targets, test_data))
    correct = (predict == test_target)
    error_rbf_svm = 1 - np.mean(correct)

    [alpha, w0, sv_index] = svm.trainSVM(data, targets, kernel.linearkernel)
    predict = np.sign(svm.discriminant(alpha, w0, sv_index, data, targets, test_data))
    correct = (predict == test_target)
    error_linear_svm = 1 - np.mean(correct)

    print(f"RBF-SVM has error of {error_rbf_svm} and linear SVM has error rate of {error_linear_svm}")


if __name__ == '__main__':
    main()
