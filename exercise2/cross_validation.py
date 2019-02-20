import exercise2.mnist_subset_and_feature_utils as mnist
import numpy as np
from exercise2.SVM import SVM
import exercise2.Kernel as kernel
import matplotlib.pyplot as plt


def main():
    print("start")
    # useful values
    nr_sets = 150
    nr_sample = 35
    digits = [0, 7]
    sigma_range = np.linspace(4, 50, 12)
    c_range = np.linspace(5, 50, 12)

    # variables for results
    cv_error = dict()

    # data sets

    [data, targets] = mnist.get_digits_compact(digits, nr_sample, nr_sets)

    # test set
    test_data, test_target = mnist.get_test_digits(digits,
                                                   100000)  # high value to get all the digits of the two classes
    test_data = mnist.transform_images(test_data).T  # transform into a matrix where images are in column

    svm = SVM()

    for sigma in sigma_range:
        for c in c_range:
            avg_error = svm.cross_validation(data, targets, kernel.rbfkernel, sigma, c, nr_sets)
            cv_error[(c, sigma)] = avg_error
            print(avg_error)

    print(cv_error)

    best_params = min(cv_error, key=cv_error.get)
    print(f"Best Parameters (c, sigma) = {best_params} with cv_error = {cv_error[best_params]}")

    sorted_list = sorted(cv_error, key=lambda key_value: key_value[0])
    error_matrix = np.array([cv_error[key] for key in sorted_list]).reshape(len(c_range), len(sigma_range))

    plt.figure()
    plt.contourf(error_matrix)
    plt.show()

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

    print(f"RBF-SVM has error = {np.mean(error_rbf_svm)}")
    print(f"linear-SVM has avg = {np.mean(error_linear_svm)}")

    rbf_error = []
    linear_error = []

    for i in range(nr_sets):
        data_set = data[i::nr_sets]
        target_set = targets[i::nr_sets]
        svm = SVM()
        svm.setSigma(best_params[1])

        # train
        [alpha, w0, sv_index] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, best_params[0])

        # test error
        predicted_test = np.sign(svm.discriminant(alpha, w0, sv_index, data_set, target_set, test_data))
        correct_test = (predicted_test == test_target)
        rbf_error.append(1 - np.mean(correct_test))

        # train
        [alpha, w0, sv_index] = svm.trainSVM(data_set, target_set, kernel.linearkernel)

        # test error
        predicted_test = np.sign(svm.discriminant(alpha, w0, sv_index, data_set, target_set, test_data))
        correct_test = (predicted_test == test_target)
        linear_error.append(1 - np.mean(correct_test))

    print(f"RBF-SVM has avg-error = {np.mean(rbf_error)} min = {np.min(rbf_error)} max = {np.max(rbf_error)}")
    print(
        f"linear-SVM has avg-error = {np.mean(linear_error)} min = {np.min(linear_error)} max = {np.max(linear_error)}")


if __name__ == '__main__':
    main()
