import exercise2.mnist_subset_and_feature_utils as mnist
import numpy as np
from exercise2.SVM import SVM
import exercise2.Kernel as kernel
import matplotlib.pyplot as plt


def main():
    ## useful values
    nr_sets = 150
    nr_sample = 35
    digits = [0, 7]
    sigma_range = np.linspace(5,55,11)
    c_range = np.linspace(5,105,11)

    # variables for results
    test_error = []
    train_error = []
    nr_support_vectors = []

    avg_test_error = []
    avg_train_error = []
    avg_nr_support_vectors = []

    # data sets

    [data, targets] = mnist.get_digits_compact(digits, nr_sample, nr_sets)

    # test set
    test_data, test_target = mnist.get_test_digits(digits,
                                                   100000)  # high value to get all the digits of the two classes
    test_data = mnist.transform_images(test_data).T  # transform into a matrix where images are in column
    nr_test_samples = np.size(test_target)

    for c in c_range:
        for sigma in sigma_range:
            test_error.clear()
            train_error.clear()
            nr_support_vectors.clear()

            for i in range(nr_sets):
                data_set = data[i::nr_sets]
                target_set = targets[i::nr_sets]
                svm = SVM()
                svm.setSigma(sigma)

                # train
                [alpha, w0, sv_index] = svm.trainSVM(data_set, target_set, kernel.rbfkernel, c)

                # test error
                predicted_test = np.sign(svm.discriminant(alpha, w0, sv_index, data_set, target_set, test_data))
                correct_test = (predicted_test == test_target)
                test_error.append(1-np.mean(correct_test))

                # train error
                predicted_train = np.sign(svm.discriminant(alpha, w0, sv_index, data_set, target_set, data_set))
                correct_train = (predicted_train == target_set)
                train_error.append(1-np.mean(correct_train))

                # number of support vectors
                nr_support_vectors.append(np.sum(sv_index))
            # Calculate averages
            avg_test_error.append(np.mean(test_error))
            avg_train_error.append(np.mean(train_error))
            avg_nr_support_vectors.append(np.mean(nr_support_vectors))
            print(f"For sigma = {sigma} and c={c} averages: test error = {np.mean(test_error)}, "
                  f"training error={np.mean(train_error)}, number of support vectors = {np.mean(nr_support_vectors)}.")
            # print(f"For sigma = {sigma} and c={c} maximum: test error = {max(test_error)}, "
            #      f"training error={max(train_error)}, number of support vectors = {max(nr_support_vectors)}.")

    # plots
    plt.figure()
    plt.scatter(avg_train_error, avg_test_error)
    plt.show()

    plt.figure()
    plt.scatter(avg_nr_support_vectors, avg_test_error)
    plt.show()

    plt.figure()
    for i in enumerate(c_range):
        plt.plot(sigma_range, avg_test_error[(len(sigma_range)*i[0]):(len(sigma_range)*(i[0]+1))], label=f"c={i[1]}")
    plt.legend()
    plt.title("Test error for sigma and c")
    plt.xlabel("Sigma")
    plt.ylabel("Test Error")
    plt.show()

    for i in enumerate(c_range):
        plt.plot(sigma_range, avg_train_error[(len(sigma_range)*i[0]):(len(sigma_range)*(i[0]+1))], label=f"c={i[1]}")
    plt.legend()
    plt.title("Training Error for sigma and c")
    plt.xlabel("Sigma")
    plt.ylabel("Test Error")
    plt.show()


if __name__ == '__main__':
    main()
