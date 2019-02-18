import exercise2.mnist_subset_and_feature_utils as mnist
import numpy as np
from exercise2.SVM import SVM
import exercise2.Kernel as kernel
import exercise1.part1.Perceptron as perc
import matplotlib.pyplot as plt


def main():
    ## useful values
    nr_sets = 150
    nr_sample = 35
    digits = [0, 7]


    [data, targets] = mnist.get_digits_compact(digits,nr_sample,nr_sets)

    # test set
    test_data, test_target = mnist.get_test_digits(digits, 100000)  # high value to get all the digits of the two classes
    test_data = mnist.transform_images(test_data).T  # transform into a matrix where images are in column
    nr_test_samples = np.size(test_target)

    svm = SVM()

    error_svm = []
    error_perceptron = []

    for i in range(nr_sets):
        data_set = data[i::nr_sets]
        target_set = targets[i::nr_sets]
        # svm
        # train
        [alpha, w0, support_vectors] = svm.trainSVM(data[i::nr_sets], targets[i::nr_sets], kernel.linearkernel)

        # predict
        predicted_svm = np.sign(svm.discriminant(alpha,w0,support_vectors,data[i::nr_sets],targets[i::nr_sets],test_data))

        # calculate error
        correct = (predicted_svm == test_target)
        error_svm.append(1-np.sum(correct)/nr_test_samples)
        # print(f"SVM has an error of {1-np.sum(correct)/nr_test_samples}.")

        #perceptron
        ## perceptron
        data_set_perc = mnist.augment_data(data_set.T)
        test_data_perc = mnist.augment_data(test_data.T)

        weights = perc.percTrain(data_set_perc, target_set, 1000, True)

        # is linearly seperable?
        # test_predict = perc.perc(weights,data_set_perc)
        # if not np.all(test_predict == target_set):
        #     print("Ooopsie")

        predicted_targets_perc = perc.perc(weights, test_data_perc)  # calculation of predicted target for perceptron
        correct_perc = (predicted_targets_perc == test_target)
        error_perceptron.append(1-np.sum(correct_perc)/nr_test_samples)
        # print(f"Perceptron has an error of {1-np.sum(correct_perc)/nr_test_samples}.")

    # Evaluate average error rate
    error_rate_svm = sum(error_svm)/nr_sets
    print(f"Linear SVM without slack has average error of {error_rate_svm}")
    print(f"and maximal error of {max(error_svm)}.")

    error_rate_perc = sum(error_perceptron)/nr_sets
    print(f"Perceptron has average error of {error_rate_perc}")
    print(f"and maximal error of {max(error_perceptron)}.")

    # show error rate over sets
    x = range(0, nr_sets)
    plt.figure()
    plt.scatter(x, error_svm, c="b", marker='.')
    plt.scatter(x, error_perceptron, c="r", marker='.')
    plt.axhline(error_rate_svm, color='b')
    plt.axhline(error_rate_perc, color='r')
    # plt.scatter(x, SVM_soft_error_rate, c="y", marker='+')
    plt.show()

    plt.figure()
    plt.boxplot([error_svm,error_perceptron])
    plt.show()


if __name__ == '__main__':
    main()
