import numpy as np
import matplotlib.pyplot as plt
import cv2
import mnist
import PerceptronOnlineTraining


def get_digits(digits):
    data_set_complete = mnist.train_images()
    targets_complete = mnist.train_labels()
    digit_logic_indices = np.isin(targets_complete,digits)
    targets = targets_complete[np.isin(targets_complete,digits)]
    data_set = data_set_complete[digit_logic_indices, :, :]
    return data_set, targets

def calculate_features(image_array):
    shape = np.shape(image_array)


    return feature_vector

digits = [1, 5]
training_set, training_targets = get_digits(digits)
print(training_set)

cv2.imshow(training_set[1234,:,:])
cv2.waitKey()
cv2.destroyAllWindows()