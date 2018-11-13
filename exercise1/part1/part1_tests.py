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
    feature_vector = np.zeros([2,shape[1]])
    im2, contours, hierarchy = cv2.findContours(image_array,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour,returnPoints = True)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    feature_vector = np.array([[area], [solidity]])
    return feature_vector

digits = [1, 5]
training_set, training_targets = get_digits(digits)

features = calculate_features(training_set[2])
print(features)