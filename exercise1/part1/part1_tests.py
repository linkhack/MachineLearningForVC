import numpy as np
import matplotlib.pyplot as plt
import cv2
import mnist
import PerceptronOnlineTraining


def get_digits(digits, nr_samples_in_class=500):
    data_set_complete = mnist.train_images()
    targets_complete = mnist.train_labels()
    digit1_indices = np.where(targets_complete == digits[0])[0]
    digit2_indices = np.where(targets_complete == digits[1])[0]
    nr_samples_in_class = np.min([nr_samples_in_class, np.size(digit1_indices), np.size(digit2_indices)])
    digit1_indices = digit1_indices[:nr_samples_in_class]
    digit2_indices = digit2_indices[:nr_samples_in_class]
    indices = np.concatenate((digit1_indices, digit2_indices))
    targets = targets_complete[indices]
    targets[targets==digits[0]] = 1
    targets[targets==digits[1]] = -1
    data_set = data_set_complete[indices, :, :]
    return data_set, targets


def calculate_features(image_array):
    # biggest contour or what?
    shape = np.shape(image_array)
    feature_vector = np.zeros([2, shape[0]])
    multiple_contours = 0
    for idx, img in enumerate(image_array):
        img[img > 0] = 255
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = 0
        hull_area = 0

        for contour in contours:
            area += cv2.contourArea(contour)
            hull = cv2.convexHull(contour, returnPoints=True)
            hull_area += cv2.contourArea(hull)
        solidity = float(area) / hull_area
        feature_vector[:, idx] = np.array([area, solidity])
    print(multiple_contours)
    return feature_vector


def calculate_regionprops(image):
    regionprops = dict()
    image[image > 0] = 255
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # select biggest contour
    contour = max(contours, key=cv2.contourArea)

    # perimeter
    perimeter = cv2.arcLength(contour, closed=True)

    # area
    area = cv2.contourArea(contour)

    # bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h

    # extent
    rect_area = h * w
    extent = float(area)/rect_area

    # solidity
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area

    # equivalent Diameter
    equi_diameter = np.sqrt(4*area/np.pi)

    # Orientation
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    # eccentricity
    eccentricity = np.sqrt(1-float(ma*ma)/(MA*MA))

    regionprops['area'] = area
    regionprops['aspect_ratio'] = aspect_ratio
    regionprops['extent'] = extent
    regionprops['solidity'] = solidity
    regionprops['equivalent diameter'] = equi_diameter
    regionprops['orientation'] = angle
    regionprops['convex_area'] = hull_area
    regionprops['eccentricity'] = eccentricity
    regionprops['perimeter'] = len(contour)
    regionprops['perimeter'] = perimeter

    return regionprops


digits = [1, 5]
training_set, training_targets = get_digits(digits)

features = calculate_features(training_set)
plt.figure(0)
plt.scatter(features[0,:],features[1,:],c=training_targets)
plt.show()