import numpy as np
import matplotlib.pyplot as plt
import cv2
import mnist
import itertools
from exercise1.part1.PerceptronOnlineTraining import perceptron_online_training, perc


def get_digits(digits, nr_samples_in_class=500):
    data_set_complete = mnist.train_images()
    targets_complete = mnist.train_labels()
    digit1_indices = np.where(targets_complete == digits[0])[0]
    digit2_indices = np.where(targets_complete == digits[1])[0]
    nr_samples_in_class = np.min([nr_samples_in_class, np.size(digit1_indices), np.size(digit2_indices)])
    digit1_indices = digit1_indices[:nr_samples_in_class]
    digit2_indices = digit2_indices[:nr_samples_in_class]
    #important if first all 1 then all -1 gives not a good result
    indices = np.random.permutation(np.concatenate((digit1_indices, digit2_indices)))
    unsigned_targets = targets_complete[indices]
    targets = np.zeros(2*nr_samples_in_class)
    targets[unsigned_targets == digits[0]] = 1
    targets[unsigned_targets == digits[1]] = -1
    data_set = data_set_complete[indices, :, :]
    return data_set, targets


def calculate_features(image_array, props=None):
    # biggest contour or what?
    if props is None or len(props) < 2:
        props = ['solidity', 'eccentricity']
    shape = np.shape(image_array)
    # feature_vector = np.zeros([2, shape[0]])
    prop_dict = collect_regionprops(image_array, props)
    feature_vector = np.array([[feature_dict[props[0]] for feature_dict in prop_dict],
                               [feature_dict[props[1]] for feature_dict in prop_dict]])

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
    aspect_ratio = float(w) / h

    # extent
    rect_area = h * w
    extent = float(area) / rect_area

    # solidity
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area

    # equivalent Diameter
    equi_diameter = np.sqrt(4 * area / np.pi)

    # Orientation
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    # eccentricity
    eccentricity = np.sqrt(1 - float(MA * MA) / (ma * ma))

    # moments
    moments = cv2.moments(contour)

    regionprops['area'] = area
    regionprops['aspect_ratio'] = aspect_ratio
    regionprops['extent'] = extent
    regionprops['solidity'] = solidity
    regionprops['equivalent_diameter'] = equi_diameter
    regionprops['orientation'] = angle
    regionprops['convex_area'] = hull_area
    regionprops['eccentricity'] = eccentricity
    regionprops['perimeter'] = perimeter

    return regionprops


# regionprops for all test images
def collect_regionprops(image_array, props=None):
    if props is None:
        props = ['area', 'aspect_ratio', 'solidity', 'equivalent_diameter', 'orientation', 'convex_area',
                 'eccentricity', 'perimeter']
    prop_list = []
    for img in image_array:
        all_props = calculate_regionprops(img)
        props = {prop: all_props[prop] for prop in props}
        prop_list.append(props)
    return prop_list


def scatter_matrix_from_dict(prop_array, targets):
    numdata = len(prop_array)
    numvars = len(prop_array[0])
    keys = list(prop_array[0].keys())  # I hope this respects order ¯\_(ツ)_/¯

    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16, 16))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            data_x = [prop[keys[x]] for prop in prop_array]
            data_y = [prop[keys[y]] for prop in prop_array]
            axes[x, y].scatter(data_x, data_y, c=list(targets), s=1)

    # Label the diagonal subplots...
    for i, label in enumerate(keys):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig


def show_decision_boundary_simple(feature_matrix, weights):
    fig = plt.figure()


def augment_data(data):
    data_dimension = np.size(data, 0)
    nr_of_datapoints = np.size(data, 1)
    augmented_data = np.ones([data_dimension + 1, nr_of_datapoints])
    augmented_data[:-1, :] = data
    return augmented_data


def main():
    digits = [2, 5]
    training_set, training_targets = get_digits(digits,100)

    features = calculate_features(training_set)
    properties = collect_regionprops(training_set)

    features = augment_data(features)

    weights = perceptron_online_training(features, targets=training_targets, max_iterations=1000000)
    perc_result = perc(weights, features)
    correct = np.equal(perc_result, training_targets)
    correct_nr = sum(correct)
    print(str(correct_nr))
    correct_precentage = np.sum(correct) / np.size(features,1)
    print("Correct percentage:" + str(correct_precentage))

    plt.scatter(features[0,:], features[1,:],c = training_targets)
    # fig = scatter_matrix_from_dict(properties, training_targets)
    plt.show()
    return


if __name__ == '__main__':
    main()
