from collections import Counter
import sys
import numpy as np


def fetch_data(input_file):
    """
    Read data from input file and return actual_data and classes
    :param input_file:
    :return: data_values - numpy 2D array representing data without classes
             classes - vector representing respective classes
    """
    data = np.loadtxt(fname=input_file)
    columns = data.shape[1]
    return data[:, 0:columns - 1], data[:, columns - 1]


def normalize_data(data, mean_values, std_values):
    """
    Perform normalization on data using mean_values and std_values
    Normalized_value = (original_value - mean_value)/standard deviation
    :param data: data matrix
    :param mean_values: vector representing mean for each column
    :param std_values: vector representing standard deviation for each column
    :return: normalized data
    """
    columns = data.shape[1]
    for i in range(columns - 1):
        # print(data[:, i])
        data[:, i] -= mean_values[i]
        data[:, i] /= std_values[i]
    return data


def get_distance(x, y):
    """
    Calculate distance between two sample data
    :param x: sample row 1
    :param y: sample row 2
    :return: distance between x and y
    """
    return np.linalg.norm(x - y)


def get_most_frequent(classes):
    """
    Returns list of most frequent classes from k_nearest_classes
    :param classes: list of k classes
    :return: list of most frequent classes
    """
    table = Counter(iter(classes)).most_common()
    max_freq = table[0][1]
    most_frequent_classes = []
    for i in range(len(table)):
        if table[i][1] != max_freq:
            break
        most_frequent_classes.append(table[i][0])
    return most_frequent_classes


if __name__ == '__main__':

    # Get training data file, test data file and value of K for KNN classification from arguments
    train_file, test_file, k = sys.argv[1:]

    k = int(k)
    # Read training and test data files
    train_data, train_classes = fetch_data(train_file)
    test_data, test_classes = fetch_data(test_file)

    # Calculate mean and standard deviation of each column of training data for normalization
    mean = np.mean(train_data, axis=1)
    std = np.std(train_data, axis=1)

    # Normalize training and testing data using mean and std values
    train_data = normalize_data(train_data, mean, std)
    test_data = normalize_data(test_data, mean, std)

    train_rows, train_cols = train_data.shape
    test_rows, test_cols = test_data.shape

    total_accuracy = 0
    for test_index in range(test_rows):
        distances = np.ones((train_rows, 2))

        # Calculate distance between training data and test row
        for train_index in range(train_rows):
            distance = get_distance(test_data[test_index], train_data[train_index])
            distances[train_index][0] = distance
            distances[train_index][1] = train_classes[train_index]

        # Find k nearest classes based on distances
        distances = distances[distances[:, 0].argsort()]
        k_nearest_classes = distances[:k, 1]

        tied_classes = get_most_frequent(k_nearest_classes)
        actual_class = test_classes[test_index]
        accuracy = 0
        if actual_class in tied_classes:
           accuracy = 1 / len(tied_classes)
           total_accuracy += accuracy
        print(str(test_index) + ':  ' + str(accuracy))
    print('classification accuracy= ' + str(total_accuracy / test_rows))
