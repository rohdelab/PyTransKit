import aeon
from aeon.datasets import load_classification
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import dtw as DTW_package #Name of package is actually dtw-python
from aeon.datasets.dataset_collections import get_available_tsc_datasets
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random


def select_samples(data, num_samples, seed=None):
    '''
    Creates training and test splits with num_samples number of samples per class in the training. The rest remains in test.

    Inputs:
        data: labeled data with label on first column.
        num_samples: number of samples per class that are going to be selected for training set.
    Output:
        train and test splits for the dataset.
    '''

    # It will randomly extract num_samples from each class present in data
    if seed != None:
        random.seed(seed)

    # Check the number of different classes in first column of data
    ul = np.unique(data[:, 0])

    # Separate the data by class
    classes = []
    for i in range(len(ul)):
        classes.append(data[data[:, 0] == ul[i]])
    samples = []

    # Randomly extract num_samples training samples per class and eliminate those instances from test_data
    for i in range(len(ul)):
        class_samples = []
        for j in range(num_samples):
            idx = random.randint(0, len(classes[i]) - 1)
            sample = classes[i][idx, :].reshape(1, -1)
            class_samples.append(sample)
            classes[i] = np.delete(classes[i], (idx), axis=0)
        class_samples = np.array(np.concatenate(class_samples, axis=0))
        samples.append(class_samples)

    training_samples = np.array(np.concatenate(samples, axis=0))
    test_data = np.concatenate(classes, axis=0)

    return training_samples, test_data


def print_samples(data, samples_per_class):
    """
    Given a dataset data, plots samples_per_class number of signals from each class.
    It automatically creates as many subplots as distinct classes in the dataset.
    """
    ul = np.unique(data[:, 0])
    # Separate the data by class
    classes = []
    for i in range(len(ul)):
        classes.append(data[data[:, 0] == ul[i]])
        if len(classes[i]) < samples_per_class:
            print('Not enough samples per class')

    fig, axs = plt.subplots(1, len(ul), figsize=(20, 8), constrained_layout=True)
    for i in range(len(axs)):
        for j in range(samples_per_class):
            axs[i].plot(classes[i][j, 1:])
        axs[i].set_title(f'{samples_per_class} samples from Class {i}')
    return None