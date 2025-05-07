import aeon
from aeon.datasets import load_classification
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import dtw as DTW_package #Name of package is actually dtw-python
from aeon.datasets.dataset_collections import get_available_tsc_datasets
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random

def smooth_random_function_with_zeros(num_points=100, seed_points=10, amplitude=1.0, smoothness=2):
    """
    Creates a random smooth function padded with zeros by the following procedure:
        It creates seed_points number of random points with random values and interpolates a polinomial.
        Cuts the function with a smooth window and padds with zeros.
        Smoothens the whole output with gaussian filter.

    Input:
        num_points: part of the domain where a function will be created.
        seed_points: number of random points to interpolate
        amplitude: maximum posible value for random points.
        smoothness: determines the variance of the gaussian filter. The largest the smoothest.
    Output:
        smooth_function: The output function will be of size 3*num_points. Will have zeros in the first and last num_points points.
    """

    # Generate a random points
    random_values = np.random.randn(seed_points) * amplitude

    # Smooth the random points using a Gaussian filter
    interp_fun = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, seed_points), random_values)

    # Apply a window function (Hann window) to force it to zero at the boundaries
    window = np.hanning(num_points)
    interp_fun *= window

    # Pads with zeros
    interp_fun = np.concatenate([np.zeros_like(interp_fun), interp_fun, np.zeros_like(interp_fun)])

    # Smooths by gaussian filtering
    smooth_function = gaussian_filter1d(interp_fun, sigma=smoothness)

    return smooth_function


def smooth_non_decreasing_fun(num_points=100, seed_points=10, amplitude=1.0, smoothness=2, hanning=True):
    """
    Creates a random smooth non-decreasing function to use as a deformation.
        It creates seed_points number of random points with random positive values and interpolates a polinomial.
        Cuts the polinomial with a smooth window and padds with zeros.
        Smoothens the polinomial with gaussian filter.
        Integrates the polinomial to create a non-decreasing function.
        Rescales the output to map [0,1] to [0.3,0.7]. (This is to avoid stretching functions too much so the support stays in [0,1].

    Input:
        num_points: part of the domain where a function will be created.
        seed_points: number of random points to interpolate
        amplitude: maximum posible value for random points.
        smoothness: determines the variance of the gaussian filter. The largest the smoothest.
        hanning: True (rescales the output)
    Output:
        smooth_function: The output function will be of size 3*num_points. Will have zeros in the first and last num_points points.
    """

    # Generate seed_points random positive values to interpolate
    random_values = np.abs(np.random.rand(seed_points))

    # Interpolate, cut the negative values and smooth
    interpolated_values = np.maximum(
        np.interp(np.linspace(0, 1, 3 * num_points), np.linspace(0, 1, seed_points), random_values), 0)
    smoothed_function = gaussian_filter1d(interpolated_values, sigma=smoothness)

    if hanning == True:
        # Apply a window function (Hann window )
        window = np.hanning(3 * num_points)
        smoothed_function *= window
        density_fun = smoothed_function / smoothed_function.sum()

        b = random.uniform(0, 0.33)
        a = random.uniform(0.67, 1) - b

        out = a * density_fun.cumsum() + b
    else:
        density_fun = smoothed_function / smoothed_function.sum()
        out = density_fun.cumsum()
    return out


def synthetic_dataset_builder(num_classes=5, num_samples_per_class=50, noise_level=0, create_template=True):
    """
    Creates a synthetic dataset from a template $phi$ and deformations $g$ according to formula $s(t)=phi(g(t))$

    Input:
        num_classes
        num_samples_per_class
        create_template: True (automatically creates own templates) -
                         Otherwise pass an array with num_classes number of rows,
                         where each row represent one template for a class.
    Output:
        synthetic_dataset: Array of size (num_classes * num_samples_per_class, 601), where the first column of
                           each row is the class of the row and the clumns [1:] are the signal.
    """

    # Example usage
    num_points = 200
    seed_points = 10  # Higher values make it smoother
    amplitude = 2.0  # Adjust the scale for amplitude

    classes = range(0, num_classes)
    samples_per_class = num_samples_per_class * np.ones_like(classes)

    dataset_functions = []
    for class_num in classes:
        template = smooth_random_function_with_zeros(num_points, seed_points, amplitude)
        # plt.plot(template,'r')
        if type(create_template) != bool:
            assert create_template.shape[0] == num_classes, 'Not enough templates provided'
            template = create_template[class_num]
        for num_sample in range(samples_per_class[class_num]):
            random_deformation = smooth_non_decreasing_fun(num_points, seed_points, amplitude)
            sample_fun = np.interp(random_deformation, np.linspace(0, 1, len(template)), template)
            noise = noise_level * np.random.randn(*sample_fun.shape)
            sample_fun += noise
            # plt.plot(sample_fun)
            # Add class at position 0
            sample_fun = np.insert(sample_fun, 0, class_num)
            dataset_functions.append(sample_fun)
        # plt.show()

    synthetic_dataset = np.vstack(dataset_functions)
    return synthetic_dataset


