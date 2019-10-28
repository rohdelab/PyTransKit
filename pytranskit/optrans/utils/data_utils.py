import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

from .validation import check_array, assert_equal_shape


def signal_to_pdf(input, sigma=0., epsilon=1e-8, total=1.):
    """
    Get the (smoothed) probability density function of a signal.

    Performs the following operations:
    1. Smooth sigma with a Gaussian filter
    2. Normalize signal such that it sums to 1
    3. Add epsilon to ensure signal is strictly positive
    4. Re-normalize signal such that it sums to total

    Parameters
    ----------
    input : ndarray
        Input array
    sigma : scalar
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single
        number, in which case it is equal for all axes.
    epsilon : scalar
        Offset to ensure that signal is strictly positive.
    total : scalar
        Value of the signal summation.

    Returns
    -------
    pdf : ndarray
        Returned array of same shape as input
    """
    input = check_array(input, dtype=['float32','float64'])

    if sigma < 0:
        raise ValueError('sigma must be >= 0.')

    if epsilon <= 0:
        raise ValueError('epsilon must be > 0.')

    if total <= 0:
        raise ValueError('total must be > 0.')

    pdf = gaussian_filter(input, sigma)
    pdf /= pdf.sum()
    pdf += epsilon
    pdf /= pdf.sum()
    pdf *= total
    return pdf


def match_shape2d(a, b):
    """
    Crop array B such that it matches the shape of A.

    Parameters
    ----------
    a : 2d array
        Array of desired size.
    b : 2d array
        Array to crop. Shape must be larger than (or equal to) the shape
        of array a.

    Returns
    -------
    b_crop : 2d array
        Cropped version of b, with the same shape as a.
    """
    a = check_array(a, ndim=2, force_all_finite=False)
    b = check_array(b, ndim=2, force_all_finite=False)

    # Difference in dimensions
    h, w = a.shape
    ylo = (b.shape[0] - h) // 2
    xlo = (b.shape[1] - w) // 2
    if ylo < 0 or xlo < 0:
        raise ValueError("A is bigger than B: "
                         "{!s} vs {!s}".format(a.shape, b.shape))

    return b[ylo:ylo+h,xlo:xlo+w]


def interp2d(img, f, order=1, fill_value=0.):
    """
    Interpolation for 2D gridded data.

    Parameters
    ----------
    img : 2d array, shape (height, width)
        Image to interpolate. This function assumes a grid of sample points:
        x = 0:(width-1), y = 0:(height-1).
    f : 3d array, shape (2, height, width)
        Coordinates of interpolated points. First dimension f[0] corresponds to
        y-coordinates, second dimension f[1] is x-coordinates.
    order : int (default=1)
        Order of the spline interpolation. Must be in the range 0-5.
    fill_value : float (default=0.)
        Value used for points outside the boundaries.

    Returns
    -------
    out, 2d array, shape (height, width)
        Image interpolated at points defined by f.
    """
    # Check inputs
    img = check_array(img, ndim=2, force_all_finite=True)
    f = check_array(f, ndim=3, force_all_finite=True)
    assert_equal_shape(img, f[0])

    # Interpolate
    h, w = img.shape
    out = map_coordinates(img, f.reshape((2,h*w)), order=order, cval=fill_value)

    return out.reshape((h,w))


def griddata2d(img, f, order=1, fill_value=0.):
    """
    Interpolate 2d scattered data

    Parameters
    ----------
    img : 2d array, shape (height, width)
        Image to interpolate.
    f : 3d array, shape (2, height, width)
        Coordinates of at which img is defined. First dimension f[0] corresponds
        to y-coordinates, second dimension f[1] is x-coordinates.
    order : int (default=1)
        Order of the interpolation. Must be in the range 0-2.
    fill_value : float (default=0.)
        Value used for points outside the boundaries.

    Returns
    -------
    out, 2d array, shape (height, width)
        Image interpolated on to regular gird defined by: x = 0:(width-1),
        y = 0:(height-1).
    """
    # Check inputs
    img = check_array(img, ndim=2, force_all_finite=True)
    f = check_array(f, ndim=3, force_all_finite=True)
    assert_equal_shape(img, f[0])

    # Determine interpolation type
    method = 'linear'
    if order == 0:
        method = 'nearest'
    elif order == 2:
        method = 'cubic'

    # Create grid of points
    h, w = img.shape
    xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

    # Interpolate image on to grid
    out = griddata(f.reshape((2,-1)).T, img.ravel(),
                   np.c_[yv.ravel(), xv.ravel()], method=method,
                   fill_value=fill_value)

    return out.reshape((h,w))
