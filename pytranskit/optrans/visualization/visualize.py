import numpy as np
import matplotlib.pyplot as plt
import warnings

from mpl_toolkits.mplot3d import Axes3D

from ..utils import check_decomposition, check_array
from ..decomposition import get_mode_variation, get_mode_histogram
from ..continuous import BaseTransform


def plot_mode():
    return

def get_mode_image(pipeline, component=0, shape=None, transform=None,
                   img0=None, n_std=3., n_steps=5, padding=2):
    """
    Get the mode of variation along a given direction through the data (in image
    space) as a single, big image.

    This function combines the output images from
    optrans.decomposition.get_mode_variation() into one single image.

    Parameters
    ----------
    pipeline : list of objects
        The data processing pipeline. For example, if the data underwent PCA
        then PLDA, pipeline=[pca, plda], where pca and plda are trained
        decomposition objects.
    component : int (default=0.)
        Index of the desired component.
    shape : array_like or None
        Shape of the 2D input data, in the format (height, width). Note: this
        is the shape of the data that were input into the first element in the
        pipeline. For example, if the original inputs were Radon-CDTs, then
        shape might be (100,180). If None, the shape is estimated based on the
        number of data features.
    transform : object or None
        Optimal transport transform object. This allows data to be transformed
        back from transport space to image space. If None, no inverse transform
        is computed.
    img0 : array, shape (height, width) or None
        Reference image used in the transform. If None, no inverse transform
        is computed.
    n_std : float (default=3.)
        Range of standard deviations along the direction. For example if
        n_std=3, the returned data will correspond to the range -1.5*std to
        +1.5*std.
    n_steps : int (default=5)
        Number of steps along the direction.
    padding : int (default=2)
        Padding (in pixels) between images in the mode of variation.

    Returns
    -------
    img : array, shape (height, n_steps*(width+padding)+padding)
        Reconstructed data along desired direction as a single, big image.
    """

    # Check that the pipeline comprises decomposition objects
    for p in pipeline:
        _, _, _ = check_decomposition(p)

    # Check that transform is a BaseTransform
    if (transform is not None) and (not isinstance(transform, BaseTransform)):
        raise ValueError("Transform must be a optrans.continuous transform "
                         "or None.")

    # If a transform is supplied, then we must have an img0
    if (transform is not None) and (img0 is None):
        raise ValueError("Transform requires a reference image img0.")

    # Warn the user if img0 exists but there's no transform
    if (transform is None) and (img0 is not None):
        warnings.warn("Reference image img0 provided without a transform. This "
                      "may cause unexpected reshaping errors.", UserWarning)

    # Get the mode of variation from the final decomposition
    mode = list(get_mode_variation(pipeline[-1], component=component,
                n_steps=n_steps))

    # Perform inverse transforms to get back to original data space
    for p in pipeline[-2::-1]:
        for i in range(n_steps):
            mode[i] = p.inverse_transform(mode[i])

    # Get shape of reconstructed images
    if img0 is not None:
        h, w = img0.shape
    elif shape is not None:
        h, w = shape
    else:
        h, w = _image_shape(mode[0].size)

    # Initialise big image of the mode
    img = np.zeros((h+2*padding,padding+(w+padding)*n_steps))

    # Add each mode image to the big overall image
    for i,m in enumerate(mode):
        # Reshape back to 2d
        tmp = m.reshape(shape)

        # Apply inverse transform (e.g. RadonCDT)
        if transform is not None:
            tmp = transform.apply_inverse_map(tmp, img0)

        # Add reconstructed image to big image
        y = padding
        x = padding + i * (w + padding)
        img[y:y+h,x:x+w] = tmp

    return img


def get_extent(shape, n_std=3., n_steps=5, padding=2):
    """
    Get the location, in units of standard deviation, of the lower-left and
    upper-right corners of a mode image.

    This is useful for setting the x-axis when plotting the output of
    optrans.visualization.get_mode_image().

    Parameters
    ----------
    shape : array_like
        Mode image shape, in the format (heigth, width).
    n_std : float (default=3.)
        Range of standard deviations along the direction. For example if
        n_std=3, the returned data will correspond to the range -1.5*std to
        +1.5*std.
    n_steps : int (default=5)
        Number of steps along the direction.
    padding : int (default=2)
        Padding (in pixels) between images in the mode of variation.

    Returns
    -------
    (left, right, bottom, top) : float
        Location, in units of standard deviation, of the lower-left and
        upper-right corners of the mode image.
    """
    rng = np.linspace(-0.5*n_std, 0.5*n_std, n_steps)

    # Width of a single image in the big mode image
    w = (shape[1] - (n_steps+1)*padding) / n_steps

    # Get two x-values in pixel coordinates and std dev. coordinates
    x1_px = padding + 0.5 * w
    x2_px = 0.5 * shape[1]
    x1_std = -0.5 * n_std
    x2_std = 0

    # Get linear relationship between pixel and std dev. space
    m = (x2_std - x1_std) / (x2_px - x1_px)
    c = x2_std - m * x2_px

    # Convert the pixel coordinates into std dev. coordinates
    left = m * 0 + c
    right = m * (shape[1]-1) + c
    bottom = m * 0 + c
    top = m * (shape[0]-1) + c

    return (left, right, bottom, top)


def _image_shape(n_features):
    """
    Estimate the image shape based on the number of data features. This tries
    to make the image close to square.

    Parameters
    ----------
    n_features : int
        Number of data features.

    Returns
    -------
    (height, width) : int
        Estimated image shape.
    """
    if (np.floor(np.sqrt(n_features))**2 == n_features):
        # If n_features is a square number, height == width
        height = np.sqrt(n_features)
        width = height
    else:
        # Try to make image as square as possible
        height = np.ceil(np.sqrt(n_features))
        while (n_features % height != 0 and height < 1.2*np.sqrt(n_features)):
            height += 1
        width = np.ceil(n_features/height)

    return (int(height), int(width))


def plot_mode_histogram(X, y=None, component=0, decomp=None, n_std=3.,
                        n_bins=10, **kwargs):
    """
    Plot the histogram of data projected on to a particular direction.

    Parameters
    ----------
    X : array, shape (n_samples, n_components)
        Data after decomposition (e.g. PCA, PLDA).
    y : array, shape (n_samples,) or None
        Class labels. If None, all data are assumed to belong to the same class.
    component : int (default=0.)
        Index of the desired component.
    decomp : object
        A trained PCA or PLDA object. This is used to compute the standard
        deviation of the training data projected on to the desired component.
        If None, the standard deviation is estimated using X.
    n_std : float (default=3.)
        Range of standard deviations along the direction. For example if
        n_std=3, the returned data will correspond to the range -1.5*std to
        +1.5*std.
    n_bins : int (default=10)
        Number of histogram bins.

    Returns
    -------
    ax : matplotlib.axes.Axes object
        Axes object.
    """
    X = check_array(X, ndim=2)

    # Default to "all data from class 0"
    if y is None:
        y = np.zeros(X.shape[0])

    # Check array dimensions match
    y = check_array(y, ndim=1)
    if y.size != X.shape[0]:
        raise ValueError("Number of samples in X and y does not match.")

    # Class labels
    labels = np.unique(y)

    if decomp is None:
        # If decomp is NOT provided, std is the std dev. of the input data
        std = X[:,component].std()
    else:
        # If decomp is provided, std is the std dev. of the decomp train
        _, _, std, = check_decomposition(decomp)
        std = std[component]

    # Histogram range (in units of std dev.)
    rng = np.array([-n_std/2, n_std/2])

    # Get histogram of data projected on to component
    hist, bin_centers = get_mode_histogram(X/std, y, component=component,
                                           n_bins=n_bins, rng=rng)

    # Set bar width and positions
    spacing = 0.8 * np.abs(bin_centers[1] - bin_centers[0])
    pos = np.linspace(-spacing/2, spacing/2, labels.size+1)
    width = 0.9 * np.abs(pos[1] - pos[0])
    pos = pos[:-1] + np.diff(pos)/2

    # Plot histogram(s)
    fig, ax = plt.subplots(1, 1, **kwargs)
    if isinstance(hist, list):
        for p,h in zip(pos,hist):
            ax.bar(bin_centers+p, h, width=width)
    else:
        ax.bar(bin_centers, hist, width=width)

    # Axis labels
    ax.set_xlabel("$\sigma$")
    ax.set_ylabel("Proportion of data")
    ax.set_title("Data projected on to component {}".format(component))

    return ax


def plot_mode_image(pipeline, component=0, shape=None, transform=None,
                    img0=None, n_std=3., n_steps=5, padding=2, cmap=None,
                    **kwargs):
    """
    Plot the mode of variation along a given direction through the data (in
    image space).

    This function plots the output of optrans.visualization.get_mode_image().

    Parameters
    ----------
    pipeline : list of objects
        The data processing pipeline. For example, if the data underwent PCA
        then PLDA, pipeline=[pca, plda], where pca and plda are trained
        decomposition objects.
    component : int (default=0.)
        Index of the desired component.
    shape : array_like or None
        Shape of the 2D input data, in the format (height, width). Note: this
        is the shape of the data that were input into the first element in the
        pipeline. For example, if the original inputs were Radon-CDTs, then
        shape might be (100,180). If None, the shape is estimated based on the
        number of data features.
    transform : object or None
        Optimal transport transform object. This allows data to be transformed
        back from transport space to image space. If None, no inverse transform
        is computed.
    img0 : array, shape (height, width) or None
        Reference image used in the transform. If None, no inverse transform
        is computed.
    n_std : float (default=3.)
        Range of standard deviations along the direction. For example if
        n_std=3, the returned data will correspond to the range -1.5*std to
        +1.5*std.
    n_steps : int (default=5)
        Number of steps along the direction.
    padding : int (default=2)
        Padding (in pixels) between images in the mode of variation.
    cmap : matplotlib.Colormap or None
        Image colormap.

    Returns
    -------
    ax : matplotlib.axes.Axes object
        Axes object.
    """

    # Get the mode of variation as a single, big image
    img = get_mode_image(pipeline, component=component, shape=shape,
                         transform=transform, img0=img0, n_std=n_std,
                         n_steps=n_steps, padding=padding)


    # Convert pixel coordinates to std dev. units
    extent = get_extent(img.shape, n_std=n_std, n_steps=n_steps,
                        padding=padding)

    # Plot the mode of variation
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(img, cmap=cmap, extent=extent)
    ax.set_xlabel("$\sigma$")
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("Mode of variation along component {}".format(component))

    return ax


def plot_mode_histogram_image():
    return


def plot_displacements2d(disp, ax=None, scale=1., count=50, lw=1, c='k'):
    """
    Plot 2D pixel displacements as a wireframe grid.

    Parameters
    ----------
    disp : array, shape (2, height, width)
        Pixel displacements. First index denotes direction: disp[0] is
        y-displacements, and disp[1] is x-displacements.
    ax : matplotlib.axes.Axes object or None (default=None)
        Axes in which to plot the wireframe. If None, a new figure is created.
    scale : float (default=1.)
        Exaggeration scale applied to the displacements before visualization.
    count : int (default=50)
        Use at most this many rows and columns in the wireframe.

    Returns
    -------
    ax : matplotlib.axes.Axes object
        Axes object.
    """
    # Note: could use matplotlib.plot_wireframe(), but that uses 3d axes which
    # causes complications when plotting displace alongside other subplots.

    # Check input
    disp = check_array(disp, ndim=3)

    # If necessary, create new figure
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Create regular grid points
    h, w = disp.shape[1:]
    yv = np.arange(h)
    xv = np.arange(w)

    # Create lines in y-direction
    for i in np.linspace(0, w-1, count):
        # x- and y-coordinates of the line
        ind = int(np.floor(i))
        x = i*np.ones(w) + scale*disp[1,:,ind]
        y = yv + scale*disp[0,:,ind]

        # If i is not an integer index, linearly interpolate displacements
        t = i - ind
        if t > 0:
            x += scale * t * (disp[1,:,ind+1]-disp[1,:,ind])
            y += scale * t * (disp[0,::-1,ind+1]-disp[0,::-1,ind])
        ax.plot(x, y, c=c, lw=lw)

    # Create lines in x-direction
    for i in np.linspace(0, h-1, count):
        # x- and y-coordinates of the line
        ind = int(np.floor(i))
        x = xv + scale*disp[1,ind,:]
        y = i*np.ones(h) + scale*disp[0,ind,:]

        # If i is not an integer index, linearly interpolate displacements
        t = i - ind
        if t > 0:
            x += scale * t * (disp[1,ind+1,:]-disp[1,ind,:])
            y += scale * t * (disp[0,ind+1,::-1]-disp[0,ind,::-1])
        ax.plot(x, y, c=c, lw=lw)

    # Format axes
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
