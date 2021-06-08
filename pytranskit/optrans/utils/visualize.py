import numpy as np
import matplotlib.pyplot as plt
import warnings


from ..utils import check_array


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
        x = i*np.ones(h) + scale*disp[1,:,ind]
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
        y = i*np.ones(w) + scale*disp[0,ind,:]

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
