import six
import warnings
import numpy as np

def assert_all_finite(X):
    """
    Throw a ValueError if X contains NaN or infinity.
    """
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def assert_equal_shape(a, b, names=None):
    """
    Throw a ValueError if a and b are not the same shape.
    """
    if names is None:
        names = ['a', 'b']

    if a.shape != b.shape:
        raise ValueError("{} and {} must be the same shape: "
                         "{!s} vs {!s}".format(names[0], names[1],
                         a.shape, b.shape))


def check_array(array, ndim=None, dtype='numeric', force_all_finite=True,
                force_strictly_positive=False):
    """
    Input validation on an array, list, or similar.

    Parameters
    ----------
    array : object
        Input object to check/convert
    ndim : int or None (default=None)
        Number of dimensions that array should have. If None, the dimensions
        are not checked
    dtype : string, type, list of types or None (default='numeric')
        Data type of result. If None, the dtype of the input is preserved.
        If 'numeric', dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in array
    force_strictly_positive : boolean (default=False)
        Whether to raise an error if any array elements are <= 0

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # Do we want a numeric dtype?
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == 'numeric'

    # Original dtype of the array
    dtype_orig = getattr(array, 'dtype', None)
    if not hasattr(dtype_orig, 'kind'):
        # Not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # If we want a numeric dtype
    if isinstance(dtype, six.string_types) and dtype == 'numeric':
        if dtype_orig is not None and dtype_orig.kind == "O":
            # If input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    # If dtype is a list of possible types
    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # No dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    array = np.array(array, dtype=dtype)

    if ndim is not None and array.ndim != ndim:
        if ndim == 1 and array.ndim == 2 and array.shape[1] == 1:
            array = np.squeeze(array)
            warnings.warn("Converted column vector into 1D array.")
        else:
            raise ValueError("Expected {}D array, "
                             "got {}D array instead".format(ndim, array.ndim))

    # Check for NaNs and infs
    if force_all_finite:
        assert_all_finite(array)

    # Check that array is strictly positive
    if force_strictly_positive and np.any(array <= 0):
        raise ValueError("Array must be strictly positive (i.e. > 0).")

    return array


def check_decomposition(obj):
    """
    Check that an object is a PCA or PLDA (i.e. decomposition) object.

    Parameters
    ----------
    obj : object
        Object to check

    Returns
    -------
    mean : array, shape (n_features,)
        Mean of the data in the decomposition object
    components : array, shape (n_components, n_features)
        Components learned by the decomposition object
    std : array, shape (n_components,)
        Standard deviation of the training data projected on to each component
    """
    # Check that decomp is a PCA or PLDA object
    try:
        mean = obj.mean_
        components = obj.components_
        std = np.sqrt(obj.explained_variance_)
        return mean, components, std
    except AttributeError:
        raise
