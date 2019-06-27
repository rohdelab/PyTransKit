import numpy as np

from ..utils import check_array, assert_equal_shape, interp2d, griddata2d


class BaseTransform(object):
    """
    Base class for optimal transport transform methods.

    .. warning::

       This class should **not** be used directly. Use derived classes instead.
    """
    def __init__(self):
        self.is_fitted = False
        self.sig0_ = None
        self.displacements_ = None
        self.transport_map_ = None


    def _check_is_fitted(self):
        if not self.is_fitted:
            raise AssertionError("The forward transform of {0!s} has not been "
                                 "called yet. Call 'forward' before using "
                                 "this method".format(type(self).__name__))


    def forward(self):
        """
        Placeholder for forward transform.
        Subclasses should implement this method!
        """
        raise NotImplementedError


    def inverse(self):
        """
        Inverse transform.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        self._check_is_fitted()
        return self.apply_inverse_map(self.transport_map_, self.sig0_)


    def apply_forward_map(self):
        """
        Placeholder for application of forward transport map.
        Subclasses should implement this method!
        """
        raise NotImplementedError


    def apply_inverse_map(self):
        """
        Placeholder for application of inverse transport map.
        Subclasses should implement this method!
        """
        raise NotImplementedError



class BaseMapper2D(BaseTransform):
    """
    Base class for 2D optimal transport transform methods (e.g. CLOT, VOT2D).

    .. warning::

       This class should **not** be used directly. Use derived classes instead.
    """
    def __init__(self):
        super(BaseMapper2D, self).__init__()
        return


    def apply_forward_map(self, transport_map, sig1):
        """
        Appy forward transport map.

        Parameters
        ----------
        transport_map : array, shape (2, height, width)
            Forward transport map.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        sig0_recon : array, shape (height, width)
            Reconstructed reference signal sig0.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig1, ['transport_map', 'sig1'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Reconstruct sig0
        sig0_recon = detJ * interp2d(sig1, transport_map, fill_value=sig1.min())

        return sig0_recon


    def apply_inverse_map(self, transport_map, sig0):
        """
        Appy inverse transport map.

        Parameters
        ----------
        transport_map : array, shape (2, height, width)
            Forward transport map. Inverse is computed in this function.
        sig0 : array, shape (height, width)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=3,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map[0], sig0, ['transport_map', 'sig0'])

        # Jacobian and its determinant
        f0y, f0x = np.gradient(transport_map[0])
        f1y, f1x = np.gradient(transport_map[1])
        detJ = (f1x * f0y) - (f1y * f0x)

        # Let's hope there are no NaNs/Infs in sig0/detJ
        sig1_recon = griddata2d(sig0/detJ, transport_map, fill_value=sig0.min())

        return sig1_recon
