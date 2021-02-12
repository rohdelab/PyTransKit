import numpy as np
from scipy import interp
import matplotlib.pyplot as plt


from pytranskit.optrans.continuous.base import BaseTransform
from pytranskit.optrans.utils import check_array, assert_equal_shape, signal_to_pdf


class CDT(BaseTransform):
    """
    Cumulative Distribution Transform.

    Attributes
    -----------
    displacements_ : 1d array
        Displacements u.
    transport_map_ : 1d array
        Transport map f.

    References
    ----------
    [The cumulative distribution transform and linear pattern classification]
    (https://arxiv.org/abs/1507.05936)
    """
    def __init__(self):
        super(CDT, self).__init__()


    def forward(self, x0, sig0, x1, sig1, rm_edge=False):
        """
        Forward transform.

        Parameters
        ----------
        x0 : 1d array
            Independent axis variable of reference signal (sig0).
        sig0 : 1d array
            Reference signal.
        x1 : 1d array
            Independent axis variable of the signal to transform (sig1).
        sig1 : 1d array
            Signal to transform.

        Returns
        -------
        sig1_cdt : 1d array
            CDT of input signal sig1 (new definition).
        sig1_hat : 1d array
            old definition.
        xilde : 1d array
            Independent axis variable in CDT space.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=1, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=1, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        #assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        self.sig0_ = sig0

        # Cumulative sums
        cum0 = np.cumsum(sig0)
        cum1 = np.cumsum(sig1)

        # x co-ordinates and interpolated y co-ordinates
        x = x1
        y = np.linspace(0, 1, sig0.size)
        y0 = interp(y, cum0, x0)    # inverse of CDF of sig0
        if len(np.unique(sig0)) == 1:
            y1 = interp(y, cum1, x)    # inverse of CDF of sig1 for uniform reference
        else:
            y1 = interp(cum0, cum1, x)    # inverse of CDF of sig1        

        # Compute displacements: u = f(x0)-x0
        self.displacements_ = interp(x0, y0, y1-y0)
        #self.displacements_ = y1 - x0

        # Compute transport map: f = u - x0
        #self.transport_map_ = self.displacements_ - x0
        self.transport_map_ = y1
        
        # CDT (new definition)
        sig1_cdt = self.transport_map_
        self.xtilde = x0


        # OLD CDT = (f - x) * sqrt(I0)
        sig1_hat = self.displacements_ * np.sqrt(sig0)
        
        if rm_edge:
            sig1_cdt = np.delete(sig1_cdt, 0)
            sig1_cdt = np.delete(sig1_cdt, len(sig1_cdt)-1)
            
            sig1_hat = np.delete(sig1_hat, 0)
            sig1_hat = np.delete(sig1_hat, len(sig1_hat)-1)
            
            y1 = np.delete(y1, 0)
            y1 = np.delete(y1, len(y1)-1)
            
            self.xtilde = np.delete(self.xtilde, 0)
            self.xtilde = np.delete(self.xtilde, len(self.xtilde)-1)
            
            self.transport_map_ = np.delete(self.transport_map_, 0)
            self.transport_map_ = np.delete(self.transport_map_, len(self.transport_map_)-1)

        self.is_fitted = True

        return sig1_cdt, sig1_hat, self.xtilde


    def inverse(self, transport_map, sig0, x1):
        """
        Inverse transform.
        
        Parameters
        ----------
        transport_map : 1d array
            Forward transport map.
        sig0 : 1d array
            Reference signal.
        x1 : 1d array
            Independent axis variable of the signal to reconstruct.

        Returns
        -------
        sig1_recon : 1d array
            Reconstructed signal.
        """
        self._check_is_fitted()
        return self.apply_inverse_map(transport_map, sig0, x1)


    def apply_forward_map(self, transport_map, sig1):
        """
        Appy forward transport map.

        Parameters
        ----------
        transport_map : 1d array
            Forward transport map.
        sig1 : 1d array
            Signal to transform.

        Returns
        -------
        sig0_recon : 1d array
            Reconstructed reference signal sig0.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=1,
                                    dtype=[np.float64, np.float32])
        sig1 = check_array(sig1, ndim=1, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map, sig1, ['transport_map', 'sig1'])

        # Reconstruct sig0
        x = np.arange(sig1.size)
        fprime = np.gradient(transport_map)
        sig0_recon = fprime * interp(transport_map, x, sig1)
        return sig0_recon


    def apply_inverse_map(self, transport_map, sig0, x):
        """
        Apply inverse transport map.

        Parameters
        ----------
        transport_map : 1d array
            Forward transport map. Inverse is computed in this function.
        sig0 : 1d array
            Reference signal.

        Returns
        -------
        sig1_recon : 1d array
            Reconstructed signal sig1.
        """
        # Check inputs
        transport_map = check_array(transport_map, ndim=1,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=1, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        assert_equal_shape(transport_map, sig0, ['transport_map', 'sig0'])

        # Reconstruct sig1
        
        fprime = np.gradient(transport_map)
        
        if len(np.unique(sig0)) == 1:
            sig1_recon = interp(x, transport_map, 1/fprime)
        else:
            sig1_recon = interp(x, transport_map, sig0/fprime)
            
        sig1_recon = signal_to_pdf(sig1_recon, epsilon=1e-7,
                               total=1.)

        return sig1_recon




