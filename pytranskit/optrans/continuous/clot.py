import numpy as np
from scipy.fftpack import dct, idct

from .cdt import CDT
from ..utils import check_array, assert_equal_shape, signal_to_pdf, interp2d, griddata2d


class CLOT():
    """
    Continuous Linear Optimal Transport Transform.

    This uses Nesterov's accelerated gradient descent to remove the curl in the
    initial mapping.

    Parameters
    ----------
    lr : float (default=0.01)
        Learning rate.
    momentum : float (default=0.)
        Nesterov accelerated gradient descent momentum.
    decay : float (default=0.)
        Learning rate decay over each update.
    max_iter : int (default=300)
        Maximum number of iterations.
    tol : float (default=0.001)
        Stop iterating when change in cost function is below this threshold.
    verbose : int (default=1)
        Verbosity during optimization. 0=no output, 1=print cost,
        2=print all metrics.

    Attributes
    -----------
    displacements_ : array, shape (2, height, width)
        Displacements u. First index denotes direction: displacements_[0] is
        y-displacements, and displacements_[1] is x-displacements.
    transport_map_ : array, shape (2, height, width)
        Transport map f. First index denotes direction: transport_map_[0] is
        y-map, and transport_map_[1] is x-map.
    displacements_initial_ : array, shape (2, height, width)
        Initial displacements computed using the method by Haker et al.
    transport_map_initial_ : array, shape (2, height, width)
        Initial transport map computed using the method by Haker et al.
    cost_ : list of float
        Value of cost function at each iteration.
    curl_ : list of float
        Curl at each iteration.

    References
    ----------
    [A continuous linear optimal transport approach for pattern analysis in
    image datasets]
    (https://www.sciencedirect.com/science/article/pii/S0031320315003507)
    [Optimal mass transport for registration and warping]
    (https://link.springer.com/article/10.1023/B:VISI.0000036836.66311.97)
    """
    def __init__(self, lr=0.01, momentum=0., decay=0., max_iter=300, tol=0.001,
                 verbose=0):
        #super(CLOT, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        return

    def forward(self, sig0, sig1):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        lot : array, shape (2, height, width)
            LOT transform of input image sig1. First index denotes direction:
            lot[0] is y-LOT, and lot[1] is x-LOT.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Create regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # Compute initial mapping
        f = self._get_initial_map(sig0, sig1)
        self.transport_map_initial_ = np.copy(f)
        self.displacements_initial_ = f - np.stack((yv,xv))

        # Initialise evaluation measures
        self.cost_ = []
        self.curl_ = []

        # Initialise derivative of cost function wrt f
        ft = np.zeros_like(f)

        # Initialise previous update (for Nesterov momentum)
        update_prev = np.zeros_like(f)

        for i in range(self.max_iter):
            # Save previous version of f before update
            f_prev = np.copy(f)

            # Nesterov momentum "look ahead"
            f -= self.momentum * update_prev

            # Jacobian and its determinant
            f0y, f0x = np.gradient(f[0])
            f1y, f1x = np.gradient(f[1])
            detJ = (f1x * f0y) - (f1y * f0x)

            # Update evaluation measures
            cost = np.sum(((yv-f[0])**2 + (xv-f[1])**2) * sig0)
            self.cost_.append(cost)
            curl = 0.5 * (f0x - f1y)
            self.curl_.append(0.5*np.sum(curl**2))

            # Print cost value
            if self.verbose:
                print('Iteration {:>4} -- '
                      'cost = {:.4e}'.format(i, self.cost_[-1]))

            # Print curl
            if self.verbose > 1:
                print('... curl = {:.4e}'.format(self.curl_[-1]))

            # Divergence
            vx = np.gradient(-f[0]+yv, axis=1)
            uy = np.gradient(f[1]-xv, axis=0)
            div = vx + uy

            # Poisson solver
            div_dct = dct(dct(div,axis=0,norm='ortho'), axis=1, norm='ortho')
            denom = (2*np.cos(np.pi*xv/w)-2) + (2*np.cos(np.pi*yv/h)-2)
            denom[0,0] = 1.
            div_dct /= denom
            lneg = -idct(idct(div_dct,axis=1,norm='ortho'), axis=0,norm='ortho')
            lnegy, lnegx = np.gradient(lneg)

            # Derivative of cost function wrt f
            ft[0] = (-f0x*lnegy + f0y*lnegx) / sig0
            ft[1] = (-f1x*lnegy + f1y*lnegx) / sig0

            # Update transport map
            self.lr *= (1. / (1. + self.decay*i))
            update = self.momentum * update_prev + self.lr * ft
            update_prev = np.copy(update)
            f -= update

            # If change in cost is below threshold, stop iterating
            if i > 7 and \
                (self.cost_[i-7]-self.cost_[i])/self.cost_[0] < self.tol:
                break

        # Print final evaluation metrics
        if self.verbose:
            print('FINAL METRICS:')
            print('-- cost = {:.4e}'.format(self.cost_[-1]))
            print('-- curl = {:.4e}'.format(self.curl_[-1]))

        # Set final transport map, displacements, and LOT transform
        # Note: Use previous version of f, just in case something weird
        # happened in the final iteration
        self.transport_map_ = f_prev
        self.displacements_ = f_prev - np.stack((yv,xv))
        lot = self.displacements_ * np.sqrt(sig0)

        self.is_fitted = True

        return lot


    def _get_initial_map(self, sig0, sig1):
        """
        Get initial transport map using the method by Haker et al.

        Parameters
        ----------
        sig0 : array, shape (height, width)
            Reference image.
        sig1 : array, shape (height, width)
            Signal to transform.

        Returns
        -------
        f_init : array, shape (2, height, width)
            Initial transport map. First index denotes direction: f_init[0] is
            y-map, and f_init[1] is x-map.
        """
        # Create regular grid
        h, w = sig0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))

        # Set the fill value for interpolation
        fill_val = min(sig0.min(), sig1.min())

        # Integrate images along y-direction
        sum0 = signal_to_pdf(sig0.sum(axis=0))
        sum1 = signal_to_pdf(sig1.sum(axis=0))

        # Compute mass-preserving mapping between the two integrals
        cdt = CDT()
        
        x0 = np.array(range(len(sum0)))
        x1 = np.array(range(len(sum1)))
        shat,_,_ = cdt.forward(x0, sum0, x1, sum1, rm_edge=False)
        a = np.tile(shat, (h,1))
        
        # _ = cdt.forward(sum0, sum1)
        # a = np.tile(cdt.transport_map_, (h,1))
        aprime = np.gradient(a, axis=1)

        # Compute a'(x)sig1(a(x),y) for all y
        siga = aprime * interp2d(sig1, np.stack((yv,a)), fill_value=fill_val)

        # Compute b(a(x),y) one column at a time
        b = np.zeros_like(a)
        for i in range(w):
            col0 = signal_to_pdf(sig0[:,i])
            cola = signal_to_pdf(siga[:,i])
            
            x0 = np.array(range(len(col0)))
            x1 = np.array(range(len(cola)))
            shat,_,_ = cdt.forward(x0, col0, x1, cola, rm_edge=False)
            b[:,i] = shat
        
            # _ = cdt.forward(col0, cola)
            # b[:,i] = cdt.transport_map_

        # Re-grid b(a(x),y) so that we end up with b(x,y)
        b = griddata2d(b, np.stack((yv,a)))
        zero_column = np.array(np.where(np.mean(b,axis=0)==0.))

        if len(zero_column)>0:
            for z in zero_column[0]:
                if z==0:
                    b[:,z] = b[:,z+1]
                else:
                    b[:,z] = b[:,z-1]

        return np.stack((b,a))
    
    def apply_forward_map(self, transport_map, sig1):
        """
        Apply forward transport map.

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
        Apply inverse transport map.

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
