# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:13:16 2023

@author: Naqib Sad Pathan
"""
import inspect
import numpy as np
from skimage.transform import radon, iradon
from pytranskit.optrans.continuous.radoncdt import RadonCDT
from .base import BaseTransform
from .cdt import CDT
from .scdt import SCDT
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf, match_shape2d
import matplotlib.pyplot as plt


class RadonSCDT(BaseTransform):
    """
    Radon Signed Cumulative Distribution Transform.

    Parameters
    ----------
    theta : 1d array (default=np.arange(180))
        Radon transform projection angles.

    
    References
    ----------
    
    """
    def __init__(self, theta=np.arange(180)):
        super(RadonSCDT, self).__init__()
        self.theta = check_array(theta, ndim=1)
        self.epsilon = 1e-8
        self.total = 1.


    def forward(self, x0_range, sig0, x1_range, sig1, rm_edge=True):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : Reference image.
        sig1 : Signal to transform.

        Returns
        -------
        rscdt   : Radon-SCDT of input image sig1.
        ref     : reference signal used during SCDT 
        mpos_all: weight of positive part of all the projections
        mneg_all: weight of positive part of all the projections
        rad1    : Radon transform of sig1
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=False)
        sig1 = check_array(sig1, ndim=2, dtype=[np.float64, np.float32],
                           force_strictly_positive=False)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Set reference signal
        self.sig0_ = sig0

        # Radon transform of signals
       
        rad1 = radon(sig1, theta=self.theta, circle=False)
        if len(np.unique(sig0)) == 1:
            rad0 = np.ones(rad1.shape)
        else:
            rad0 = radon(sig0, theta=self.theta, circle=False)

        # Initalize CDT, Radon-CDT
        rscdt = []
        mpos_all=[]
        mneg_all=[]
        ref=[];
       
        
        # Loop over angles
        for i in range(self.theta.size):
            rad0[0,i] = 0
            
            j0=rad0[:,i];
            j1=rad1[:,i];
            
            x0 = np.linspace(x0_range[0], x0_range[1], len(j0))
            x1 = np.linspace(x1_range[0], x1_range[1], len(j1))
            scdt=[]
            s0 = np.ones(j1.shape)
            s0 = s0/s0.sum()
            scdt = SCDT(s0) 
            # Compute CDT of this projection
            lot,mpos,mneg= scdt.calc_scdt(j1, x1, j0, x0)

            # Update 2D Radon-CDT
            rscdt.append(lot)
            ss=1
            mpos_all.append(mpos*ss)
            mneg_all.append(mneg*ss)
            ref.append(j0)

        # Convert lists to arrays
        rscdt = np.asarray(rscdt).T
        ref=np.asarray(ref).T

        self.is_fitted = True

        return rscdt,ref,mpos_all,mneg_all,rad1

    def inverse(self, rscdt,ref,mpos_all,mneg_all, template, x_range):
        """
        Inverse transform.
        
        Returns
        -------
        Irec : Reconstructed signal sig1.
        rdn  : Reconstruced signal in radon space
        """
        rdn=[]
        b=rscdt.shape
        b1=(b[0]/2)
        b1=np.array(b1).astype(int)
        for i in range(self.theta.size):
            a=ref[:,i]
            s0 = np.ones(a.shape)
            s0 = s0/s0.sum()
            scdt = SCDT(s0) 
            irec=scdt.istransform(rscdt[:b1,i],rscdt[b1:,i],mpos_all[i],mneg_all[i])


            rdn.append(irec)
        rdn=np.asarray(rdn).T
        Irec=iradon(rdn, self.theta, circle=False, filter_name='ramp')

        return Irec,rdn

        


