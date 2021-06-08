import numpy as np

from numpy import sin, cos

from scipy.fftpack import fft,fftshift,ifft

from .base import BaseTransform
from .cdt import CDT
from ..utils import check_array, assert_equal_shape
from ..utils import signal_to_pdf




class RadonCDT3D(BaseTransform):
    """
    3D Radon Cumulative Distribution Transform.

    Parameters
    ----------
    Npoints: scaler, number of radon projections
    use_gpu: boolean, use GPU if True
    """
    def __init__(self, Npoints=1024, use_gpu=False):
        super(RadonCDT3D, self).__init__()
        self.Npoints = Npoints
        self.epsilon = 1e-8
        self.total = 1.
        self.use_gpu = use_gpu
        if self.use_gpu:
            import cupy as cp
            from cupyx.scipy import ndimage as nd
            self.cp = cp
        else:
            from scipy import ndimage as nd
        self.nd = nd
            
        
    def sample_sphere(self,num_pts,return_type='spherical'):
        '''
        This function "uniformly" samples a sphere on num_pts
        Inputs: 
            num_pts= number of points to sample
            return_type = return points in 'spherical' or 'cartesian' coordinates
        '''
    
        indices = np.arange(0, num_pts, dtype=float) + 0.5
    
        phi = np.arccos(1 - 2*indices/num_pts)
        theta = (np.pi * (1 + 5**0.5) * indices)%(2*np.pi)
        
        if return_type=='spherical':
            return phi*180.0/np.pi,theta*180.0/np.pi
        elif return_type=='cartesian':
            x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
            return x,y,z
        else:
            return 
    
    def get_rotation_matrix(self,phi,theta,forward=True):
        '''
        Calculates the Rotation matrix for,
        Input:
            phi: Rotation along the x-axis (1,0,0) in degrees
            theta: Rotation along the y-axis (0,1,0) in degrees
        output:
            M: The rotation matrix 
        '''
        phi=phi*np.pi/180.
        theta=theta*np.pi/180.
        if forward:
            Mx=np.array([[1.,0.,0.],[0.,cos(phi),-sin(phi)],[0.,sin(phi),cos(phi)]])
            My=np.array([[cos(theta),0.,sin(theta)],[0.,1.,0.],[-sin(theta),0.,cos(theta)]])    
            M=np.matmul(My,Mx)
        else:
            Mx=np.array([[1.,0.,0.],[0.,cos(phi),sin(phi)],[0.,-sin(phi),cos(phi)]])
            My=np.array([[cos(theta),0.,-sin(theta)],[0.,1.,0.],[sin(theta),0.,cos(theta)]])    
            M=np.matmul(Mx,My)
        if self.use_gpu:
            return self.cp.array(M)
        else:
            return M
    
    def rotate3D(self,img3D,phi,theta,forward=True):
        '''
        Rotates an image around the x and y axis with,
        Inputs: 
            img3D: Input image numpy 3D
            phi: Angle rotation around x
            theta: Angle rotation around y
        return:
            img3D_rot: Rotated image (numpy 3D)
        '''
        M=self.get_rotation_matrix(phi,theta,forward)
        if self.use_gpu:
            center=0.5*self.cp.array(img3D.shape)
            offset=(center-center.dot(M)).dot(self.cp.linalg.inv(M))    
            img3D_rot=self.cp.asnumpy(self.nd.affine_transform(self.cp.array(img3D),M,offset=-offset,order=1))
        else:
            center=0.5*np.array(img3D.shape)
            offset=(center-center.dot(M)).dot(np.linalg.inv(M))    
            img3D_rot=self.nd.affine_transform(img3D,M,offset=-offset,order=1)
        return img3D_rot
    
    def dRamp(self,x,h,d=3):
        rampFilter=fftshift(np.linspace(-h/2.,h/2.,h)**(d-1))
        fx=fft(x)
        fx*=rampFilter
        xhat=ifft(fx).real
        return xhat
    
    def radon_3D(self,img3D,N):
        phis,thetas=self.sample_sphere(N)
        self.phis = phis
        self.thetas = thetas
        d=np.array(img3D.shape).max()
        img3Dhat=np.zeros((d,N))
        for n,(phi,theta) in enumerate(zip(phis,thetas)):
            img3D_rot=self.rotate3D(img3D,phi,theta)
            img3Dhat[:,n]=img3D_rot.mean(0).mean(0)        
        return img3Dhat
    
    
    def iradon_3D(self,img3Dhat, phis, thetas):
        h,N=img3Dhat.shape
        img3D_recon=np.zeros((h,h,h))
        #phis,thetas=sample_sphere(N)
        for n in (range(N)):
            temp=np.repeat(self.dRamp(img3Dhat[:,n],h)[np.newaxis,:],(h),axis=0)
            temp=np.repeat(temp[np.newaxis,:],(h),axis=0)
            img3D_recon+=self.rotate3D(temp,phis[n],thetas[n],forward=False)/float(N)
        img3D_recon[img3D_recon<0]=0
        return img3D_recon
    
    
    def forward(self, x0_range, sig0, x1_range, sig1, rm_edge=False):
        """
        Forward transform.

        Parameters
        ----------
        sig0 : array, shape (height, width, depth)
            Reference image.
        sig1 : array, shape (height, width, depth)
            Signal to transform.

        Returns
        -------
        rcdt : array, shape (t, N)
            3D Radon-CDT of input image sig1.
        """
        # Check input arrays
        sig0 = check_array(sig0, ndim=3, dtype=[np.float64, np.float32],
                           force_strictly_positive=False)
        sig1 = check_array(sig1, ndim=3, dtype=[np.float64, np.float32],
                           force_strictly_positive=False)

        # Input signals must be the same size
        assert_equal_shape(sig0, sig1, ['sig0', 'sig1'])

        # Set reference signal
        self.sig0_ = sig0

        # 3D Radon transform of signals
        rad1 = self.radon_3D(sig1,self.Npoints)
        if len(np.unique(sig0)) == 1:
            rad0 = np.ones(rad1.shape)
        else:
            rad0 = self.radon_3D(sig0,self.Npoints)

        # Initalize CDT, Radon-CDT, displacements, and transport map
        cdt = CDT()
        rcdt = []
        u = []
        f = []

        # Loop over angles
        for i in range(self.Npoints):
            # Convert projection to PDF
            j0 = signal_to_pdf(rad0[:,i], epsilon=self.epsilon,
                               total=self.total)
            j1 = signal_to_pdf(rad1[:,i], epsilon=self.epsilon,
                               total=self.total)
            x0 = np.linspace(x0_range[0], x0_range[1], len(j0))
            x1 = np.linspace(x1_range[0], x1_range[1], len(j1))

            # Compute CDT of this projection
            lot, _,_ = cdt.forward(x0, j0, x1, j1, rm_edge)

            # Update 2D Radon-CDT, displacements, and transport map
            rcdt.append(lot)
            u.append(cdt.displacements_)
            f.append(cdt.transport_map_)

        # Convert lists to arrays
        rcdt = np.asarray(rcdt).T
        self.displacements_ = np.asarray(u).T
        self.transport_map_ = np.asarray(f).T

        self.is_fitted = True

        return rcdt
    
    def inverse(self, transport_map, sig0, x1):
        """
        Inverse transform.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        self._check_is_fitted()
        return self.apply_inverse_map(transport_map, sig0, x1)
    
    def apply_inverse_map(self, transport_map, sig0, x_range):
        """
        Appy inverse transport map.

        Parameters
        ----------
        transport_map : 2d array, shape (t, N)
            Forward transport map. Inverse is computed in this function.
        sig0 : array, shape (height, width, depth)
            Reference signal.

        Returns
        -------
        sig1_recon : array, shape (height, width)
            Reconstructed signal sig1.
        """
        # Check input arrays
        transport_map = check_array(transport_map, ndim=2,
                                    dtype=[np.float64, np.float32])
        sig0 = check_array(sig0, ndim=3, dtype=[np.float64, np.float32],
                           force_strictly_positive=True)

        # Initialize Radon transforms
        if len(np.unique(sig0)) == 1:
            rad0 = np.ones(transport_map.shape)
        else:
            rad0 = self.radon_3D(sig0,self.Npoints)
        rad1 = np.zeros_like(rad0)

        # Check transport map and Radon transforms are the same size
        assert_equal_shape(transport_map, rad0,
                           ['transport_map', 'Radon transform of sig0'])

        # Loop over angles
        cdt = CDT()
        for i in range(self.Npoints):
            # Convert projection to PDF
            j0 = signal_to_pdf(rad0[:,i], epsilon=1e-8, total=1.)
            
            x = np.linspace(x_range[0], x_range[1], len(j0))

            # Radon transform of sig1 comprised of inverse CDT of projections
            rad1[:,i] = cdt.apply_inverse_map(transport_map[:,i], j0, x)

        # Inverse Radon transform
        sig1_recon = self.iradon_3D(rad1, self.phis, self.thetas)

        # Crop sig1_recon to match sig0
        #sig1_recon = match_shape2d(sig0, sig1_recon)

        return sig1_recon
    
    
    
