
import numpy as np
import numpy.linalg as LA
import multiprocessing as mp

from pytranskit.optrans.continuous.radoncdt3D import RadonCDT3D

eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]

class RCDT_NS_3D:
    def __init__(self, num_classes, Npoints=500, use_gpu=False, rm_edge=False):
        """
        Parameters
        ----------
        num_classes : integer, total number of classes
        Npoints : scalar, number of radon projections
        use_gpu : boolean, use GPU if True
        rm_edge : boolean flag; IF TRUE the first and last points of RCDTs will be removed
            default = False
        """
        self.num_classes = num_classes
        self.Npoints = Npoints
        self.rm_edge = rm_edge
        self.subspaces = []
        self.len_subspace = 0
        self.use_gpu = use_gpu
        radoncdt3D = RadonCDT3D(self.Npoints,self.use_gpu)
        self.phis, self.thetas = radoncdt3D.sample_sphere(self.Npoints)

    def fit(self, Xtrain, Ytrain, no_deform_model=False):
        """Fit linear model.
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_rows, n_columns, n_width)
            Image data for training.
        Ytrain : ndarray of shape (n_samples,)
            Labels of the training images.
        no_deform_model : boolean flag; IF TRUE, no deformation model will be added
            default = False.
        """
        
        # calculate the RCDT using parallel CPUs
        print('\nCalculating RCDTs for training images ...')
        Xrcdt = self.rcdt_parallel(Xtrain)
        
        # generate the basis vectors for each class
        print('Generating basis vectors for each class ...')
        for class_idx in range(self.num_classes):
            class_data = Xrcdt[Ytrain == class_idx]
            if no_deform_model:
                flat = class_data.reshape(class_data.shape[0], -1)
            else:
                class_data_trans = self.add_trans_samples(class_data)
                flat = class_data_trans.reshape(class_data_trans.shape[0], -1)
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            self.subspaces.append(basis)


    def predict(self, Xtest, use_gpu=False):
        """Predict using the linear model
        Parameters
        ----------
        Xtest : array-like, shape (n_samples, n_rows, n_columns, n_width)
            Image data for testing.
        use_gpu: boolean flag; IF TRUE, use gpu for calculations
            default = False.
            
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in Xtest.
        """
        
        # calculate the RCDT using parallel CPUs
        print('\nCalculating RCDTs for testing images ...')
        Xrcdt = self.rcdt_parallel(Xtest)
        
        # vectorize RCDT matrix
        X = Xrcdt.reshape([Xrcdt.shape[0], -1])
        
        # import cupy for using GPU
        if use_gpu:
            import cupy as cp
            X = cp.array(X)
        
        # find nearest subspace for each test sample
        print('Finding nearest subspace for each test sample ...')
        D = []
        for class_idx in range(self.num_classes):
            basis = self.subspaces[class_idx]
            basis = basis[:self.len_subspace,:]
            
            if use_gpu:
                D.append(cp.linalg.norm(cp.matmul(cp.matmul(X, cp.array(basis).T), 
                                                  cp.array(basis)) -X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if use_gpu:
            preds = cp.argmin(cp.stack(D, axis=0), axis=0)
            return cp.asnumpy(preds)
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            return preds


    def fun_rcdt_single(self, I):
        # I: (rows, columns, width)
        radoncdt3D = RadonCDT3D(self.Npoints,self.use_gpu)
        template = np.ones(I.shape, dtype=I.dtype)
        Ircdt = radoncdt3D.forward(x0_range, template / np.sum(template), 
                                 x_range, I / np.sum(I), 
                                 self.rm_edge)
        return Ircdt
    
    def fun_rcdt_batch(self, data):
        # data: (n_samples, rows, columns, width)
        dataRCDT = [self.fun_rcdt_single(data[j, :, :, :] + eps) for j in range(data.shape[0])]
        return np.array(dataRCDT)
    
    def rcdt_parallel(self, X):
        # X: (n_samples, rows, columns, width)
        # calc RCDT of images
        n_cpu = np.min([mp.cpu_count(), X.shape[0]])
        splits = np.array_split(X, n_cpu, axis=0)
        pl = mp.Pool(n_cpu)
    
        dataRCDT = pl.map(self.fun_rcdt_batch, splits)
        rcdt_features = np.vstack(dataRCDT)  # (n_samples, proj_len, num_angles)
        pl.close()
        pl.join()

        return rcdt_features
        
    def add_trans_samples(self, rcdt_features):
        # rcdt_features: (n_samples, proj_len, Npoints)
        # deformation vectors for  translation
        #v1, v2 = np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)
        v1, v2, v3 = np.cos(self.thetas*np.pi/180)*np.sin(self.phis*np.pi/180), np.sin(self.thetas*np.pi/180)*np.sin(self.phis*np.pi/180), np.cos(self.phis*np.pi/180)
        v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
        v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
        v3 = np.repeat(v3[np.newaxis], rcdt_features.shape[1], axis=0)
        return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis], v3[np.newaxis]])
        
