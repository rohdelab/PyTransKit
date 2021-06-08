

import numpy as np
import numpy.linalg as LA
import multiprocessing as mp

from pytranskit.optrans.continuous.cdt import CDT
from pytranskit.optrans.utils import signal_to_pdf

x0_range = [0, 1]
x1_range = [0, 1]

class CDT_NS:
    def __init__(self, num_classes, rm_edge=False):
        """
        Parameters
        ----------
        num_classes : integer, total number of classes
        rm_edge : boolean flag; IF TRUE the first and last points of CDTs will be removed
            default = False
        """
        self.num_classes = num_classes
        self.rm_edge = rm_edge
        self.subspaces = []
        self.len_subspace = 0
        self.epsilon = 1e-8
        self.total = 1.

    def fit(self, Xtrain, Ytrain, no_deform_model=False):
        """Fit linear model.
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_columns)
            1D data for training.
        Ytrain : ndarray of shape (n_samples,)
            Labels of the training samples.
        no_deform_model : boolean flag; IF TRUE, no deformation model will be added
            default = False.
        """
        
        # calculate the CDT using parallel CPUs
        print('\nCalculating CDTs for training data ...')
        Xcdt = self.cdt_parallel(Xtrain)
        
        # generate the basis vectors for each class
        print('Generating basis vectors for each class ...')
        for class_idx in range(self.num_classes):
            class_data = Xcdt[Ytrain == class_idx]
            if no_deform_model:
                flat = class_data
            else:
                class_data_trans = self.add_trans_samples(class_data)
                flat = class_data_trans
            
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
        Xtest : array-like, shape (n_samples, n_columns)
            1D data for testing.
        use_gpu: boolean flag; IF TRUE, use gpu for calculations
            default = False.
            
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per sample in Xtest.
        """
        
        # calculate the CDT using parallel CPUs
        print('\nCalculating CDTs for testing samples ...')
        X = self.cdt_parallel(Xtest)
        
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


    def fun_cdt_single(self, sig1):
        # sig1: (0, columns)
        cdt = CDT()
        sig0 = np.ones(sig1.shape, dtype=sig1.dtype)
        j0 = signal_to_pdf(sig0, epsilon=self.epsilon,
                               total=self.total)
        j1 = signal_to_pdf(sig1, epsilon=self.epsilon,
                           total=self.total)

        x0 = np.linspace(x0_range[0], x0_range[1], len(j0))
        x1 = np.linspace(x1_range[0], x1_range[1], len(j1))
        
        shat,_,_ = cdt.forward(x0, j0, x1, j1, self.rm_edge)
        return shat
    
    def fun_cdt_batch(self, data):
        # data: (n_samples, columns)
        dataCDT = [self.fun_cdt_single(data[j, :]) for j in range(data.shape[0])]
        return np.array(dataCDT)
    
    def cdt_parallel(self, X):
        # X: (n_samples, columns)
        # calc CDT of signals
        n_cpu = np.min([mp.cpu_count(), X.shape[0]])
        splits = np.array_split(X, n_cpu, axis=0)
        pl = mp.Pool(n_cpu)
    
        dataCDT = pl.map(self.fun_cdt_batch, splits)
        cdt_features = np.vstack(dataCDT)
        cdt_features = cdt_features.reshape([cdt_features.shape[0], -1])
        pl.close()
        pl.join()

        return cdt_features
        
    def add_trans_samples(self, cdt_features):
        # cdt_features: (n_samples, cdt)
        # deformation vector for  translation
        v1 = np.ones([1, cdt_features.shape[1]])
        return np.concatenate([cdt_features, v1])
