
import numpy as np
import numpy.linalg as LA

from pytranskit.optrans.continuous.scdt import SCDT

class SCDT_NS:
    def __init__(self, num_classes, rm_edge = False):
        """
        Parameters
        ----------
        num_classes : integer, total number of classes
        rm_edge : [optional] boolean flag; IF TRUE the first and last points of CDTs will be removed
            default = False
        """
        self.num_classes = num_classes
        self.rm_edge = rm_edge
        self.subspaces = []
        self.len_subspace = 0

    def fit(self, Xtrain, Ytrain, Ttrain=None, no_deform_model=True):
        """Fit linear model.
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_columns)
            1D data for training.
        Ytrain : ndarray of shape (n_samples,)
            Labels of the training samples.
        Ttrain : [optional] array-like, shape (n_samples, n_columns)
            domain for corresponding training signals.
        no_deform_model : [optional] boolean flag; IF TRUE, no deformation model will be added
            default = False.
        """
        
        # calculate the SCDTs
        print('\nCalculating SCDTs for training data ...')
        #Xcdt = self.cdt_parallel(Xtrain)
        
        N = Xtrain.shape[1]
        t0 = np.linspace(0,1,N) # Domain of the reference
        s0 = np.ones(N)
        s0 = s0/s0.sum()
        
        s_scdt = []
        for i in range(Xtrain.shape[0]):
            if Ttrain is None:
                s_scdt.append(self.calc_scdt(Xtrain[i],t0,s0,t0))
            else:
                s_scdt.append(self.calc_scdt(Xtrain[i],Ttrain[i],s0,t0))
        Xscdt = np.stack(s_scdt)
        
        # generate the basis vectors for each class
        print('Generating basis vectors for each class ...')
        for class_idx in range(self.num_classes):
            class_data = Xscdt[Ytrain == class_idx]
            if no_deform_model:
                flat = class_data
                self.len_subspace = 1
            else:
                class_data_trans = self.add_trans_samples(class_data)
                flat = class_data_trans
                self.len_subspace = 2
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            self.subspaces.append(basis)


    def predict(self, Xtest, Ttest=None, use_gpu=False):
        """Predict using the linear model
        Parameters
        ----------
        Xtest : array-like, shape (n_samples, n_columns)
            1D data for testing.
        Ttest : [optional] array-like, shape (n_samples, n_columns)
            domain for corresponding test signals.
        use_gpu: [optional] boolean flag; IF TRUE, use gpu for calculations
            default = False.
            
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per sample in Xtest.
        """
        
        # calculate the SCDT
        print('\nCalculating SCDTs for testing samples ...')
        #X = self.cdt_parallel(Xtest)
        
        N = Xtest.shape[1]
        t0 = np.linspace(0,1,N) # Domain of the reference
        s0 = np.ones(N)
        s0 = s0/s0.sum()
        
        s_scdt = []
        for i in range(Xtest.shape[0]):
            if Ttest is None:
                s_scdt.append(self.calc_scdt(Xtest[i],t0,s0,t0))
            else:
                s_scdt.append(self.calc_scdt(Xtest[i],Ttest[i],s0,t0))
        X = np.stack(s_scdt)
        
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


    def calc_scdt(self, sig1, t1, s0, t0):
        # sig1: (0, columns)
        # t1: domain of sig1
        
        scdt = SCDT(reference=s0,x0=t0)
        Ipos, Ineg, Imasspos, Imassneg = scdt.stransform(sig1, t1)
        
        if self.rm_edge:
            shat = np.concatenate((Ipos[1:-2],Ineg[1:-2]),axis=0)
        else:
            shat = np.concatenate((Ipos[:-1],Ineg[:-1]),axis=0)
        return shat
        
    def add_trans_samples(self, scdt_features):
        # scdt_features: (n_samples, scdt)
        # deformation vector for  translation
        v1 = np.ones([1, scdt_features.shape[1]])
        return np.concatenate([scdt_features, v1])
