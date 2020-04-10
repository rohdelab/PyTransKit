import numpy as np
import numpy.linalg as LA
import cupy as cp

def add_trans_samples(rcdt_features, theta):
    # rcdt_features: (n_samples, proj_len, num_angles)
    # deformation vectors for  translation
    v1, v2 = np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)
    v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
    v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
    return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])


class RCDT_NS:
    def __init__(self, theta, no_deform_model=False, use_image_feature=False, count_flops=False, use_gpu=False):
        """
        Parameters
        ----------
        theta : array-like, angles in degrees for taking radon projections
            default = [0,180) with increment of 4 degrees.
        """
        self.num_classes = None
        self.subspaces = []
        self.len_subspace = 0

        self.theta = theta
        self.no_deform_model = no_deform_model
        self.use_image_feature = use_image_feature
        self.count_flops = count_flops
        self.use_gpu = use_gpu

    def fit(self, X, y, num_classes):
        """Fit linear model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_proj, n_angles))
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        self :
            Returns an instance of self.
        """
        if self.count_flops:
            assert X.dtype == np.float64
        self.num_classes = num_classes
        for class_idx in range(num_classes):
            # generate the bases vectors
            class_data = X[y == class_idx]
            if self.no_deform_model or self.use_image_feature:
                flat = class_data.reshape(class_data.shape[0], -1)
            else:
                class_data_trans = add_trans_samples(class_data, self.theta)
                flat = class_data_trans.reshape(class_data_trans.shape[0], -1)
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            self.subspaces.append(basis)

            if self.count_flops:
               assert basis.dtype == np.float64

    def predict(self, X):
        """Predict using the linear model
        Parameters
        ----------
        X : array-like, sparse matrix, shape (n_samples, n_proj, n_angles))
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        X = X.reshape([X.shape[0], -1])
        #X = np.transpose(X,(0,2,1)).reshape(X.shape[0],-1)
        D = []
        for class_idx in range(self.num_classes):
            basis = self.subspaces[class_idx]
            basis = basis[:self.len_subspace,:]
            
            if self.use_gpu:
                D.append(cp.linalg.norm(cp.matmul(cp.matmul(X, cp.array(basis).T), cp.array(basis)) -X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if self.use_gpu:
            preds = cp.argmin(cp.stack(D, axis=0), axis=0)
            return cp.asnumpy(preds)
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            return preds

