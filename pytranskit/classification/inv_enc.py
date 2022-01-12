
import numpy as np
import numpy.linalg as LA
import multiprocessing as mp
import os
import h5py

from pytranskit.optrans.continuous.radoncdt import RadonCDT

eps = 1e-6
x0_range = [-1, 1]
x_range = [-1, 1]
Rdown = 4  # downsample radon projections (w.r.t. angles)
theta = np.linspace(0, 176, 180 // Rdown)

class INV_ENC:
    def __init__(self, num_classes, thetas=theta, rm_edge=False):
        """
        Parameters
        ----------
        num_classes : integer, total number of classes
        thetas : array-like, angles in degrees for taking radon projections
            default = [0,180) with increment of 4 degrees.
        rm_edge : boolean flag; IF TRUE the first and last points of RCDTs will be removed
            default = False
        """
        self.num_classes = num_classes
        self.thetas = thetas
        self.rm_edge = rm_edge
        self.subspaces = []
        self.len_subspace = 0

    def fit(self, Xtrain, Ytrain, no_deform_model=False):
        """Fit linear model.
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_rows, n_columns)
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
                # class_data_trans = self.add_trans_samples(class_data)
                class_data_trans = self.add_affine_samples(class_data)
                flat = class_data_trans.reshape(class_data_trans.shape[0], -1)
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            self.subspaces.append(basis)


    def predict(self, Xtest, use_gpu=False, datanm='nothing'):
        """Predict using the linear model
        
        Let :math:`B^k` be the basis vectors of class :math:`k`, and :math:`x` be the RCDT sapce feature vector of an input, 
        the NS method performs classification by
        
        .. math::
            arg\min_k \| B^k (B^k)^T x - x\|^2
        
        Parameters
        ----------
        Xtest : array-like, shape (n_samples, n_rows, n_columns)
            Image data for testing.
        use_gpu: boolean flag; IF TRUE, use gpu for calculations
            default = False.
            
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in Xtest.
           
        """
        
        # calculate the RCDT using parallel CPUs
        

        fnm='./metadata/'+datanm+'_test_rcdt.hdf5';
        if os.path.exists(fnm):
            print('\nExists - Loading test RCDTs ...')
            with h5py.File(fnm, 'r') as f:
                XrcdtBig = f['XrcdtBig'][()]
                XrcdtBig = XrcdtBig.astype(np.float32)
        else:
            print('\nDoesnot exist - Calculating and saving test RCDTs ...')
            Xrcdt = self.rcdt_parallel(Xtest)
            XrcdtBig=self.rcdt_permut(Xrcdt) 
            with h5py.File(fnm, 'w') as f:
                XrcdtBig = XrcdtBig.astype(np.float32)
                f.create_dataset('XrcdtBig', data= XrcdtBig)
        # print('\nCalculating (and not saving) test RCDTs ...')
        # Xrcdt = self.rcdt_parallel(Xtest)
        
        print('RCDT completed')
                
        # XrcdtBig=self.rcdt_permut(Xrcdt)    
        NN=XrcdtBig.shape[0]
        # NN=1

        # print(XrcdtBig.shape)
        # print(XrcdtBig.shape[1])
        # print(sdasd)

        # find nearest subspace for each test sample
        print('Finding NS on tests: ',end='-')

        Dbig=np.zeros((NN,self.num_classes,XrcdtBig.shape[1]))
        for ii in range(NN):
            print(ii+1,end='-')
            
            Xrcdt=XrcdtBig[ii]
            # vectorize RCDT matrix
            X = Xrcdt.reshape([Xrcdt.shape[0], -1])
            
            D = []
            for class_idx in range(self.num_classes):
                basis = self.subspaces[class_idx]
                basis = basis[:self.len_subspace,:]
                
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
            
            D = np.stack(D, axis=0)
            Dbig[ii]=D
        print(' ')
        
        Dfinal=np.min(Dbig,axis=0)
        
        
        # print(D.shape)
        # print(Dbig.shape)
        # print(Dfinal.shape)
        
        
        
        preds = np.argmin(Dfinal, axis=0)
        return preds


    def fun_rcdt_single(self, I):
        # I: (rows, columns)
        radoncdt = RadonCDT(self.thetas)
        template = np.ones(I.shape, dtype=I.dtype)
        Ircdt = radoncdt.forward(x0_range, template / np.sum(template), 
                                 x_range, I / np.sum(I), 
                                 self.rm_edge)
        return Ircdt
    
    def fun_rcdt_batch(self, data):
        # data: (n_samples, rows, columns)
        dataRCDT = [self.fun_rcdt_single(data[j, :, :] + eps) for j in range(data.shape[0])]
        return np.array(dataRCDT)
    
    def rcdt_parallel(self, X):
        # X: (n_samples, rows, columns)
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
        # rcdt_features: (n_samples, proj_len, num_angles)
        # deformation vectors for  translation
        v1, v2 = np.cos(self.thetas*np.pi/180), np.sin(self.thetas*np.pi/180)
        v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
        v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
        return np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])

    def add_affine_samples(self, rcdt_features):
        # rcdt_features: (n_samples, proj_len, num_angles)
        # deformation vectors for  translation, scaling, shear
        aa=0.2; bb=0.2;

        v1, v2 = np.cos(self.thetas*np.pi/180), np.sin(self.thetas*np.pi/180)
        v3o, v4o = np.cos(self.thetas*np.pi/180)*np.cos(self.thetas*np.pi/180), np.sin(self.thetas*np.pi/180)*np.sin(self.thetas*np.pi/180)
        v5o = bb*bb*np.cos(self.thetas*np.pi/180)*np.cos(self.thetas*np.pi/180) + bb*np.sin(self.thetas*np.pi/90)
        v6o = aa*aa*np.sin(self.thetas*np.pi/180)*np.sin(self.thetas*np.pi/180) + aa*np.sin(self.thetas*np.pi/90)

        v1 = np.repeat(v1[np.newaxis], rcdt_features.shape[1], axis=0)
        v2 = np.repeat(v2[np.newaxis], rcdt_features.shape[1], axis=0)
        v3o = np.repeat(v3o[np.newaxis], rcdt_features.shape[1], axis=0)
        v4o = np.repeat(v4o[np.newaxis], rcdt_features.shape[1], axis=0)
        v5o = np.repeat(v5o[np.newaxis], rcdt_features.shape[1], axis=0)
        v6o = np.repeat(v6o[np.newaxis], rcdt_features.shape[1], axis=0)

        outres=np.concatenate([rcdt_features, v1[np.newaxis], v2[np.newaxis]])
        
        # print("shape before")
        # print(outres.shape)
        
        # print("also loop time")
        # print(rcdt_features.shape[0])

        for a in range(rcdt_features.shape[0]):
            v3 = v3o*rcdt_features[a]; 
            v4 = v4o*rcdt_features[a]; 
            v5 = v5o*rcdt_features[a]; 
            v6 = v6o*rcdt_features[a]
            outres=np.concatenate([outres, v3[np.newaxis], v4[np.newaxis], v5[np.newaxis], v6[np.newaxis]])
        
        
        # print("shape after")
        # print(outres.shape)

        

        return outres

    def rcdt_permut(self,shat):
        
        n_samp=shat.shape[0]
        Nang=10;
        
        shatBig=np.zeros((Nang*2+1,shat.shape[0],shat.shape[1],shat.shape[2]))
        for a in range(n_samp):
            temp=shat[a]
            shatBig[0,a]=temp
            
            j2=np.concatenate((temp,-np.flipud(temp)),axis=1)
            j2a=j2; j2b=j2;
            for b in range(Nang):
                j2a=np.roll(j2a, 1,axis=1)
                j2b=np.roll(j2b, -1,axis=1)
                
                tempa=j2a[:,:shat.shape[2]]
                tempb=j2b[:,:shat.shape[2]]
                
                shatBig[2*b+1,a]=tempa
                shatBig[2*b+2,a]=tempb

        # print(shat.shape)
        # print(shatBig.shape)
        # print(n_samp)
        # print(temp.shape)
        # print(shat.shape[2])

        return shatBig
