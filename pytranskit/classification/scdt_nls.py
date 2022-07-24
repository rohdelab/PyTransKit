
import numpy as np
import numpy.linalg as LA

from pytranskit.optrans.continuous.scdt import SCDT
from sklearn.model_selection import train_test_split

class SCDT_NLS:
    def __init__(self, num_classes, rm_edge = False):
        """
        Parameters
        ----------
        num_classes : integer, total number of classes
        rm_edge : boolean flag; IF TRUE the first and last points of CDTs will be removed
            default = False
        """
        self.num_classes = num_classes
        self.rm_edge = rm_edge
        self.Nset = []
        self.subspaces = []
        self.len_subspace = 0
        self.k = 1
        self.label = []
        self.pca_basis = []
        self.N = 1

    def fit(self, X, Y, Ttrain=None, no_local_enrichment=True):
        """Fit SCDT-NLS.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_columns)
            1D data for training.
        Y : ndarray of shape (n_samples,)
            Labels of the training samples.
        Ttrain : [optional] array-like, shape (n_samples, n_columns)
            domain for corresponding training signals.
        no_local_enrichment: [optional] boolean, default TRUE
            IF FALSE, apply deformation while searching k samples
        """
        
        # calculate the SCDTs
        print('+++++++++++ Training Phase +++++++++++')
        print('\nCalculating SCDTs for training data ...\n')
        #Xcdt = self.cdt_parallel(Xtrain)
        
        N = X.shape[1]
        t0 = np.linspace(0,1,N) # Domain of the reference
        s0 = np.ones(N)
        s0 = s0/s0.sum()
        self.t0 = t0
        self.s0 = s0
        
        s_scdt = []
        for i in range(X.shape[0]):
            if Ttrain is None:
                s_scdt.append(self.calc_scdt(X[i],t0,s0,t0))
            else:
                s_scdt.append(self.calc_scdt(X[i],Ttrain[i],s0,t0))
        Xscdt = np.stack(s_scdt)
        Xtrain, Xval, Ytrain, Yval = train_test_split(Xscdt, Y, test_size=0.3, random_state=0)
        self.bas = []
        for class_idx in range(self.num_classes):
            # generate the bases vectors
            class_data = Xtrain[Ytrain == class_idx]
            self.Nset.append(class_data)
            self.label.append(class_idx)
            bas = []
            for j in range(class_data.shape[0]):
                flat = np.copy(class_data[j].reshape(1,-1))
                u, s, vh = LA.svd(flat,full_matrices=False)
                bas.append(vh[:flat.shape[0]])
            self.bas.append(bas)
            
        if Xtrain.shape[0]//self.num_classes == 1:
            self.k = 1
        else:
            smp_class = []
            for i in range(len(np.unique(Ytrain))):
                smp_class.append(np.count_nonzero(Ytrain == i))
            # k_range = range(1,min(smp_class)) # min(min(smp_class),100)
            k_range = range(1,min(min(smp_class),100))
            n_range = range(-1,6)
            print('Tune parameters using validation set ...\n')
            self.k, self.N = self.find_kN(Xval, Yval, k_range, n_range)        
        self.Nset = []
        self.label = []
        self.bas = []
        for class_idx in range(self.num_classes):
            # generate the bases vectors
            class_data = Xscdt[Y == class_idx]
            self.Nset.append(class_data)
            self.label.append(class_idx)
            bas = []
            for j in range(class_data.shape[0]):
                if no_local_enrichment:
                    flat = np.copy(class_data[j].reshape(1,-1))
                else:
                    flat = self.enrichment(class_data[j].reshape(1,-1), k=self.N) # k=0 => translation only
                u, s, vh = LA.svd(flat,full_matrices=False)
                bas.append(vh[:flat.shape[0]])
            self.bas.append(bas)

    def predict(self, Xtest, Ttest=None, k=None, N=None):
        """Predict using SCDT-NLS
        Parameters
        ----------
        Xtest : array-like, shape (n_samples, n_columns)
            1D data for testing.
        Ttest : [optional] array-like, shape (n_samples, n_columns)
            domain for corresponding test signals.
        k : [pre-tuned parameter] number of closest points to test sample
        N : [pre-tuned parameter] number of sinusoidal bases used for subspace enrrichment
            
        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per sample in Xtest.
        """  

        if k is not None:
            k_opt = k
        else:
            k_opt = self.k
            
        if N is not None:
            n_opt = N
        else:
            n_opt = self.N
        print('+++++++++++ Testing Phase +++++++++++')
        print('\nCalculating SCDTs for testing data ...\n')
            
        s_scdt = []
        for i in range(Xtest.shape[0]):
            if Ttest is None:
                s_scdt.append(self.calc_scdt(Xtest[i],self.t0,self.s0,self.t0))
            else:
                s_scdt.append(self.calc_scdt(Xtest[i],Ttest[i],self.s0,self.t0))
        X = np.stack(s_scdt)
            
        print('Apply NLS algorithm in SCDT domain\n')
        D = []
        for class_idx in range(self.num_classes):
            Xi = self.Nset[class_idx]
            Xi_bas = self.bas[class_idx]
            d = np.zeros([X.shape[0],1])
            B = []
            L_basis = []
            for i in range(X.shape[0]):
                x = X[i,:]
                dist_i = []
                    
                for j in range(Xi.shape[0]):
                    basj = Xi_bas[j]#[:self.len_subspace,:]
                    projR = x @ basj.T  @ basj  # (n_samples, n_features)
                    # projR = x @ flat.T@np.linalg.inv(flat@flat.T + lmd*np.identity(flat.shape[0]))@flat
                    dist_i.append(LA.norm(projR - x))
                dist_i = np.stack(dist_i)
                
                indx = dist_i.argsort()[:k_opt]
                #Ni = np.concatenate([Xi[indx[0:k_opt],:], V], axis=0)
                Ni = self.enrichment(Xi[indx,:], k=n_opt) # k=0 => translation only
                
                u, s, vh = LA.svd(Ni,full_matrices=False)
                
                cum_s = np.cumsum(s)
                cum_s = cum_s/np.max(cum_s)
                basis = vh[:Ni.shape[0]]
                B.append(basis)
                L_basis.append((np.where(cum_s>=0.99)[0])[0]+1)
            max_basis = min(L_basis)
            for i in range(X.shape[0]):
                x = X[i,:]
                basis = B[i][:max_basis,:]
                
                proj = x @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                d[i]=LA.norm(projR - x)
                
            D.append(np.squeeze(d))


        D = np.stack(D, axis=0)
        preds = np.argmin(D, axis=0)
        pred_label = [self.label[i] for i in preds]
        return  pred_label
    
    def score(self, X, y):
        #print('Optimum k: {}'.format(self.k))
        #print('Optimum N: {}'.format(self.N))
        n = X.shape[0]
        y_pred = self.predict(X)
        n_correct = np.sum(y_pred == y)
        return n_correct/n, y_pred

    def calc_scdt(self, sig1, t1, s0, t0):
        # sig1: (0, columns)
        # t1: domain of sig1
        
        scdt = SCDT(reference=s0,x0=t0)
        Ipos, Ineg, Imasspos, Imassneg = scdt.stransform(sig1, t1)
        
        if self.rm_edge:
            shat = np.concatenate((Ipos[1:-2],Ineg[1:-2],Imasspos.reshape(1),Imassneg.reshape(1)),axis=0)
        else:
            shat = np.concatenate((Ipos[:-1],Ineg[:-1],Imasspos.reshape(1),Imassneg.reshape(1)),axis=0)
        return shat
    
    def find_kN(self, X, y, k_range, n_range):
        n = X.shape[0]        
        max_acc = 0.
        score_prev = 0.
        k_opt = 1
        count = 0
        acc_count = 0

        ### calculate distances for samples in validation set
        indx = []
        for i in range(X.shape[0]):
            x = np.copy(X[i,:])
            indXi = []
            for class_idx in range(self.num_classes):
                Xi = self.Nset[class_idx]
                Xi_bas = self.bas[class_idx]
                dist_i = []

                for j in range(Xi.shape[0]):
                    basj = Xi_bas[j]#[:self.len_subspace,:]
                    projR = x @ basj.T  @ basj  # (n_samples, n_features)
                    dist_i.append(LA.norm(projR - x))
                dist_i = np.stack(dist_i)

                indXi.append(dist_i.argsort()[:max(k_range)+1])
            indx.append(indXi)

        ### tune k using validation set
        for k in k_range:
            D = []
            for class_idx in range(self.num_classes):
                Xi = self.Nset[class_idx]
                d = np.zeros([X.shape[0],1])
                B = []
                L_basis = []
                for i in range(X.shape[0]):
                    x = np.copy(X[i,:])
                    ind = indx[i][class_idx]
                    Ni = np.copy(Xi[ind[:k],:])
                    u, s, vh = LA.svd(Ni,full_matrices=False)
                    cum_s = np.cumsum(s)
                    cum_s = cum_s/np.max(cum_s)
                    basis = vh[:Ni.shape[0]]
                    B.append(basis)
                    L_basis.append((np.where(cum_s>=0.99)[0])[0]+1)
                max_basis = min(L_basis)
                for i in range(X.shape[0]):
                    x = np.copy(X[i,:])
                    basis = B[i][:max_basis,:]
                    projR = x @ basis.T @ basis  # (n_samples, n_features)
                    d[i]=LA.norm(projR - x)
                D.append(np.squeeze(d))
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            pred_label = [self.label[i] for i in preds]
            score = (np.sum(pred_label == y))/n
            #print('Validation accuracy: {} with k = {}'.format(score, k))
            if score >= max_acc:
                max_acc = score
                k_opt = k
                acc_count = 0
            else:
                acc_count = acc_count + 1
            if score > score_prev:
                count = 0
            else:
                count = count + 1
            if count == 10 or acc_count == 20:
                break
            score_prev = score
            
        n_iter = []
        max_acc = 0.
        score_prev = 0.
        n_opt = 1
        count = 0
        acc_count = 0
        
        for n_enr in n_range:  
            #print('\nN = {}'.format(n_enr))
            n_iter.append(n_enr)
    
            D = []
            for class_idx in range(self.num_classes):
                Xi = self.Nset[class_idx]
                d = np.zeros([X.shape[0],1])
                B = []
                L_basis = []
                for i in range(X.shape[0]):
                    x = np.copy(X[i,:])
                    ind = indx[i][class_idx]
                    Ni = self.enrichment(Xi[ind[:k_opt],:], k=n_enr) # k=0 => translation only
                    u, s, vh = LA.svd(Ni,full_matrices=False)
                    cum_s = np.cumsum(s)
                    cum_s = cum_s/np.max(cum_s)
                    basis = vh[:Ni.shape[0]]
                    B.append(basis)
                    L_basis.append((np.where(cum_s>=0.99)[0])[0]+1)
                max_basis = min(L_basis)
                for i in range(X.shape[0]):
                    x = X[i,:]
                    basis = B[i][:max_basis,:]
                    projR = x @ basis.T @ basis  # (n_samples, n_features)
                    d[i]=LA.norm(projR - x)
                D.append(np.squeeze(d))
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            pred_label = [self.label[i] for i in preds]
            score = (np.sum(pred_label == y))/n
            #print('Validation accuracy: {} with k = {}'.format(score, k_opt))
            if score > max_acc or score==1.:
                max_acc = score
                acc_count = 0
                n_opt = n_enr
            else:
                acc_count = acc_count + 1
            if score > score_prev:
                count = 0
            else:
                count = count + 1
            if count == 10 or acc_count == 20:
                break
            score_prev = score            
        return k_opt, n_opt
    
    def enrichment(self, scdt_features, k):
        # scdt_features: (n_samples, scdt)
        if k<0:
            return scdt_features
        v= np.ones([1, scdt_features.shape[1]]) # add translation
        indx = 0
        for i in range(-k,k+1):
            if i != 0:
                vi = scdt_features-np.sin(i*np.pi*scdt_features)/(np.abs(i)*np.pi)
                v = np.concatenate((v,vi))            
            indx = indx+1
        return np.concatenate((scdt_features,v))
