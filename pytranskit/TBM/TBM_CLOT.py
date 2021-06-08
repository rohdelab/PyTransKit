#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:54:50 2020

@author: Imaging and Data Science Lab
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy.linalg as LA

from pytranskit.optrans.continuous import VOT2D
from pytranskit.optrans.utils import signal_to_pdf
from sklearn.decomposition import PCA
from pytranskit.optrans.decomposition import PLDA, CCA

from pytranskit.classification.utils import take_train_samples
from sklearn.metrics import accuracy_score, confusion_matrix
from pytranskit.TBM.utils import plot_confusion_matrix

class batch_VOT:
    def __init__(self, lr=0.0001, alpha=0., max_iter=300):
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.sigma=1. 
        self.epsilon=8. 
        self.total=100.
        
    def forward_seq(self, X, template):
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        x_hat = np.zeros([X.shape[0],2,X.shape[1],X.shape[2]])
        for i in range(X.shape[0]):
            img1 = signal_to_pdf(X[i,:], sigma=self.sigma, epsilon=self.epsilon, total=self.total)
            x_hat[i,:] = vot.forward(template, img1)
        return x_hat
    
    def forward(self, X, template):
        # X: (n_samples, width, height)
        self.template = template
        if len(X.shape)<3:
            Xhat = self.fun_clot_single(X)
        elif X.shape[0] == 1:
            Xhat = self.fun_clot_single(X[0,:,:])
        else:
            Xhat = self.clot_parallel(X)
        return Xhat
    
    def inverse(self, Xhat, template):
        # X: (n_samples, width, height)
        self.template = template
        if len(Xhat.shape)<4:
            X_recon = self.fun_iclot_single(Xhat)
        elif Xhat.shape[0] == 1:
            X_recon = self.fun_iclot_single(Xhat[0,:])
        else:
            X_recon = self.iclot_parallel(Xhat)
        return X_recon
    
    def fun_clot_single(self, I):
        # I: (width, height)
        img0 = signal_to_pdf(self.template, sigma=self.sigma, epsilon=self.epsilon, total=self.total)
        img1 = signal_to_pdf(I, sigma=self.sigma, epsilon=self.epsilon, total=self.total)
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        lot = vot.forward(img0, img1)
        return lot
    
    def fun_clot_batch(self, data):
        # data: (n_samples, width, height)
        dataCLOT = [self.fun_clot_single(data[j, :, :]) for j in range(data.shape[0])]
        return np.array(dataCLOT)
    
    def clot_parallel(self, X):
        # X: (n_samples, width, height)
        n_cpu = np.min([mp.cpu_count(), X.shape[0]])
        splits = np.array_split(X, n_cpu, axis=0)
        pl = mp.Pool(mp.cpu_count())
    
        dataCLOT = pl.map(self.fun_clot_batch, splits)
        clot_features = np.vstack(dataCLOT)  # (n_samples, proj_len, num_angles)
        pl.close()
        pl.join()    
        return clot_features        
 
    def fun_iclot_single(self, transport_map):
        img0 = signal_to_pdf(self.template, sigma=self.sigma, epsilon=self.epsilon, total=self.total)
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        Iiclot = vot.apply_inverse_map(transport_map, img0)
        return Iiclot
    
    def fun_iclot_batch(self, data):
        # data: (n_samples, width, height)
        dataiCLOT = [self.fun_iclot_single(data[j, :]) for j in range(data.shape[0])]
        return np.array(dataiCLOT)
    
    def iclot_parallel(self, Xhat):
        # X: (n_samples, width, height)
        n_cpu = np.min([mp.cpu_count(), Xhat.shape[0]])
        splits = np.array_split(Xhat, n_cpu, axis=0)
        pl = mp.Pool(mp.cpu_count())
    
        dataiCLOT = pl.map(self.fun_iclot_batch, splits)
        Xrecon = np.vstack(dataiCLOT)
        pl.close()
        pl.join()   
        return Xrecon
   

class VOT_PCA:
    def __init__(self, n_components=2, lr=0.0001, alpha=0., max_iter=300):
        self.n_components = n_components
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        
    def vot_pca(self, x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Mtr, self.Rtr, self.Ctr] = x_train_hat.shape
        [self.Nte, self.Mte, self.Rte, self.Cte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1)
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1)
        
        pca=PCA(n_components=self.n_components)
        self.pca_proj_tr = pca.fit_transform(x_train_hat_vec)
        self.pca_proj_te = pca.transform(x_test_hat_vec)
        b_hat = pca.inverse_transform(np.identity(self.n_components))
        self.basis_hat = np.reshape(b_hat,(self.n_components,self.Mtr,self.Rtr,self.Ctr))
        
        return self.basis_hat, self.pca_proj_tr, self.pca_proj_te
    
    def visualize(self, directions=5, points=5, SD_spread=1):
        dir_num=directions
        gI_num=points
        b_hat = self.basis_hat
        s_tilde_tr = self.pca_proj_tr
        s_tilde_te = self.pca_proj_te
        pca_dirs=b_hat[:dir_num,:]
        pca_proj=s_tilde_tr[:,:dir_num]
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        
        img0 = self.template
        h, w = img0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))
        
        ## figure 1 of 3
        viz_pca=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb=np.linspace(-SD_spread*np.std(pca_proj[:,a]),SD_spread*np.std(pca_proj[:,a]), num=gI_num)
            mode_var = np.zeros([gI_num,self.Mtr,self.Rtr,self.Ctr])
            for u in range(self.Mtr):
                for b in range(gI_num):
                    mode_var[b,u,:,:]=self.mean_tr[u,:,:]+lamb[b]*pca_dirs[a,u,:,:];  
            displacements_ = mode_var/np.sqrt(img0)
            transport_maps_ = displacements_ + np.stack((yv,xv))
    
            mode_var_recon = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):
                mode_var_recon[b,:,:] = vot.apply_inverse_map(transport_maps_[b,:], img0)
                t=mode_var_recon[b]
                t=t-np.min(t); t=t/np.max(t)
                mode_var_recon[b]=t
            viz_pca[a,:] = mode_var_recon.transpose(2,0,1).reshape(self.C,-1)
            
        for a in range(dir_num):
            if a==0:
                F1=viz_pca[a,:]
            else:
                F1=np.concatenate((F1,viz_pca[a,:]),axis=0)
        r,c=np.shape(F1)
        
        plt.figure(figsize=(7,7))
        plt.imshow(np.transpose(F1),cmap='gray')
        plt.xticks(np.linspace(r/(2*dir_num),r-r/(2*dir_num),dir_num),np.array(range(1,dir_num+1)))
        plt.xlabel('Modes of variation',fontsize=12)
        plt.yticks(np.linspace(1,c,5),np.array([-SD_spread,-SD_spread/2,0,SD_spread/2,SD_spread]))
        plt.ylabel('($\sigma$)',fontsize=12)
        plt.title('Variation along the prominant PCA modes')
        
        ## figure 2 of 3
        y_train = self.y_train
        y_test = self.y_test
        viz_dirs=viz_pca[:2,:]; proj_tr=self.pca_proj_tr[:,:2]; proj_te=self.pca_proj_te[:,:2]

        plt.figure(figsize=(18, 7))
        leg_str=['class 1','class 2']
        
        bas1=np.array([0,1])
        bas1a=bas1*np.min(proj_tr[:,0]); bas1b=bas1*np.max(proj_tr[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 1), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        y_unique=np.unique(y_train)
        for a in range(len(y_unique)):
            t=np.where(a==y_train)
            X=proj_tr[t]
            ax0.scatter(X[:,0],X[:,1],color='C'+str(a+1))
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of training data on the first 2 PCA directions');    
        ax1=plt.subplot2grid((4, 10), (3, 1), colspan=3,rowspan=1)
        xax=viz_dirs[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        ax2=plt.subplot2grid((4, 10), (0, 0), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        bas1a=bas1*np.min(proj_te[:,0]); bas1b=bas1*np.max(proj_te[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 6), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        y_unique=np.unique(y_test)
        for a in range(len(y_unique)):
            t=np.where(a==y_test)
            X=proj_te[t]
            ax0.scatter(X[:,0],X[:,1],color='C'+str(a+1))
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of test data on the first 2 PCA directions') 
        ax1=plt.subplot2grid((4, 10), (3, 6), colspan=3,rowspan=1)
        xax=viz_dirs[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]) 
        ax2=plt.subplot2grid((4, 10), (0, 5), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([])
        
        ## figure 3 of 3
        which_direction=1
        viz_dirs=viz_pca[which_direction-1:which_direction,:]; 
        proj_tr=s_tilde_tr[:,which_direction-1]; proj_te=s_tilde_te[:,which_direction-1]
        plt.figure(figsize=(16, 7))
        leg_str=['class 1','class 2']
        
        ax0=plt.subplot2grid((4, 8), (0, 0), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        y_unique=np.unique(y_train)
        for a in range(len(y_unique)):
            t=np.where(a==y_train)
            y=y_train[t]
            X=proj_tr[t]; X=np.reshape(X,(len(y)))
            if a==0:
                XX=[X]  
            else:
                XX.append(X)
        ax0.hist(XX,color=['C1','C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of training data on the first PCA direction')   
        ax1=plt.subplot2grid((4, 8), (2, 0), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]) 
        
        ax0=plt.subplot2grid((4, 8), (0, 5), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        y_unique=np.unique(y_test)
        for a in range(len(y_unique)):
            t=np.where(a==y_test)
            y=y_test[t]
            X=proj_te[t]; X=np.reshape(X,(len(y)))
            if a==0:
                XX=[X]  
            else:
                XX.append(X)
        ax0.hist(XX,color=['C1','C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of test data on the first PCA direction')    
        ax1=plt.subplot2grid((4, 8), (2, 5), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs[0,:],cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])

class VOT_PLDA:
    def __init__(self, n_components=2, lr=0.0001, alpha=0., max_iter=300):
        self.n_components = n_components
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        
    def vot_plda(self,x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Mtr, self.Rtr, self.Ctr] = x_train_hat.shape
        [self.Nte, self.Mte, self.Rte, self.Cte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1)
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1)
        
        pca=PCA()
        x_train_hat_vec_pca = pca.fit_transform(x_train_hat_vec) 
        x_test_hat_vec_pca = pca.transform(x_test_hat_vec)
        
        plda=PLDA(alpha=.001,n_components=self.n_components)

        self.plda_proj_tr = plda.fit_transform(x_train_hat_vec_pca,y_train);
        self.plda_proj_te = plda.transform(x_test_hat_vec_pca);
        b_hat = pca.inverse_transform(plda.inverse_transform(np.identity(self.n_components)))
        self.basis_hat = np.reshape(b_hat,(self.n_components,self.Mtr,self.Rtr,self.Ctr))
        
        return self.basis_hat, self.plda_proj_tr, self.plda_proj_te
    
    def visualize(self, directions=5, points=5, SD_spread=1):
        dir_num=directions
        gI_num=points
        b_hat = self.basis_hat
        s_tilde_tr = self.plda_proj_tr
        s_tilde_te = self.plda_proj_te
        plda_dirs=b_hat[:dir_num,:]
        plda_proj=s_tilde_tr[:,:dir_num]
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        
        img0 = self.template
        h, w = img0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))
        
        ## figure 1 of 3
        viz_plda=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb=np.linspace(-SD_spread*np.std(plda_proj[:,a]),SD_spread*np.std(plda_proj[:,a]), num=gI_num)
            mode_var = np.zeros([gI_num,self.Mtr,self.Rtr,self.Ctr])
            for u in range(self.Mtr):
                for b in range(gI_num):
                    mode_var[b,u,:,:]=self.mean_tr[u,:,:]+lamb[b]*plda_dirs[a,u,:,:];  
            displacements_ = mode_var/np.sqrt(img0)
            transport_maps_ = displacements_ + np.stack((yv,xv))
    
            mode_var_recon = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):
                mode_var_recon[b,:,:] = vot.apply_inverse_map(transport_maps_[b,:], img0)
                t=mode_var_recon[b]
                t=t-np.min(t); t=t/np.max(t)
                mode_var_recon[b]=t
            viz_plda[a,:] = mode_var_recon.transpose(2,0,1).reshape(self.C,-1)
            
        for a in range(dir_num):
            if a==0:
                F1=viz_plda[a,:]
            else:
                F1=np.concatenate((F1,viz_plda[a,:]),axis=0)
        r,c=np.shape(F1)
        
        plt.figure(figsize=(7,7))
        plt.imshow(np.transpose(F1),cmap='gray')
        plt.xticks(np.linspace(r/(2*dir_num),r-r/(2*dir_num),dir_num),np.array(range(1,dir_num+1)))
        plt.xlabel('Modes of variation',fontsize=12)
        plt.yticks(np.linspace(1,c,5),np.array([-SD_spread,-SD_spread/2,0,SD_spread/2,SD_spread]))
        plt.ylabel('($\sigma$)',fontsize=12)
        plt.title('Variation along the prominant PLDA modes')
        
        ## figure 2 of 3
        y_train = self.y_train
        y_test = self.y_test
        viz_dirs=viz_plda[:2,:]; proj_tr=self.plda_proj_tr[:,:2]; proj_te=self.plda_proj_te[:,:2]

        plt.figure(figsize=(18, 7))
        leg_str=['class 1','class 2']
        
        bas1=np.array([0,1])
        bas1a=bas1*np.min(proj_tr[:,0]); bas1b=bas1*np.max(proj_tr[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 1), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        y_unique=np.unique(y_train)
        for a in range(len(y_unique)):
            t=np.where(a==y_train)
            X=proj_tr[t]
            ax0.scatter(X[:,0],X[:,1],color='C'+str(a+1))
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of training data on the first 2 PLDA directions')    
        ax1=plt.subplot2grid((4, 10), (3, 1), colspan=3,rowspan=1)
        xax=viz_dirs[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]) 
        ax2=plt.subplot2grid((4, 10), (0, 0), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        bas1a=bas1*np.min(proj_te[:,0]); bas1b=bas1*np.max(proj_te[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 6), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        y_unique=np.unique(y_test)
        for a in range(len(y_unique)):
            t=np.where(a==y_test)
            X=proj_te[t]
            ax0.scatter(X[:,0],X[:,1],color='C'+str(a+1))
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of test data on the first 2 PLDA directions');  
        ax1=plt.subplot2grid((4, 10), (3, 6), colspan=3,rowspan=1)
        xax=viz_dirs[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        ax2=plt.subplot2grid((4, 10), (0, 5), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        ## figure 3 of 3
        which_direction=1
        viz_dirs=viz_plda[which_direction-1:which_direction,:]; 
        proj_tr=s_tilde_tr[:,which_direction-1]; proj_te=s_tilde_te[:,which_direction-1]
        plt.figure(figsize=(16, 7))
        leg_str=['class 1','class 2']
        
        ax0=plt.subplot2grid((4, 8), (0, 0), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        y_unique=np.unique(y_train)
        for a in range(len(y_unique)):
            t=np.where(a==y_train)
            y=y_train[t]
            X=proj_tr[t]; X=np.reshape(X,(len(y)))
            if a==0:
                XX=[X]  
            else:
                XX.append(X)
        ax0.hist(XX,color=['C1','C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of training data on the first PLDA direction')   
        ax1=plt.subplot2grid((4, 8), (2, 0), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]) 
        
        ax0=plt.subplot2grid((4, 8), (0, 5), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        y_unique=np.unique(y_test)
        for a in range(len(y_unique)):
            t=np.where(a==y_test)
            y=y_test[t]
            X=proj_te[t]; X=np.reshape(X,(len(y)))
            if a==0:
                XX=[X]  
            else:
                XX.append(X)
        ax0.hist(XX,color=['C1','C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of test data on the first PLDA direction')    
        ax1=plt.subplot2grid((4, 8), (2, 5), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs[0,:],cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])

class VOT_CCA:
    def __init__(self, n_components=2, lr=0.0001, alpha=0., max_iter=300):
        self.n_components = n_components
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter  
        
    def vot_cca(self, x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Mtr, self.Rtr, self.Ctr] = x_train_hat.shape
        [self.Nte, self.Mte, self.Rte, self.Cte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1)
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1)
        
        pca=PCA()
        x_train_hat_vec_pca = pca.fit_transform(x_train_hat_vec) 
        x_test_hat_vec_pca = pca.transform(x_test_hat_vec)
        t0=np.where(0==y_train); t1=np.where(1==y_train);
        X_tr=x_train_hat_vec_pca[t0];Y_tr=x_train_hat_vec_pca[t1]
        t0=np.where(0==y_test); t1=np.where(1==y_test);
        X_te=x_test_hat_vec_pca[t0];Y_te=x_test_hat_vec_pca[t1]

        n_components=self.n_components
        cca=CCA(n_components=n_components)

        self.cca_proj_tr1,self.cca_proj_tr2 = cca.fit_transform(X_tr,Y_tr);
        self.cca_proj_te1,self.cca_proj_te2 = cca.transform(X_te,Y_te);
        b_hat1,b_hat2 = pca.inverse_transform(cca.inverse_transform(np.identity(n_components),np.identity(n_components))); 
        self.basis_hat1=np.reshape(b_hat1,(n_components,self.Mtr,self.Rtr,self.Ctr))
        self.basis_hat2=np.reshape(b_hat2,(n_components,self.Mtr,self.Rtr,self.Ctr))
        return self.basis_hat1, self.basis_hat2, self.cca_proj_tr1,self.cca_proj_tr2, self.cca_proj_te1,self.cca_proj_te2
    
    def visualize(self, directions=5, points=5,  SD_spread=1):
        dir_num=directions
        gI_num=points
        b_hat1 = self.basis_hat1
        b_hat2 = self.basis_hat2
        s_tilde_tr1 = self.cca_proj_tr1
        s_tilde_tr2 = self.cca_proj_tr2
        s_tilde_te1 = self.cca_proj_te1
        s_tilde_te2 = self.cca_proj_te2
        cca_dirs1=b_hat1[:dir_num,:]
        cca_dirs2=b_hat2[:dir_num,:]
        cca_proj1=s_tilde_tr1[:,:dir_num]
        cca_proj2=s_tilde_tr2[:,:dir_num]
        vot = VOT2D(lr=self.lr, alpha=self.alpha, max_iter=self.max_iter, verbose=0)
        
        img0 = self.template
        h, w = img0.shape
        xv, yv = np.meshgrid(np.arange(w,dtype=float), np.arange(h,dtype=float))
        
        ## figure 1 of 3
        viz_cca1=np.zeros((dir_num,self.R,self.C*gI_num))
        viz_cca2=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb1=np.linspace(-SD_spread*np.std(cca_proj1[:,a]),SD_spread*np.std(cca_proj1[:,a]), num=gI_num)
            lamb2=np.linspace(-SD_spread*np.std(cca_proj2[:,a]),SD_spread*np.std(cca_proj2[:,a]), num=gI_num)
            mode_var1 = np.zeros([gI_num,self.Mtr,self.Rtr,self.Ctr]) 
            mode_var2 = np.zeros([gI_num,self.Mtr,self.Rtr,self.Ctr])
            for u in range(self.Mtr):
                for b in range(gI_num):
                    mode_var1[b,u,:,:]=self.mean_tr[u,:,:]+lamb1[b]*cca_dirs1[a,u,:,:]
                    mode_var2[b,u,:,:]=self.mean_tr[u,:,:]+lamb2[b]*cca_dirs2[a,u,:,:]
            displacements_1 = mode_var1/np.sqrt(img0)
            transport_maps_1 = displacements_1 + np.stack((yv,xv))
            displacements_2 = mode_var2/np.sqrt(img0)
            transport_maps_2 = displacements_2 + np.stack((yv,xv))
    
            mode_var_recon1 = np.zeros([gI_num,self.R,self.C])
            mode_var_recon2 = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):
                mode_var1[b,:,:]=self.mean_tr+lamb1[b]*cca_dirs1[a,:]
                mode_var2[b,:,:]=self.mean_tr+lamb2[b]*cca_dirs2[a,:]
                mode_var_recon1[b,:,:] = vot.apply_inverse_map(transport_maps_1[b,:], img0)
                mode_var_recon2[b,:,:] = vot.apply_inverse_map(transport_maps_2[b,:], img0)
                t1=mode_var_recon1[b]; t2=mode_var_recon2[b]
                t1=t1-np.min(t1); t1=t1/np.max(t1)
                t2=t2-np.min(t2); t2=t2/np.max(t2)
                mode_var_recon1[b]=t1
                mode_var_recon2[b]=t2 
           
            viz_cca1[a,:] = mode_var_recon1.transpose(2,0,1).reshape(self.C,-1)
            viz_cca2[a,:] = mode_var_recon2.transpose(2,0,1).reshape(self.C,-1)
            
        for a in range(dir_num):
            if a==0:
                F1=viz_cca1[a,:]; F2=viz_cca2[a,:]
            else:
                F1=np.concatenate((F1,viz_cca1[a,:]),axis=0)
                F2=np.concatenate((F2,viz_cca2[a,:]),axis=0)
        r1,c1=np.shape(F1); r2,c2=np.shape(F2)
        
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 8), sharex=True, sharey=True)
        ax0.imshow(np.transpose(F1),cmap='gray')
        ax0.set_xlabel('Modes of variation',fontsize=12)
        ax0.set_ylabel('($\sigma$)',fontsize=12)
        ax0.set_title('Variation along the prominant CCA modes (Class 0)')
        plt.xticks(np.linspace(r1/(2*dir_num),r1-r1/(2*dir_num),dir_num),np.array(range(1,dir_num+1)))
        plt.yticks(np.linspace(1,c1,5),np.array([-SD_spread,-SD_spread/2,0,SD_spread/2,SD_spread]))
        
        ax1.imshow(np.transpose(F2),cmap='gray')
        ax1.set_xlabel('Modes of variation',fontsize=12)
        ax1.set_ylabel('($\sigma$)',fontsize=12)
        ax1.set_title('Variation along the prominant CCA modes (Class 1)')
        plt.xticks(np.linspace(r1/(2*dir_num),r1-r1/(2*dir_num),dir_num),np.array(range(1,dir_num+1)))
        plt.yticks(np.linspace(1,c1,5),np.array([-SD_spread,-SD_spread/2,0,SD_spread/2,SD_spread]))
        plt.show()
        
        ## figure 2 of 3
        viz_dirs1=viz_cca1[:2,:]; viz_dirs2=viz_cca2[:2,:]
        proj_tr1=s_tilde_tr1[:,:2]; proj_tr2=s_tilde_tr2[:,:2]
        proj_te1=s_tilde_te1[:,:2]; proj_te2=s_tilde_te2[:,:2]
        
        plt.figure(figsize=(18, 7))
        leg_str=['Variable X','Variable Y']
        
        bas1=np.array([0,1])
        bas1a=bas1*np.min(proj_tr1[:,0]); bas1b=bas1*np.max(proj_tr1[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 1), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        
        X=proj_tr1; Y=proj_tr2;
        ax0.scatter(X[:,0],X[:,1],color='C'+str(1))
        ax0.scatter(Y[:,0],Y[:,1],color='C'+str(2))
        
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of training data on the first 2 CCA directions');    
        ax1=plt.subplot2grid((4, 10), (3, 1), colspan=3,rowspan=1)
        xax=viz_dirs1[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        ax2=plt.subplot2grid((4, 10), (0, 0), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs1[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([]); ax2.set_yticks([]); 
        
        bas1a=bas1*np.min(proj_te1[:,0]); bas1b=bas1*np.max(proj_te1[:,0])
        basy=[bas1a[0],bas1b[0]]; basx=[bas1a[1],bas1b[1]]
        ax0=plt.subplot2grid((4, 10), (0, 6), colspan=3,rowspan=3)
        ax0.grid(linestyle='--')
        
        X=proj_te1; Y=proj_te2;
        ax0.scatter(X[:,0],X[:,1],color='C'+str(1))
        ax0.scatter(Y[:,0],Y[:,1],color='C'+str(2))
        
        ax0.legend(leg_str)
        ax0.plot(basx,basy,color='C4')
        ax0.set_title('Projection of test data on the first 2 CCA directions');  
        ax1=plt.subplot2grid((4, 10), (3, 6), colspan=3,rowspan=1)
        xax=viz_dirs1[0,:]
        ax1.imshow(xax,cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        ax2=plt.subplot2grid((4, 10), (0, 5), colspan=1,rowspan=3)
        yax=np.transpose(viz_dirs1[1,:])
        ax2.imshow(yax,cmap='gray')
        ax2.set_xticks([]); ax2.set_yticks([])
        
        ## figure 3 of 3
        which_direction=1
        viz_dirs1=viz_cca1[which_direction-1:which_direction,:]
        viz_dirs2=viz_cca2[which_direction-1:which_direction,:]
        proj_tr1=s_tilde_tr1[:,which_direction-1]; proj_te1=s_tilde_te1[:,which_direction-1]
        proj_tr2=s_tilde_tr2[:,which_direction-1]; proj_te2=s_tilde_te2[:,which_direction-1]
        
        plt.figure(figsize=(16, 7))
        
        leg_str=['Variable X']
        ax0=plt.subplot2grid((4, 8), (0, 0), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        XX=proj_tr1
        ax0.hist(XX,color=['C1']); ax0.legend(leg_str)
        ax0.set_title('Projection of training data on the first CCA direction');    
        ax1=plt.subplot2grid((4, 8), (2, 0), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs1[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        
        ax0=plt.subplot2grid((4, 8), (0, 5), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        XX=proj_te1
        ax0.hist(XX,color=['C1']); ax0.legend(leg_str)
        ax0.set_title('Projection of test data on the first CCA direction');    
        ax1=plt.subplot2grid((4, 8), (2, 5), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs1[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]);
        
        
        plt.figure(figsize=(16, 7))
        
        leg_str=['Variable Y']
        ax0=plt.subplot2grid((4, 8), (0, 0), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        XX=proj_tr2
        ax0.hist(XX,color=['C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of training data on the first CCA direction');    
        ax1=plt.subplot2grid((4, 8), (2, 0), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs2[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([]); 
        
        ax0=plt.subplot2grid((4, 8), (0, 5), colspan=3,rowspan=2); ax0.grid(linestyle='--')
        XX=proj_te2
        ax0.hist(XX,color=['C2']); ax0.legend(leg_str)
        ax0.set_title('Projection of test data on the first CCA direction');    
        ax1=plt.subplot2grid((4, 8), (2, 5), colspan=3,rowspan=1)
        ax1.imshow(viz_dirs2[0,:],cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([])
        
class VOT_NS_Classifier:
    def __init__(self, train_sample=None, use_gpu=False):
        self.train_sample = train_sample
        self.subspaces = []
        self.label = []
        self.len_subspace = 0
        self.use_gpu = use_gpu
        
    def classify_VOT_NS(self,x_train, y_train, x_test, y_test):
        train_sample = self.train_sample
        numclass = len(np.unique(y_train))
        self.num_classes = numclass

        if train_sample is not None:
            # Calculate number of samples of the class with smallest number of train samples
            unique, count = np.unique(y_train, return_counts=True)
            mincount = np.min(count)
            train_sample = np.min([train_sample, mincount])
            x_train_sub, y_train_sub = take_train_samples(x_train, y_train, train_sample, 
                                                      numclass, repeat=0)                  # function from utils.py
        else:
            x_train_sub, y_train_sub = x_train, y_train
            
        self.fit(x_train_sub, y_train_sub)
        y_predicted = self.predict(x_test)
        
        accuracy = 100*self.score(y_test) 
        print("Accuracy: {:0.2f}%".format(accuracy))
        
        conf_mat = confusion_matrix(y_test, y_predicted)
        print('Confusion Matrix:')
        target_names = []
        for c in range(numclass):
            class_label = 'Class '+str(c)
            target_names.append(class_label)
        plot_confusion_matrix(conf_mat, target_names)
        
        return y_predicted

    def fit(self, X, y):
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
        for class_idx in range(self.num_classes):
            # generate the bases vectors
            class_data = X[y == class_idx]
            flat = class_data.reshape(class_data.shape[0], -1)
            #flat = np.transpose(class_data,(0,2,1)).reshape(class_data.shape[0],-1)
            
            u, s, vh = LA.svd(flat,full_matrices=False)
            
            cum_s = np.cumsum(s)
            cum_s = cum_s/np.max(cum_s)

            max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            
            if max_basis > self.len_subspace:
                self.len_subspace = max_basis
            
            basis = vh[:flat.shape[0]]
            
            self.subspaces.append(basis)
            self.label.append(class_idx)

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
        if self.use_gpu:
            import cupy as cp
        X = X.reshape([X.shape[0], -1])
        #X = np.transpose(X,(0,2,1)).reshape(X.shape[0],-1)
        print('Len basis: {}'.format(self.len_subspace))
        D = []
        for class_idx in range(self.num_classes):
            basis = self.subspaces[class_idx]
            basis = basis[:self.len_subspace,:]
            
            
            if self.use_gpu:
                D.append(cp.linalg.norm(cp.matmul(cp.matmul(X, cp.array(basis).T), 
                                                  cp.array(basis)) -X, axis=1))
            else:
                proj = X @ basis.T  # (n_samples, n_basis)
                projR = proj @ basis  # (n_samples, n_features)
                D.append(LA.norm(projR - X, axis=1))
        if self.use_gpu:
            preds = cp.argmin(cp.stack(D, axis=0), axis=0)
            self.preds_label = [self.label[i] for i in cp.asnumpy(preds)]
            return self.preds_label
        else:
            D = np.stack(D, axis=0)
            preds = np.argmin(D, axis=0)
            self.preds_label = [self.label[i] for i in preds]
            return  self.preds_label
        
    def score(self, y_test):
        return accuracy_score(y_test, self.preds_label)

        
        