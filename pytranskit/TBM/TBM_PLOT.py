#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:14:01 2020

@author: Imaging and Data Science Lab
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy.linalg as LA

import ot
from sklearn.decomposition import PCA
from pytranskit.optrans.decomposition import PLDA, CCA

from pytranskit.classification.utils import take_train_samples
from sklearn.metrics import accuracy_score, confusion_matrix
from pytranskit.TBM.utils import plot_confusion_matrix

import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.measure import regionprops, label
from numpy import matlib as mb
import scipy as sc

from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.cluster import KMeans

def gaussian2D(x,mean,sigma):
    sigma2=sigma**2
    if len(x.shape)==1:
        return (1./(2*np.pi*sigma2))*np.exp(-((x-mean)**2).sum()/(2*sigma2))
    else: 
        return (1./(2*np.pi*sigma2))*np.exp(-((x-mean)**2).sum(1)/(2*sigma2))
    
def particle2image(x,a,sigma,imgshape):
    '''
    This function gets a set of coordinates, x, their amplitude, a, and generates a PDF image using a gaussian kernel
    '''
    X,Y=np.meshgrid(np.arange(imgshape[0]),np.arange(imgshape[1]))
    coords=np.stack([X.reshape(-1),Y.reshape(-1)],1)
    I=np.zeros(imgshape)
    for i,mean in enumerate(x):
        I+=a[i]*gaussian2D(coords,mean,sigma).reshape(imgshape)
    return I.T

def get_particles(img,N):
    thresh=.01 # If you want to discard low intensity values, otherwise set it to zero
    xfull=np.argwhere(img>thresh)
    features= np.concatenate([xfull,img[xfull[:,0],xfull[:,1]][:,np.newaxis]],1)
    kmeans=KMeans(n_clusters=N)
    kmeans.fit(features)
    x=kmeans.cluster_centers_
    return x

def particleApproximation(imgs,Nmasses):
    PPl=list();
    for i in range(imgs.shape[0]):
        img=imgs[i]
        x=get_particles(img,Nmasses)
        PPl.append(x);
    return PPl

def pLOT_single(x_temp,x_targ,a_temp,a_targ):
    C=ot.dist(x_targ,x_temp)
    w2,log=ot.emd2(a_targ,a_temp,C,return_matrix=True)
    M=x_targ.shape[0]
    gamma=log['G']
    gamma2=np.array([g/(g.sum()) for g in gamma.T]).T
    
    #V=np.matmul(gamma2.T,x_targ)-x_temp
    V=np.matmul(gamma2.T,x_targ)
    return V

class batch_PLOT:
    def __init__(self, Nmasses=50):
        self.Nmasses = Nmasses
        
    def forward_seq(self, x_train, x_test, x_template):
        N = self.Nmasses
        PPl_train=particleApproximation(x_train, N)
        PPl_test=particleApproximation(x_test,N)
        
        #x0=np.mean(x_train,axis=0)
        PPl_tem=get_particles(x_template,N)
        
        x_temp=PPl_tem[:,:2]
        a_temp=PPl_tem[:,2]/PPl_tem[:,2].sum()
        
        V=list(); M=x_train.shape[0]
        for ind in range(M):
            xa_tr=PPl_train[ind]
            x_tr=xa_tr[:,:2]
            a_tr=xa_tr[:,2]/xa_tr[:,2].sum()
            V_single=pLOT_single(x_temp,x_tr,a_temp,a_tr)
            V.append(V_single)
        V=np.asarray(V)
        
        x_train_hat=np.zeros((len(V),V[0].shape[0]*V[0].shape[1]))
        for a in range(len(V)):
            x_train_hat[a,:]=np.reshape(V[a],(V[0].shape[0]*V[0].shape[1],),order='F')
            
        V=list(); M=x_test.shape[0]
        for ind in range(M):
            xa_te=PPl_test[ind]
            x_te=xa_te[:,:2]
            a_te=xa_te[:,2]/xa_te[:,2].sum()
            V_single=pLOT_single(x_temp,x_te,a_temp,a_te)
            V.append(V_single)
        V=np.asarray(V)
        
        x_test_hat=np.zeros((len(V),V[0].shape[0]*V[0].shape[1]))
        for a in range(len(V)):
            x_test_hat[a,:]=np.reshape(V[a],(V[0].shape[0]*V[0].shape[1],),order='F')
            
        return x_train_hat, x_test_hat, x_temp, a_temp
    
class PLOT_PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    
    def plot_pca(self, x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Ptr] = x_train_hat.shape
        [self.Nte, self.Pte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1,order='F')
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1,order='F')
        
        pca=PCA(n_components=self.n_components)
        self.pca_proj_tr = pca.fit_transform(x_train_hat_vec)
        self.pca_proj_te = pca.transform(x_test_hat_vec)
        b_hat = pca.inverse_transform(np.identity(self.n_components))
        self.basis_hat = np.reshape(b_hat,(self.n_components,self.Ptr))
        
        return self.basis_hat, self.pca_proj_tr, self.pca_proj_te
    
    def visualize(self, mean_x_train_hat, Intensity, directions=5, points=5, SD_spread=2):
        dir_num=directions
        gI_num=points
        b_hat = self.basis_hat
        s_tilde_tr = self.pca_proj_tr
        s_tilde_te = self.pca_proj_te
        pca_dirs=b_hat[:dir_num,:]
        pca_proj=s_tilde_tr[:,:dir_num]
        
        ## figure 1 of 3
        viz_pca=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb=np.linspace(-SD_spread*np.std(pca_proj[:,a]),SD_spread*np.std(pca_proj[:,a]), num=gI_num)
            mode_var_recon = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):

                mode_var=mean_x_train_hat+lamb[b]*pca_dirs[a,:];
                mode_var=mode_var.reshape(int(len(mode_var)/2),-1,order='F')
                mode_var_recon[b,:]=particle2image(mode_var,Intensity,3,(self.R,self.C))
                
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
        ax2.set_yticks([])
        
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


class PLOT_PLDA:
    def __init__(self, n_components=2):
        self.n_components = n_components  
        
    def plot_plda(self, x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Ptr] = x_train_hat.shape
        [self.Nte, self.Pte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1,order='F')
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1,order='F')
        
        pca=PCA()
        x_train_hat_vec_pca = pca.fit_transform(x_train_hat_vec) 
        x_test_hat_vec_pca = pca.transform(x_test_hat_vec)
        
        plda=PLDA(alpha=1.618,n_components=self.n_components)

        self.plda_proj_tr = plda.fit_transform(x_train_hat_vec_pca,y_train);
        self.plda_proj_te = plda.transform(x_test_hat_vec_pca);
        b_hat = pca.inverse_transform(plda.inverse_transform(np.identity(self.n_components)))
        self.basis_hat = np.reshape(b_hat,(self.n_components,self.Ptr))
        
        return self.basis_hat, self.plda_proj_tr, self.plda_proj_te
    
    def visualize(self, mean_x_train_hat, Intensity, directions=5, points=5, SD_spread=2):
        dir_num=directions
        gI_num=points
        b_hat = self.basis_hat
        s_tilde_tr = self.plda_proj_tr
        s_tilde_te = self.plda_proj_te
        plda_dirs=b_hat[:dir_num,:]
        plda_proj=s_tilde_tr[:,:dir_num]
        
        ## figure 1 of 3
        viz_plda=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb=np.linspace(-SD_spread*np.std(plda_proj[:,a]),SD_spread*np.std(plda_proj[:,a]), num=gI_num)
            mode_var_recon = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):

                mode_var=mean_x_train_hat+lamb[b]*plda_dirs[a,:];
                mode_var=mode_var.reshape(int(len(mode_var)/2),-1,order='F')
                mode_var_recon[b,:]=particle2image(mode_var,Intensity,3,(self.R,self.C))
                
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
    
class PLOT_CCA:
    def __init__(self, n_components=2):
        self.n_components = n_components  
        
    def plot_cca(self, x_train_hat, y_train, x_test_hat, y_test, template):
        self.y_train = y_train
        self.y_test = y_test
        self.template = template
        [self.R, self.C] = template.shape
        [self.Ntr, self.Ptr] = x_train_hat.shape
        [self.Nte, self.Pte] = x_test_hat.shape
        
        self.mean_tr=np.mean(x_train_hat, axis=0)
        self.mean_te=np.mean(x_test_hat, axis=0)
        x_train_hat_vec=(x_train_hat - self.mean_tr).reshape(self.Ntr,-1,order='F')
        x_test_hat_vec=(x_test_hat - self.mean_tr).reshape(self.Nte,-1,order='F')
        
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
        self.basis_hat1=np.reshape(b_hat1,(n_components,self.Ptr))
        self.basis_hat2=np.reshape(b_hat2,(n_components,self.Ptr))
        return self.basis_hat1, self.basis_hat2, self.cca_proj_tr1,self.cca_proj_tr2, self.cca_proj_te1,self.cca_proj_te2
    
    def visualize(self, mean_x_train_hat, Intensity, directions=5, points=5, SD_spread=1):
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
        
        ## figure 1 of 3
        viz_cca1=np.zeros((dir_num,self.R,self.C*gI_num))
        viz_cca2=np.zeros((dir_num,self.R,self.C*gI_num))
        for a in range(dir_num):
            lamb1=np.linspace(-SD_spread*np.std(cca_proj1[:,a]),SD_spread*np.std(cca_proj1[:,a]), num=gI_num)
            lamb2=np.linspace(-SD_spread*np.std(cca_proj2[:,a]),SD_spread*np.std(cca_proj2[:,a]), num=gI_num)
            
            mode_var_recon1 = np.zeros([gI_num,self.R,self.C])
            mode_var_recon2 = np.zeros([gI_num,self.R,self.C])
            for b in range(gI_num):
                #mode_var1=Pl_tem_vec+self.mean_tr+lamb1[b]*cca_dirs1[a,:]
                #mode_var2=Pl_tem_vec+self.mean_tr+lamb2[b]*cca_dirs2[a,:]
                #mode_var_recon1[b,:]=Visualize_LOT(mode_var1,Intensity,self.R,self.C,scale=1)
                #mode_var_recon2[b,:]=Visualize_LOT(mode_var2,Intensity,self.R,self.C,scale=1)
                mode_var1=mean_x_train_hat+lamb1[b]*cca_dirs1[a,:]
                mode_var1=mode_var1.reshape(int(len(mode_var1)/2),-1,order='F')
                mode_var_recon1[b,:]=particle2image(mode_var1,Intensity,3,(self.R,self.C))
                mode_var2=mean_x_train_hat+lamb2[b]*cca_dirs2[a,:]
                mode_var2=mode_var2.reshape(int(len(mode_var2)/2),-1,order='F')
                mode_var_recon2[b,:]=particle2image(mode_var2,Intensity,3,(self.R,self.C))
                
                t1=mode_var_recon1[b]; t2=mode_var_recon2[b]
                t1=t1-np.min(t1); t1=t1/np.max(t1)
                t2=t2-np.min(t2); t2=t2/np.max(t2)
                mode_var_recon1[b]=t1; mode_var_recon2[b]=t2 
           
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
        ax1.set_xticks([]); ax1.set_yticks([]);
    
 
class PLOT_NS_Classifier:
    def __init__(self, train_sample=None, use_gpu=False):
        self.train_sample = train_sample
        self.subspaces = []
        self.label = []
        self.len_subspace = 0
        self.use_gpu = use_gpu
        
    def classify_PLOT_NS(self,x_train, y_train, x_test, y_test):
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

            #TODO: fix the bug to use original max_basis
            #max_basis = (np.where(cum_s>=0.99)[0])[0] + 1
            max_basis = 30
            
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
    
    
## ------------ PAST CODES FROM HERE ------------------------

def fromInd2Coord(ind,Ny):
    coor=np.zeros((2,len(ind)))
    coor[0,:] = np.fix(ind/Ny)+1
    coor[1,:] = np.remainder(ind,Ny)
    return coor

def L2_distance(a,b):
    b=np.reshape(b,(len(b),1))
    c=a-b
    nrm=np.zeros(c.shape[1])
    for i in range(c.shape[1]):
        tmp=c[:,i]
        nrm[i]=np.linalg.norm(tmp)
    return nrm
    
def img2pts_Lloyd(img,Nmasses):
    stopLloyd = 0.5
    [ny,nx]=np.shape(img)
    img_t = img/img.max()
    
    level = threshold_otsu(img_t)*0.22
    BW = cv2.threshold(img_t,level,1,cv2.THRESH_BINARY)[1]; BW=BW.astype(bool)
    BW = morphology.remove_small_objects(BW, min_size=7); BW=BW.astype(int); BW = label(BW); 
    
    STATS=regionprops(BW)
    bb=STATS[0].bbox
    sy=bb[0]; sx=bb[1]; Ny=bb[2]-sy; Nx=bb[3]-sx    
    
    img_t_vec=np.squeeze(np.reshape(img_t,(ny*nx,1),order='F'))
    img_vec=np.squeeze(np.reshape(img,(ny*nx,1),order='F'))
    
    w = np.squeeze(np.argwhere(img_t_vec < level))
    
    img_vec[w]=0;
    img=np.reshape(img_vec,(ny,nx),order='F')
    
    ind = np.squeeze(np.argwhere(img_t_vec>=level))

    output_Index = np.random.choice(ind,np.min((Nmasses,len(ind))),replace=False)
    output_Index=np.sort(output_Index)
    
    res_P=fromInd2Coord(output_Index+1,ny)    
    res_c=res_P[0,:]*0; res_c[:]=1

    BW=img_t*0
    img_x=img/np.sum(img) 
    BW[sy-1:sy+Ny,sx-1:sx+Nx] = img_x[sy-1:sy+Ny,sx-1:sx+Nx]
    
    BW_vec=np.reshape(BW,(nx*ny,),order='F'); ii=np.squeeze(np.argwhere(BW_vec))
    V=BW_vec[ii]
    rc=fromInd2Coord(ii+1,ny) 
    col=rc[0,:]; row=rc[1,:]
        
    tmp_row=np.reshape(row,(1,len(row))); tmp_col=np.reshape(col,(1,len(col)))
    Pl=np.concatenate((tmp_col,tmp_row),axis=0)
    
    if len(ind)<Nmasses:
        res_P2 = fromInd2Coord(ind,ny)
        nlz = np.sum(img_vec[ind])
        res_c2 = img_vec[ind]/nlz
        var_out = np.zeros((1,len(ind)))
        llerr = 0
    else:
        cur = 0; differ = 1        
        while differ>stopLloyd:
            neighbors_map = Pl[0,:]*0
            for k in range(Pl.shape[1]):
                Pk = Pl[:,k]; Pk=np.reshape(Pk,(2,1))
                BP=mb.repmat(Pk,1,res_P.shape[1])
                err=np.sum((BP-res_P)*(BP-res_P),axis=0); err[np.isnan(err)]=1e6 
        
                w=np.argwhere(err==err.min())
                neighbors_map[k]=w[0]
            
            errUB=np.zeros(res_P.shape[1])
            for k in range(res_P.shape[1]):
                w=np.argwhere(np.absolute(neighbors_map-k)<0.01)
                cx = np.sum(V[w]*Pl[0,w])/np.sum(V[w]+1e-10)
                cy = np.sum(V[w]*Pl[1,w])/np.sum(V[w]+1e-10)
                tmp_center=np.array([cx,cy]); tmp_center=np.reshape(tmp_center,(2,1))
                
                ld=L2_distance(Pl[:,np.squeeze(w)],tmp_center)
                dist_cent = ld*ld

                t1=np.reshape(V[w],(len(V[w]),)); 
                t2=np.reshape(dist_cent,(len(dist_cent),));

                if t1.shape[0]==t2.shape[0]:
                    errUB[k]=t1.dot(t2)
                else:                          
                    errUB[k]=np.sum(t1*t2) 
                
                res_P[:,k]=np.squeeze(tmp_center)
                res_c[k]=np.sum(V[w])
                
            if cur==0:
                llerr=np.sum(errUB)
            else:
                llerr=np.append(llerr,np.sum(errUB))
                
            if cur>=3:
                differ = (llerr[cur]-llerr[cur-1])/(llerr[cur-1]-llerr[cur-2])
            else:
                differ=1
                
            cur=cur+1
        
        vari=np.zeros(res_P.shape[1])
        for k in range(res_P.shape[1]):
            w=np.argwhere(np.absolute(neighbors_map-k)<0.01)
            w=np.reshape(w,(len(w),))
            temp = Pl[:,w]-mb.repmat(np.reshape(res_P[:,k],(2,1)),1,len(w))
            vari[k]=np.std(np.diag(temp.T.dot(temp)))
            
        eps=1e-10
        w=np.argwhere(res_c<eps)
        # res_c=np.delete(res_c,w) # ? # ?
        var_out=vari
        # ?
        res_c=res_c/np.sum(res_c)
        res_P2=res_P.T
        res_c2=res_c

    return (res_P2,res_c2)

def particleApproximation_v0(imgs,Nmasses):
    Pl=list(); P=list()
    for i in range(imgs.shape[0]):
        (Pl_t,P_t)=img2pts_Lloyd(imgs[i],Nmasses)
        Pl.append(Pl_t); P.append(P_t);
    return (Pl,P)

def sub2ind(array_shape, rows, cols):
    return (cols-1)*array_shape[1] + rows-1

def Visualize_LOT(Data,Intensity,Nx,Ny,scale):
    NG=35
    I1=np.zeros((scale*Nx,scale*Ny))
    loc=scale*np.round(np.reshape(Data,(int(len(Data)/2),2),order='F'))

    linearind=sub2ind(np.shape(I1),loc[:,0],loc[:,1])
    linearind=linearind.astype(int)
    
    i1=np.squeeze(np.reshape(I1,(scale*Nx*scale*Ny,1),order='F'))
    i1[linearind]=Intensity
    I1=np.reshape(i1,(scale*Nx,scale*Ny),order='F')
        
    h1 = sc.signal.gaussian(NG*scale,std=7); h1=np.reshape(h1,(len(h1),1),order='F')
    h=h1.dot(h1.T); h=h/np.sum(h)
    
    I1=sc.ndimage.convolve(I1,h,mode='constant') 
    I1=I1-I1.min(); I1=I1/I1.max(); #I1=np.flipud(np.fliplr(I1))
    
    return I1

class batch_PLOT_v0:
    def __init__(self, Nmasses=50):
        self.Nmasses = Nmasses
        
    def forward_seq(self, x_train, x_test):
        N = self.Nmasses
        (Pl_train,P_train)=particleApproximation_v0(x_train, N)
        (Pl_test,P_test)=particleApproximation_v0(x_test,N)
        
        Pl_tem=0
        for a in range(2):#x_train.shape[0]):
            t=Pl_train[a]
            Pl_tem=Pl_tem+t
        Pl_tem=Pl_tem/2#x_train.shape[0]
        P_tem = np.ones((N,))/float(N)
        
        #Pl_tem_vec=np.reshape(Pl_tem,(Pl_tem.shape[0]*Pl_tem.shape[1],),order='F')
        
        V=list(); M=x_train.shape[0]
        for ind in range(M):
            Ni=Pl_train[ind].shape[0]
            C=ot.dist(Pl_train[ind],Pl_tem)
            b=P_tem # b=np.ones((N,))/float(N)
            a=P_train[ind] # a=np.ones((Ni,))/float(Ni)    
            p=ot.emd(a,b,C) # exact linear program
            
            #V.append(np.matmul((N*p).T,Pl_train[ind])-Pl_tem)
            V.append(np.matmul((N*p).T,Pl_train[ind])+Pl_tem) # already giving transport displacement?
            
        V=np.asarray(V)
        
        x_train_hat=np.zeros((len(V),V[0].shape[0]*V[0].shape[1]))
        for a in range(len(V)):
            x_train_hat[a,:]=np.reshape(V[a],(V[0].shape[0]*V[0].shape[1],),order='F')
            
        V=list(); M=x_test.shape[0]
        for ind in range(M):
            Ni=Pl_test[ind].shape[0]
            C=ot.dist(Pl_test[ind],Pl_tem)
            b=P_tem # b=np.ones((N,))/float(N)
            a=P_test[ind] # a=np.ones((Ni,))/float(Ni)
            p=ot.emd(a,b,C) # exact linear program
            
            #V.append(np.matmul((N*p).T,Pl_test[ind])-Pl_tem)
            V.append(np.matmul((N*p).T,Pl_test[ind])+Pl_tem)
            
        V=np.asarray(V)
        
        x_test_hat=np.zeros((len(V),V[0].shape[0]*V[0].shape[1]))
        for a in range(len(V)):
            x_test_hat[a,:]=np.reshape(V[a],(V[0].shape[0]*V[0].shape[1],),order='F')
            
        return x_train_hat, x_test_hat, Pl_tem, P_tem
