#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 00:37:37 2020

Imaging and Data Science Lab
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import rotate
from sklearn.decomposition.pca import PCA

from skimage.filters import threshold_otsu

def image_preprocessing(im_array,flag_tran=1,flag_scale=1,flag_orient=1,flag_group=1):
    if flag_tran:
        im_array = im_center(im_array)
    if flag_scale:
        im_array = im_scale(im_array)
    if flag_orient:
        im_array = im_align(im_array)
    if flag_group:
        im_array = im_flip(im_array)
    return im_array

def im_center(im_array):
    # Center and crop to eliminate the translation
    [nz,ny,nx] = im_array.shape
    img_array_centered = np.zeros(im_array.shape)
    mx = np.round(nx/2);
    my = np.round(ny/2);
    
    for k in range(nz):
        img = im_array[k,:]
        [cx2,cy2] = ndimage.measurements.center_of_mass(img)
        trans = {'theta':0.}
        trans['scx'] = 1.
        trans['scy'] = 1.
        trans['sy'] = 0.
        trans['tx'] = np.round(cx2 - mx);
        trans['ty'] = np.round(cy2 - my);
        trans['cx'] = 0.
        trans['cy'] = 0.
        result_b = apply_trans2d(img,trans,1)
        res = result_b[np.uint8(my-np.round(ny/2)): np.uint8(my+np.floor(ny/2)+1.),
                       np.uint8(mx-np.round(nx/2)): np.uint8(mx+np.floor(ny/2)+1.)]
        img_array_centered[k,:,:] = res
    return img_array_centered

def im_align(im_array):
    # Aligning each image by rotating the image such that the major axis is verticle 
    # (Using PCA method).
    # Images are centralized first.
    #im_array_center = im_center(im_array)
    im_array_center = im_array
    [nz,ny,nx] = im_array_center.shape
    img_array_aligned = np.zeros(im_array_center.shape)
    
    for k in range(nz):
        img = im_array_center[k,:]
        result_b = verticalize_img(img)
        mx = np.round(result_b.shape[1]/2);
        my = np.round(result_b.shape[1]/2);
        img_array_aligned[k,:,:] = result_b[np.uint8(my-np.round(ny/2)): 
            np.uint8(my+np.floor(ny/2)),np.uint8(mx-np.round(nx/2)): 
                np.uint8(mx+np.floor(ny/2))]    
    return img_array_aligned

def im_scale(im_array):
    pa = np.sum(im_array[0])
    trans = {'theta':0.}
    trans['tx'] = 0.
    trans['ty'] = 0.
    trans['scx'] = 1.
    trans['scy'] = 1.
    trans['sy'] = 0.
    [nz,ny,nx] = im_array.shape
    NDS = 300
    ds = np.linspace(0,0.5,NDS);
    dsp = np.linspace(0,1,NDS);
    trans['cx'] = nx/2
    trans['cy'] = ny/2
    
    img_array_scaled=np.zeros(np.shape(im_array))
    
    for k in range(nz):
        img=im_array[k]
        ca=np.sum(img)
        if ca<pa:
            minv = np.abs(ca-pa); minl = 0
            for m in range(NDS):
                trans['scx'] = 1-ds[m]
                trans['scy'] = 1-ds[m]
                result_b = apply_trans2d(img,trans,1)
                cv = np.sum(result_b)
                if np.abs(cv-pa)<minv:
                    minv = np.abs(cv-pa)
                    minl = ds[m]
            trans['scx'] = 1-minl
            trans['scy'] = 1-minl
            result_b = apply_trans2d(img,trans,1)
            img_array_scaled[k,:,:]=result_b
        else:
            minv = np.abs(ca-pa); minl = 0;
            for m in range(NDS):
                trans['scx'] = 1+ds[m]
                trans['scy'] = 1+ds[m]
                result_b = apply_trans2d(img,trans,1)
                cv = np.sum(result_b)
                if np.abs(cv-pa)<minv:
                    minv = np.abs(cv-pa)
                    minl = dsp[m]
            trans['scx'] = 1+minl
            trans['scy'] = 1+minl
            result_b = apply_trans2d(img,trans,1)
            img_array_scaled[k,:,:]=result_b
    
    return img_array_scaled

def im_flip(im_array):
    Num_it = 10
    [nz,ny,nx] = im_array.shape
    
    trans = {'theta':0.}
    trans['tx'] = 0.
    trans['ty'] = 0.
    trans['scx'] = 1.
    trans['scy'] = 1.
    trans['sy'] = 0.
    trans['cx'] = nx/2
    trans['cy'] = ny/2
    
    img_array_flipped=np.copy(im_array) # np.zeros(np.shape(im_array))
    
    for k in range(Num_it):
        for i in range(nz):
            mI=np.mean(img_array_flipped,axis=0)
            img = img_array_flipped[i]
            
            imgc = img
            imgfr = np.fliplr(img)
            imgfd = np.flipud(img)
            imgfrd = np.fliplr(imgfd)
            
            ac=np.sum((img-mI)*(img-mI))
            
            acr=np.sum((imgfr-mI)*(imgfr-mI))
            if acr < ac:
                imgc = imgfr
                
            acd=np.sum((imgfd-mI)*(imgfd-mI))
            if acd < ac:
                imgc = imgfd
                
            acdr=np.sum((imgfrd-mI)*(imgfrd-mI))
            if acdr < ac:
                imgc = imgfrd
                
            img_array_flipped[i,:,:] = imgc;
    return img_array_flipped

def verticalize_img(img):
    """
    Method to rotate a greyscale image based on its principal axis.

    :param img: Two dimensional array-like object, values > 0 being interpreted as containing to a line
    :return rotated_img: 
    """# Get the coordinates of the points of interest:
    X = np.array(np.where(img > threshold_otsu(img)*0.8)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    angle = np.arctan2(*pca.components_[0])
    # Rotate the image by the computed angle:
    rotated_img = rotate(img,angle/np.pi*180-90,reshape=True)
    return rotated_img

def build_trans2d(M,N,transf):
    Xt, Yt = np.meshgrid(range(N), range(M), sparse=False, indexing='ij')
    
    a = transf['theta']
    R = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    S = np.array([[transf['scx'],transf['sy']],[0,transf['scy']]])
    M2 = S@R
    
    cx = transf['cx']
    cy = transf['cy']
    
    Xt = Xt-cx
    Yt = Yt-cy
    
    X = M2[0,0]*Xt + M2[0,1]*Yt + transf['tx'] + cx
    Y = M2[1,0]*Xt + M2[1,1]*Yt + transf['ty'] + cy
    return X,Y


def apply_trans2d(img,trans,degree=1):
    [M,N] = img.shape
    [X,Y] = build_trans2d(M,N,trans)
    
    if degree == 0:
        result = ndimage.map_coordinates(img,[X.ravel(),Y.ravel()], 
                                        order=0, mode='nearest').reshape(img.shape)
    else: 
        result = ndimage.map_coordinates(img,[X.ravel(),Y.ravel()], 
                                        order=degree, mode='nearest').reshape(img.shape)
    return result
