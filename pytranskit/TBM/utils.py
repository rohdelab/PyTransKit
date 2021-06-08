#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 00:59:00 2020

@author: Imaging and Data Science Lab
"""

import numpy as np
from scipy.io import loadmat
import os
from skimage.io import imread
import glob
import matplotlib.pyplot as plt



def load_image_data(data_dir):
    types = ('*.png','*.tiff','*.bmp','*.jpg','*.jpeg','*.mat')
    class_dir = [os.path.join(data_dir,dI) for dI in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,dI))]
    
    x_data, y_data = [], []
    
    for cl in range(len(class_dir)):
        for ext in types:   
            files_read = []
            file_path = os.path.join(class_dir[cl], ext)
            files_read.extend(glob.glob(file_path))
            
            if files_read != []:
                for files in files_read:
                    if ext == '*.mat':
                        x_data.append(loadmat(files)['image'])
                    else:
                        x_data.append(imread(files))
                    y_data.append(cl)
    if x_data == []:
        print('Wrong file format provided. File types supported: ')
        print(types)
        print('\n In case of mat files, variable name needs to be \'image\'.')
        return (None, None)
    else:   
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return (x_data, y_data)

        
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix (%)',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    #plt.colorbar()

#    if target_names is not None:
#        tick_marks = np.arange(len(target_names))
#        plt.xticks(tick_marks, target_names, rotation=45)
#        plt.yticks(tick_marks, target_names)
#
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#
    #thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = (cm.max()+cm.min())/2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]*100.),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()
    plt.show()        
        
        
        
        