# these are the functions for TBM software
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.

import numpy as np
from pytranskit.TBM3D.multVOT import multVOT
import pickle
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_tbm(image, label, reference, output_path = None, numScales=4, dc = 0.1):
    '''
    Running TBM embedding for an image
    Input:
        image: image to embed
        label: labels of inpuy image
        reference: reference of transport maps
        output_path: the output path to save the results, if it is None, no results will be saved
    Output:
        results: a dict, including transport maps, MSE and other values.
    '''
    results = multVOT(reference,image, numScales=numScales, dc = dc)
    results['label'] = label
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    return results

def process_features(features, reference):
    '''
    Convert features to numpy for PCA or other ML methods
    Input:
        features: a list, each element of the list corresponds to the calculation result of each image, including transport maps, MSE and other values.
        reference: reference of transport maps(normalized)
    Output:
        output_features: numpy array with shape n*d
        output_labels: numpy array with shape n*1
    '''
    output_features = []
    output_labels = []

    reference = reference.copy()

    mask = np.zeros(reference.shape)
    mask[reference>np.min(reference)] = 1
    mask = binary_dilation(mask, iterations=2)
    reference[mask==0] = 0

    X, Y, Z = np.meshgrid(
            np.arange(reference.shape[0]),
            np.arange(reference.shape[1]),
            np.arange(reference.shape[2]),
            indexing='ij'
        )

    for result in features:
        f1 = result['f1']
        f2 = result['f2']
        f3 = result['f3']
        f1 = (f1-X)*np.sqrt(reference)
        f2 = (f2-Y)*np.sqrt(reference)
        f3 = (f3-Z)*np.sqrt(reference)
        features = np.concat([f1,f2,f3],axis=0).flatten()
        output_features.append(features)
        output_labels.append(result['label'])

    return np.asarray(output_features), np.asarray(output_labels)

def read_features(data_folder):
    '''
    Read tranports maps from the folder
    Input:
        data_folder: path to transport maps
    Output:
        features: a list, each element of the list corresponds to the calculation result of each image, including transport maps, MSE and other values.
    '''
    features = []
    for i in range(len(os.listdir(data_folder))):
        # print(i)
        with open(os.path.join(data_folder,str(i)+'.pkl'), 'rb') as f:
            result = pickle.load(f)
        features.append(result)
    return features

def TBM3D_forward(images, labels, reference, output_folder = None, parallel = True, num_workers = 12, numScales=4, b = 0, **kwargs):
    '''
    Running TBM embedding
    Input:
        images: list of images
        labels: labels for each images with corresponding order
        reference: reference of transport maps
        output_folder: the output path if you want to save the maps, if it is None, no maps will be saved
        parallel: True or False, if you parallel computing
        num_workers: number of cores will be used for parallel computing
    Output:
        features: a list, each element of the list corresponds to the calculation result of each image, including transport maps, MSE and other values.
    '''
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if parallel:
            tasks = []
            for i in range(len(images)):
                tasks.append((images[i],labels[i],reference, os.path.join(output_folder,str(i+b)+'.pkl')))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_tbm, image, label,reference, out_path, numScales, **kwargs) for image, label, reference, out_path in tasks]
            features = read_features(output_folder)
        else:
            features = []
            for i in range(len(images)):
                features.append(run_tbm(images[i],labels[i],reference, os.path.join(output_folder,str(i+b)+'.pkl'), numScales, **kwargs))

    else:
        if parallel:
            tasks = []
            for i in range(len(images)):
                tasks.append((images[i],labels[i],reference))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_tbm, image, label,reference, None, numScales, **kwargs) for image, label, reference in tasks]
                features = [f.result() for f in futures]
        else:
            features = []
            for i in range(len(images)):
                features.append(run_tbm(images[i],labels[i],reference, None, numScales, **kwargs))

    return features

if __name__ == '__main__':
    from image_processing import load_data
    train_file_path = './data/train.csv'
    test_file_path = './data/test.csv'
    train_maps_folder = './data/train_maps'
    test_maps_folder = './data/test_maps'
    reference_path = None
    normalized = False
    train_images, train_labels, reference = load_data(file_path=train_file_path, reference_path=reference_path, normalized=normalized, mode = 'train',epsilon = 1e-5)
    TBM3D_forward(train_images, train_labels, reference, output_folder = train_maps_folder, parallel = True, num_workers = 12, b=0)
    test_images, test_labels = load_data(file_path=test_file_path, normalized=normalized, mode = 'test')
    TBM3D_forward(test_images, test_labels, reference, output_folder = test_maps_folder, parallel = True, num_workers = 12)
