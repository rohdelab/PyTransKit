# these are the functions for TBM software
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.


import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from skimage import exposure
import pandas as pd

def GP_ker():
    # Convolutional Kernel
    Wt = np.asarray([0.05,0.25,0.4,0.25,0.05])
    Wt3 = np.ones((5, 5, 5))
    for i in range(5):
        Wt3[i, :, :] *= Wt[i]
        Wt3[:, i, :] *= Wt[i]
        Wt3[:, :, i] *= Wt[i]
    return Wt3

def GPReduce(I):
    # downscale image I by factor 2
    # input: 3D image
    # output: reduces image
    Wt3 = GP_ker()

    # padding
    I_padded = np.pad(I, pad_width=((2, 2), (2, 2), (2, 2)), mode='edge')

    IResult = convolve(I_padded, Wt3, mode='mirror')[2:-2,2:-2,2:-2]

    # return convolution results, downscale by factor 2
    return convolve(I_padded, Wt3, mode='mirror')[2:-2,2:-2,2:-2][::2, ::2, ::2]

def GPExpand(I, output_shape):
    # expend image I to output_shape, output_shape <= 2* I.shape
    # input: 
    #   I: 3D image
    #   output_shape: the shape of output image
    # output: expended image with output_shape
    Wt3 = GP_ker()

    # padding
    I_padded = np.pad(I, pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')

    IResult = np.zeros(output_shape)
    m = np.arange(-2, 3)

    # This part is different from traditional convolution, so it can only be implemented using loops
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                pixeli = (i - m) / 2 + 2 -1
                pixelj = (j - m) / 2 + 2 -1
                pixelk = (k - m) / 2 + 2 -1

                idxi = np.where(np.floor(pixeli) == pixeli)
                idxj = np.where(np.floor(pixelj) == pixelj)
                idxk = np.where(np.floor(pixelk) == pixelk)

                A = I_padded[np.ix_(pixeli[idxi[0]].astype(np.uint8), pixelj[idxj[0]].astype(np.uint8), pixelk[idxk[0]].astype(np.uint8))] * \
                    Wt3[np.ix_(m[idxi[0]] + 2, m[idxj[0]] + 2, m[idxk[0]] + 2)]

                IResult[i, j, k] = 8 * np.sum(A)
    return  IResult


def gen_pdf(I_in, sigma, dc = 0.1):
    #Preprocessing of 3D image
    #Inputs:     I_in       3D input image
    #            dc         a positive constant
    #            sigma      sigma for gaussion filter
    #
    #Outputs:    I_out      3D output image
    if sigma != 0:
        I_smoothed = gaussian_filter(I_in, sigma=sigma, mode = 'nearest', radius= 2 * np.ceil(2 * sigma).astype(int) + 1)
        I_normalized = exposure.rescale_intensity(I_smoothed, out_range=(0, 1))
    else:
        I_normalized = exposure.rescale_intensity(I_in, out_range=(0, 1))
    
    I = I_normalized + dc
    return I/np.sum(I)


def gaussian_bf(X,Y,Z,sigma):

    phi = (1 / (2 * np.pi * sigma**2)) * np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    phi /= np.sum(phi)  
    return phi

def load_data(file_path, reference_path = None, normalized = False, mode = 'train', epsilon = 1e-5):
    '''
    input:
        file_path: the path of your csv file
        reference_path: the path of your reference, if None, the reference will be the average of all training data
        normalized: if your data is already normalized, if False, all the data and reference will be normailized
        mode: 'train' or 'test', only 'train' mode will return the reference

    output:
        images: list of images with shape [h,w,d]
        labels: list of labels
        reference: numpy array with shape [h,w,d]
    '''
    df = pd.read_csv(file_path)
    filelist = df['image_path'].tolist()
    labels = df['label'].tolist()

    images = []
    for file in filelist:
        images.append(np.load(file))

    if reference_path:
        reference = np.load(reference_path)
    else:
        reference = np.mean(np.asarray(images),axis=0)

    if not normalized:
        for i in range(len(images)):
            image = images[i]
            images[i] = image/np.sum(image) + epsilon

        reference = reference/np.sum(reference) + epsilon

    if mode == 'train' :
        return images, labels, reference
    else:
        return images, labels



if __name__ == '__main__':

    images, labels, reference = load_data('./data/train.cvs', reference_path = './data/template.npy', normalized = True, mode = 'train')
    print(reference[0,0,0])