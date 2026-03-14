# these are the functions for TBM software
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from pytranskit.optrans.utils import plot_displacements2d
import seaborn as sns
from pytranskit.TBM3D.image_processing import gen_pdf
import warnings
# warnings.filterwarnings('ignore')

def visualize_result(result):
    '''
    Visualize LOT embedding results
    Input:
        result: a dict, including transport maps, MSE and other values.
    Output:
        None
    '''
    I0,I1,I0_recon = result["I0"],result["I1"],result["I0_recon"]
    f1, f2, f3 = result["f1"],result["f2"],result["f3"]
    X, Y, Z = np.meshgrid(
            np.arange(I0.shape[0]),
            np.arange(I0.shape[1]),
            np.arange(I0.shape[2]),
            indexing='ij'
        )
    displacements = np.stack((f2-Y,f1-X)) #* np.sqrt((I0))

    global_min = min(np.min(I0), np.min(I1), np.min(I0_recon))
    global_max = max(np.max(I0), np.max(I1), np.max(I0_recon))
    max_index = I0.shape[2]

    def plot_slice(i):
        plt.figure(figsize=(20, 5))
        im1, im2, im3 = I0[:, :, i], I1[:, :, i], I0_recon[:, :, i]
        im4 = displacements[:,:,:,i]

        plt.subplot(1, 4, 1)
        plt.imshow(im1, cmap='gray', vmin=global_min, vmax=global_max)  
        plt.title('I0', fontsize=18)
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(im2, cmap='gray', vmin=global_min, vmax=global_max)  
        plt.title('I1', fontsize=18)
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(im3, cmap='gray', vmin=global_min, vmax=global_max)  
        plt.title('$(D_{f}(\mathbf{x}))\'I_1( f(\mathbf{x}))$', fontsize=18)
        plt.axis('off')
        ax = plt.subplot(1, 4, 4)
        plot_displacements2d(im4,ax = ax)
        plt.title('displacements', fontsize=18)
        plt.show()
    slider = widgets.IntSlider(min=0, max=max_index-1, step=1, value=max_index//2, description='Slice')
    widgets.interact(plot_slice, i=slider)

    results = []
    for a in range(5):
        alpha = a/4
        f1_temp = (1-alpha)*X + alpha * f1
        f2_temp = (1-alpha)*Y + alpha * f2
        f3_temp = (1-alpha)*Z + alpha * f3

        f1x, f1y, f1z = np.gradient(f1_temp)
        f2x, f2y, f2z = np.gradient(f2_temp)
        f3x, f3y, f3z = np.gradient(f3_temp)

        detf = (
                f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
                - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
            )

        coordinates = np.array([f1.flatten(), f2.flatten(), f3.flatten()])
        It = map_coordinates(I1, coordinates, order=3, mode='constant', cval=np.min(I1), prefilter = True).reshape(f1.shape)
        It = np.abs(It)
        I0_recon = detf * It
        results.append(I0_recon)

    show_3d_arrays(np.asarray(results),titles=['α = 0.0','α = 0.25','α = 0.5','α = 0.75','α = 1.0'],cmap='gray')


def inpaint_nans3(volume):
    '''
    Remove NaN values from the image.
    Input:
        volume: 3D numpy array with NaN values
    Output:
        smoothed: 3D numpy array with NaN values inpainted
    '''
    mask = np.isnan(volume)
    volume_filled = volume.copy()
    volume_filled[mask] = 0
    smoothed = gaussian_filter(volume_filled, sigma=1)
    weight = gaussian_filter((~mask).astype(float), sigma=1)
    return smoothed / (weight + 1e-8)

def visual_roc(plda_proj_tr,train_labels,plda_proj_te,test_labels):
    '''
    Visualize the ROC curve of the prediction results.
    Input:
        plda_proj_tr: projection results of training data
        train_labels: labels of training data
        plda_proj_te: projection results of testing data
        test_labels: labels of testing data
    Output:
        auc: AUC value on testing data
    ''' 
    auc = roc_auc_score(train_labels, plda_proj_tr)
    if auc < 0.5:
        plda_proj_tr = -plda_proj_tr
        
    auc = roc_auc_score(train_labels, plda_proj_tr)
    fpr, tpr, thresholds = roc_curve(train_labels, plda_proj_tr)

    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Training data)')
    plt.legend()
    plt.grid(True)

    auc = roc_auc_score(test_labels, plda_proj_te)
    if auc < 0.5:
        plda_proj_te = -plda_proj_te

    auc = roc_auc_score(test_labels, plda_proj_te)
    fpr, tpr, thresholds = roc_curve(test_labels, plda_proj_te)

    plt.subplot(1,2,2)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Testing data)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc

def show_3d_arrays(arrs, axis=2, cmap='gray', stds = [-2,-1,0,1,2], titles = None, fontsize = 18):
    '''
    Visualize 3D arrays by slicing along a specified axis.
    Input:
        arrs: 4D numpy array with shape (n, d1, d2, d3)
        axis: int, axis along which to slice (0, 1, or 2)
        cmap: colormap for visualization
    Output:
        None
    '''

    max_index = arrs.shape[axis+1]

    global_min = np.min(arrs)
    global_max = np.max(arrs)

    ratio = arrs.shape[1]/arrs.shape[2]

    def plot_slice(i):
        plt.figure(figsize=(25,5*ratio))

        if axis == 0:
            ims = arrs[:, i, :, :]
        elif axis == 1:
            ims = arrs[:, :, i, :]
        elif axis == 2:
            ims = arrs[:, :, :, i]
        else:
            raise ValueError("axis must be 0, 1, or 2")

        for j in range(5):
            plt.subplot(1, 5, j+1)
            plt.imshow(ims[j,:,:], cmap=cmap, vmin=global_min, vmax=global_max, aspect='equal') 
            if titles is not None:
                if len(titles) == 5:
                    plt.title(titles[j], fontsize=fontsize*ratio)
                else:
                    if j == 2:
                        plt.title(titles[0], fontsize=fontsize*ratio)
            else: 
                plt.title(f'{stds[j]} \u03C3, Slice {i}', fontsize=fontsize*ratio)
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0)  
        plt.show()

    slider = widgets.IntSlider(min=0, max=max_index-1, step=1, value=max_index//2, description='Slice')
    widgets.interact(plot_slice, i=slider)

def inverse_image(X_inverse, reference):
    '''
    Convert the image from the transport domain back to the image domain.
    Input:
        X_inverse: numpy array with shape (n, d), transport maps
        reference: 3D numpy array, reference image
    Output:
        images: numpy array with shape (n, d1, d2, d3), reconstructed images
    '''
    reference = reference.copy()
    mask = np.zeros(reference.shape)
    mask[reference>np.min(reference)] = 1
    mask = binary_dilation(mask, iterations=2)
    reference[mask==0] = 0
    X_inverse_maps = X_inverse.reshape(tuple((X_inverse.shape[0],3))+reference.shape)
    X_inverse_maps = X_inverse_maps/np.sqrt(reference)
    X_inverse_maps = np.nan_to_num(X_inverse_maps, nan=0)
    X, Y, Z = np.meshgrid(
            np.arange(reference.shape[0]), np.arange(reference.shape[1]), np.arange(reference.shape[2]), indexing='ij'
        )
    X_inverse_maps[:,0,:,:,:] += X
    X_inverse_maps[:,1,:,:,:] += Y
    X_inverse_maps[:,2,:,:,:] += Z

    images = []
    for i in range(5):
        f1, f2, f3 = X_inverse_maps[i,0,:,:,:],X_inverse_maps[i,1,:,:,:],X_inverse_maps[i,2,:,:,:]
        f1x, f1y, f1z = np.gradient(f1)
        f2x, f2y, f2z = np.gradient(f2)
        f3x, f3y, f3z = np.gradient(f3)

        detf = (
            f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
            - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
        )

        points = np.vstack((f1.ravel(), f2.ravel(), f3.ravel())).T  # shape (N, 3)
        values = (reference / detf).ravel()

        grid_vals = griddata(points, values, (X, Y, Z), method='nearest')
        images.append(inpaint_nans3(grid_vals))

    images = np.asarray(images)
    images[images< np.min(reference)] = np.min(reference)
    images[images> np.max(reference)] = np.max(reference)
    # print(np.min(images),np.max(images))
    return images


def pad_to_square(arr, pad_value=0):
    '''
    Padding the image to make it a square.
    Input:
        arr: 2D numpy array
        pad_value: value for padding
    Output:
        arr_square: 2D numpy array, padded square array
    '''
    h, w = arr.shape
    size = max(h, w)  
    pad_h = size - h
    pad_w = size - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    arr_square = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=pad_value
    )
    return arr_square

def visualize_pca(X_train_pca,X_test_pca,train_labels,test_labels,reference, pca, modes = [0], task = 'cla', stds = [-2,-1,0,1,2]):
    '''
    Visualize the PCA results.
    Input:
        X_train_pca: numpy array with shape (n_train, d), PCA projection of training data
        X_test_pca: numpy array with shape (n_test, d), PCA projection of testing data
        train_labels: numpy array with shape (n_train,), labels of training data
        test_labels: numpy array with shape (n_test,), labels of testing data
        reference: 3D numpy array, reference image
        pca: PCA model
        modes: list of int, PCA modes to visualize
        task: str, 'cla' for classification, 'reg' for regression
    Output:
        outputs: list of 2D numpy arrays, images corresponding to the specified PCA modes
    '''
    x_min = min(X_train_pca[:,0].min(), X_test_pca[:,0].min())
    x_max = max(X_train_pca[:,0].max(), X_test_pca[:,0].max())
    y_min = min(X_train_pca[:,1].min(), X_test_pca[:,1].min())
    y_max = max(X_train_pca[:,1].max(), X_test_pca[:,1].max())
    x_min -= 0.1*(x_max-x_min)
    x_max += 0.1*(x_max-x_min)
    y_min -= 0.1*(y_max-y_min)
    y_max += 0.1*(y_max-y_min)

    data_std = np.std(X_train_pca[:,0])
    vec = np.zeros((5,X_train_pca.shape[1]))
    vec[:,0] = np.asarray([-2,-1,0,1,2])*data_std
    X_inverse = pca.inverse_transform(vec)
    img_x = inverse_image(X_inverse,reference)
    _,h,w,d = img_x.shape
    s = max(h, w)
    images_x = np.zeros((s,5*s))
    for i in range(5):
        images_x[:,i*s:(i+1)*s] = pad_to_square(img_x[i,:,:,d//2])

    data_std = np.std(X_train_pca[:,1])
    vec = np.zeros((5,X_train_pca.shape[1]))
    vec[:,1] = np.asarray([-2,-1,0,1,2])*data_std
    X_inverse = pca.inverse_transform(vec)
    img_y = inverse_image(X_inverse,reference)
    _,h,w,d = img_y.shape
    images_y = np.zeros((5*s,s))
    for i in range(5):
        images_y[i*s:(i+1)*s,:] = pad_to_square(img_y[i,:,:,d//2])

    plt.figure(figsize=(18,7))
    ax0=plt.subplot2grid((4, 10), (0, 1), colspan=3,rowspan=3)
    ax0.set_xlim((x_min, x_max))
    ax0.set_ylim((y_min, y_max))
    x, y = X_train_pca[:,0], X_train_pca[:,1]
    if task == 'cla':
        for label in np.unique(train_labels):
            ax0.scatter(x[train_labels == label], y[train_labels == label], label=f"Class {label}", alpha=0.7)
    else:
        ax0.scatter(x, y, alpha=0.7)

    ax1=plt.subplot2grid((4, 10), (3, 1), colspan=3,rowspan=1)
    ax2=plt.subplot2grid((4, 10), (0, 0), colspan=1,rowspan=3)
    ax1.imshow(images_x,cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(images_y,cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])


    ax0.set_xlabel("First PCA direction")
    plt.ylabel("Second PCA direction")
    ax0.legend()
    ax0.set_title("Projection of training data on the first 2 PCA directions")
    ax0.grid(linestyle='--')

    ax0=plt.subplot2grid((4, 10), (0, 6), colspan=3,rowspan=3)
    ax0.set_xlim((x_min, x_max))
    ax0.set_ylim((y_min, y_max))
    ax0.grid(linestyle='--')
    x, y = X_test_pca[:,0], X_test_pca[:,1]
    if task == 'cla':
        for label in np.unique(test_labels):
            ax0.scatter(x[test_labels == label], y[test_labels == label], label=f"Class {label}", alpha=0.7)
    else:
        ax0.scatter(x, y, alpha=0.7)

    ax0.set_xlabel("First PCA direction")
    plt.ylabel("Second PCA direction")
    ax0.legend()
    ax0.set_title("Projection of testing data on the first 2 PCA directions")
    ax0.grid(linestyle='--')

    ax1=plt.subplot2grid((4, 10), (3, 6), colspan=3,rowspan=1)
    ax2=plt.subplot2grid((4, 10), (0, 5), colspan=1,rowspan=3)
    ax1.imshow(images_x,cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(images_y,cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()

    outputs = []

    for mode in modes:
        if mode >= X_train_pca.shape[1]:
            print("Mode out of range!")
            return
        data_std = np.std(X_train_pca[:,mode])
        vec = np.zeros((5,X_train_pca.shape[1]))
        vec[:,mode] = np.asarray(stds)*data_std
        X_inverse = pca.inverse_transform(vec)
        images = inverse_image(X_inverse,reference)
        show_3d_arrays(images,stds=stds)
        outputs.append(images[:,:,:,images.shape[3]//2])

    return outputs

def inverse_visualiztion(plda_proj_tr,train_labels,reference,plda,pca):
    '''
    Visualize the PLDA results.
    Input:
        plda_proj_tr: projection results of training data
        train_labels: labels of training data
        reference: 3D numpy array, reference image
        plda: PLDA model
        pca: PCA model
    Output:
        images: numpy array with shape (5, d1, d2), images corresponding to the PLDA projection
    '''
    plt.figure(figsize=(12,6))
    for lab in np.unique(train_labels):
        sns.kdeplot(plda_proj_tr[train_labels == lab], label=f"Label {lab}", fill=True, alpha=0.3, lw=2)
    plt.legend()
    plt.title("Distribution")
    plt.xlabel("Projection score")
    plt.ylabel("Density")
    plt.show()
    data_std = np.std(plda_proj_tr)
    vec = np.zeros((5,plda.components_.shape[0]))
    vec[:,0] = np.asarray([-2,-1,0,1,2])*data_std
    inverse_pca = plda.inverse_transform(vec)
    X_inverse = pca.inverse_transform(inverse_pca)
    images = inverse_image(X_inverse, reference)
    show_3d_arrays(images)
    return images[:,:,:,images.shape[3]//2]


def intrinsic_mean_ns(mean_map, reference):
    '''
    Calculate the intrinsic mean of the image.
    Input:
        mean_map: numpy array, mean transport maps
        reference: 3D numpy array, reference image
    Output:
        I0_recon: 3D numpy array, intrinsic mean image
    '''
    reference = reference.copy()
    reference = reference - np.min(reference)

    X_inverse_maps = mean_map.reshape((3,)+reference.shape)
    X, Y, Z = np.meshgrid(
                np.arange(reference.shape[0]), np.arange(reference.shape[1]), np.arange(reference.shape[2]), indexing='ij'
            )

    f1, f2, f3 = X_inverse_maps[0,:,:,:],X_inverse_maps[1,:,:,:],X_inverse_maps[2,:,:,:]

    f1x, f1y, f1z = np.gradient(f1)
    f2x, f2y, f2z = np.gradient(f2)
    f3x, f3y, f3z = np.gradient(f3)

    detf = (
            f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
            - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
        )

    points = np.vstack((f1.ravel(), f2.ravel(), f3.ravel())).T  # shape (N, 3)
    values = (reference / detf).ravel()

    grid_vals = griddata(points, values, (X, Y, Z), method='nearest')

    I0_recon = inpaint_nans3(grid_vals)
    I0_recon[I0_recon<np.min(reference)] = np.min(reference)
    I0_recon[I0_recon>np.max(reference)] = np.max(reference)

    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(I0_recon[:,:,I0_recon.shape[2]//2],cmap='gray')
    plt.title('Intrinsic mean', fontsize=18)

    plt.subplot(1,2,2)
    plt.title('Image mean', fontsize=18)
    plt.axis('off')
    plt.imshow(reference[:,:,reference.shape[2]//2],cmap='gray')
    plt.show()

    return I0_recon

if __name__ == '__main__':
    X_inverse_maps = np.load('./testfolder/inverse_maps.npy')
    reference = np.load('./data/template.npy')
    X, Y, Z = np.meshgrid(
        np.arange(reference.shape[0]), np.arange(reference.shape[1]), np.arange(reference.shape[2]), indexing='ij'
    )
    images = []
    for i in range(5):
        f1, f2, f3 = X_inverse_maps[i,0,:,:,:],X_inverse_maps[i,1,:,:,:],X_inverse_maps[i,2,:,:,:]
        f1x, f1y, f1z = np.gradient(f1)
        f2x, f2y, f2z = np.gradient(f2)
        f3x, f3y, f3z = np.gradient(f3)

        detf = (
            f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
            - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
        )

        points = np.vstack((f1.ravel(), f2.ravel(), f3.ravel())).T  # shape (N, 3)
        values = (reference / detf).ravel()
        
        grid_vals = griddata(points, values, (X, Y, Z), method='linear')
        print(grid_vals.shape)
        images.append(inpaint_nans3(grid_vals))

    images = np.asarray(images)
    print(images.shape)