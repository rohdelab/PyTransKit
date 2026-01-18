# these are the functions for TBM software
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.

import numpy as np
from pytranskit.TBM3D.VOT3D import VOT3D, update_result
# from VOT3D_test import VOT3D, update_result
from pytranskit.TBM3D.image_processing import GPExpand, GPReduce, gaussian_bf
import time
from pytranskit.TBM3D.gradient import compVOTGradients
from scipy.ndimage import convolve as convn
from scipy.ndimage import zoom


def multVOT(I0,I1,lambda_=50,tot=1e6,sigma=1,numScales=4,level=0.25,gamma=0.1, dc = 0.1):
    '''
    Running TBM embedding for an image
    Input:
        I1: image to embed
        I0: reference of transport maps
        lambda_,tot,sigma,numScales,level,gamma: see [1]
    Output:
        results: a dict, including transport maps, MSE and other values.
    '''
    _,_,K = I1.shape
    sx,sy,sz = I0.shape

    globalIter = 1

    tic = time.time()

    for scale in range(numScales, -1, -1):
        print(f"Now starting scale {scale}")

        I0_down = I0.copy()
        I1_down = I1.copy()

        if scale != 0:
            for i in range(1, scale + 1):
                # meshgrid in Python: order is (y,x,z)
                X_down1, _, _ = np.meshgrid(
                    np.arange(0, sx, 2**(i - 1)),
                    np.arange(0, sy, 2**(i - 1)),
                    np.arange(0, sz, 2**(i - 1)),
                    indexing='ij'
                )
                I0_down = GPReduce(I0_down)
                I1_down = GPReduce(I1_down)
                newdim = X_down1.shape

        # Create meshgrid for I0_down
        X, Y, Z = np.meshgrid(
            np.arange(I0_down.shape[0]),
            np.arange(I0_down.shape[1]),
            np.arange(I0_down.shape[2]),
            indexing='ij'
        )

        if globalIter == 1:
            f0 = X.copy()
            g0 = Y.copy()
            h0 = Z.copy()
        
        results = VOT3D(I0_down, I1_down, f0, g0, h0,scale = scale, penalty= lambda_, tot=tot, sigma=sigma, level=level,  gamma = gamma, dc = dc)
        # if globalIter > 1:
        #     return

        if globalIter == numScales + 1:
            t = time.time() - tic
            final_results = results
            final_results['time'] = t

            print(f"the final curl is {results['curl'][results['iter']-1]}")
            print(f"the final MSE overall is {final_results['MSE3'][results['iter']-1]}")
            print(f"the final MSE in the tissue is {final_results['MSE1'][results['iter']-1]}")
            print(f"the final time it took was {t / 60} minutes")
            break
        else:
            # Interpolation preparation
            X2, Y2, Z2 = np.meshgrid(
                np.arange(X_down1.shape[0]),
                np.arange(X_down1.shape[1]),
                np.arange(X_down1.shape[2]),
                indexing='ij'
            )

            f0 = 2 * GPExpand(results['f1'] - X, newdim) + X2
            g0 = 2 * GPExpand(results['f2'] - Y, newdim) + Y2
            h0 = 2 * GPExpand(results['f3'] - Z, newdim) + Z2

            _, _, _, _, _, flag = compVOTGradients(f0, g0, h0,
                                                np.zeros_like(f0),
                                                np.zeros_like(f0),
                                                0, gamma)

            sigma_f = 2
            if flag:
                # Smooth the field if not diffeomorphic
                r = np.arange(-3 * sigma_f, 3 * sigma_f + 1)
                Xt, Yt, Zt = np.meshgrid(r, r, r, indexing='ij')
                phi = gaussian_bf(Xt, Yt, Zt, sigma_f)

                f0 = 2 * GPExpand(convn(results['f1'] - X, phi), newdim) + X2
                g0 = 2 * GPExpand(convn(results['f2'] - Y, phi), newdim) + Y2
                h0 = 2 * GPExpand(convn(results['f3'] - Z, phi), newdim) + Z2

                _, _, _, _, _, flag = compVOTGradients(f0, g0, h0,
                                                np.zeros_like(f0),
                                                np.zeros_like(f0),
                                                0, gamma)
                if flag:
                    def upsample_to_shape(array, target_shape):
                        factors = [t / s for t, s in zip(target_shape, array.shape)]
                        upsampled = zoom(array, zoom=factors, order=1)  
                        return upsampled
                    
                    f0 = 2 * upsample_to_shape(results['f1'] - X, newdim) + X2
                    g0 = 2 * upsample_to_shape(results['f2'] - Y, newdim) + Y2
                    h0 = 2 * upsample_to_shape(results['f3'] - Z, newdim) + Z2
                    print(f0[:10,0,0])
                    print(X2[:10,0,0])
                    print(results['f1'][:10,0,0])

                    _, _, _, _, _, flag = compVOTGradients(f0, g0, h0,
                                                np.zeros_like(f0),
                                                np.zeros_like(f0),
                                                0, gamma)
                    
                    if not flag:
                        print("now it is diffeomorphic!!")
                    else:
                        print("oh no!!!")

            globalIter += 1

    try:
        return update_result(final_results,final_results['iter'])
    except:
        return final_results



if __name__ == '__main__':
    shape = (63,64,65)
    f1,f2,f3,I0 = np.ones(shape),np.ones(shape),np.ones(shape),np.zeros(shape)
    for i in range(shape[0]):
        f1[i,:,:] = i
    for i in range(shape[1]):
        f2[:,i,:] = i
    for i in range(shape[2]):
        f3[:,:,i] = i
        
    I1 = np.zeros(shape)
    I1[1,1,1] = 1
    I0[-2,-2,-2] = 1
    I1 = I1 + 1e-5
    I0 = I0 + 1e-5
    I0 = I0/np.sum(I0)
    I1 = I1/np.sum(I1)

    # print(I1)

    # result = multVOT(I0,I1)
    # print(result)

    import pickle
    from scipy.ndimage import gaussian_gradient_magnitude, map_coordinates


    # 保存
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(result, f)

    # with open('./testfolder/data.pkl', 'rb') as f:
    with open('data.pkl', 'rb') as f:
        my_dict = pickle.load(f)

    # print(my_dict['iter'])

    # # my_dict = update_result(my_dict,my_dict['iter'])

    # print(my_dict['MSE3'].shape)

    # print(my_dict['I0_recon'][-2,-2,-2])
    # print(my_dict['I1'][1,1,1])
    # print(my_dict['I0'][-2,-2,-2])

    f1,f2,f3 = my_dict['f1'],my_dict['f2'],my_dict['f3']

    # f1 = f1 +1
    # f2 = f2 +1
    # f3 = f3 +1

    f1x, f1y, f1z = np.gradient(f1)
    f2x, f2y, f2z = np.gradient(f2)
    f3x, f3y, f3z = np.gradient(f3)

    # Second-order gradients
    f1yx, f1yy, _ = np.gradient(f1y)
    f1zx, _, f1zz = np.gradient(f1z)
    f2xx, f2xy, _ = np.gradient(f2x)
    _, f2zy, f2zz = np.gradient(f2z)
    f3xx, _, f3xz = np.gradient(f3x)
    _, f3yy, f3yz = np.gradient(f3y)

    # Determinant of Jacobian
    detf = (
        f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
        - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
    )

    coordinates = np.array([f1.flatten(), f2.flatten(), f3.flatten()])
    It = map_coordinates(my_dict['I1'], coordinates, order=3, mode='constant', cval=np.min(my_dict['I1']), prefilter = True).reshape(f1.shape)
    # It = map_coordinates(I1, coordinates, order=3, mode='constant', cval=np.min(I1), prefilter = True).reshape(f1.shape)
    # print(It.shape)
    # Taking absolute value
    It = np.abs(It)
    # Ierror = detf * It - I0
    I0_recon = detf * It
    # It = It / np.sum(It) * 10**6
    I0_recon = I0_recon/np.sum(I0_recon) * 10**6

    # print(I0[-2,-2,-2])
    # print(my_dict['f1'][-2,-2,-2])
    # print(my_dict['f2'][-2,-2,-2])
    # print(my_dict['f3'][-2,-2,-2])
    print(f1[-2,-2,-2],f2[-2,-2,-2],f3[-2,-2,-2])
    # print(I0_recon[-3:,-3:,-3:])
    # print(I0[-3:,-3:,-3:])

    print(I0_recon[:3,:3,:3])
    print(my_dict['I0_recon'][:3,:3,:3])