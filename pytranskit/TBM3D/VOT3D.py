# these are the functions for TBM software
# citations will be updated later
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.

import numpy as np
from pytranskit.TBM3D.image_processing import gen_pdf
from pytranskit.TBM3D.gradient import compVOTGradients
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def update_result(results,iter):
    '''
    Update results dict to have length iter for MSE and other values.
    Input:
        results: a dict, including transport maps, MSE and other values.
        iter: the new length of MSE and other values.
    Output:
        results: updated dict.
    '''
    new_err1, new_err2, new_err3, new_mass, new_errorcurl = np.zeros((iter)),np.zeros((iter)),np.zeros((iter)),np.zeros((iter)),np.zeros((iter))
    new_err1[:iter] = results['MSE1'][:iter]
    new_err2[:iter] = results['MSE2'][:iter]
    new_err3[:iter] = results['MSE3'][:iter]
    # new_err4[:iter] = results['err4'][:iter]
    new_mass[:iter] = results['mass'][:iter]
    new_errorcurl[:iter] = results['curl'][:iter]
    results['MSE1'], results['MSE2'], results['MSE3'], results['mass'], results['curl'] = new_err1, new_err2, new_err3, new_mass, new_errorcurl 
    return results

def curl(f1,f2,f3):
    # Compute gradients
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

    curlC1 = f2xy - f1yy - f1zz + f3xz
    curlC2 = f3yz - f2zz - f2xx + f1yx
    curlC3 = f1zx - f3xx - f3yy + f2zy

    return curlC1, curlC2, curlC3

def VOT3D(I0,I1,f1,f2,f3,scale,penalty=50,tot=1e6,sigma=1,level=0.25,gamma = 0.1, dc = 0.1):
    '''
    Running TBM embedding for an image
    Input:
        I1: image to embed
        I0: reference of transport maps
        f1,f2,f3: the x, y, and z values of maps 
        scale,penalty,tot,sigma,level,gamma: see [1]
    Output:
        results: a dict, including transport maps, MSE and other values.
    '''
    cutoff = 1e-4
    cutoff0 = 2e-4
    it = 0
    lambda_ = 0
    p = 1

    max_iters = 2000

    I0 = gen_pdf(I0,sigma,dc=dc) 
    I1 = gen_pdf(I1,sigma,dc=dc)

    M, N, K = I1.shape
    X, Y, Z = np.meshgrid(np.arange(M), np.arange(N), np.arange(K), indexing='ij')

    mask = np.ones_like(I0)
    for i in range(I0.shape[2]):
        mask[:, :, i] = I0[:, :, i] > np.min(I0)

    I0 = I0 * tot
    I1 = I1 * tot

    results = {
        "f1": f1,
        "f2": f2,
        "f3": f3
    }

    C1, C2, C3 = curl(f1, f2, f3)
    C = np.mean(C1**2 + C2**2 + C3**2)
    results["curl"] = C

    iter = 1
    converged = False

    err1, err2, err3, err4, mass, errorcurl = np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters))

    while True:
        if not p:
            print(f"Now on iteration {iter}")      

        if iter == 1:
            f1t, f2t, f3t, I0_recon, Ierror, flag = compVOTGradients(f1, f2, f3, I0, I1, lambda_, gamma)
            err3[iter-1] = (np.mean((Ierror / I0)**2))
            results["MSE3"] = err3
            results["mass"] = np.sum(((f1 - X)**2 + (f2 - Y)**2 + (f3 - Z)**2) * I0)


            err1[iter-1]=((0.5 * np.sum(((Ierror / I0) * mask)**2) / np.count_nonzero(mask)))
            results["MSE1"] = err1

            results.update({
                "I0_recon": I0_recon,
                "I0": I0,
                "I1": I1
            })    

            scale_factor = -(scale + 2 if scale == 0 or scale > 2 else scale + 1)
            step_size = (10**scale_factor) / np.max(np.sqrt((f1**2) + (f2**2) + (f3**2)))

            if flag:
                raise ValueError("The initial deformation field is not diffeomorphic")
            else:
                xk1_temp = f1 - step_size * f1t
                xk2_temp = f2 - step_size * f2t
                xk3_temp = f3 - step_size * f3t

                yk1_temp, yk2_temp, yk3_temp = xk1_temp, xk2_temp, xk3_temp
                _, _, _, _, _, flag = compVOTGradients(yk1_temp, yk2_temp, yk3_temp, I0, I1, lambda_, gamma)

                while flag and not converged:
                    step_size /= 2
                    if step_size < 1e-8:
                        converged = True
                        step_size = 0
                        results.update({"f1": f1, "f2": f2, "f3": f3})

                    xk1_temp = f1 - step_size * f1t
                    xk2_temp = f2 - step_size * f2t
                    xk3_temp = f3 - step_size * f3t

                    yk1_temp, yk2_temp, yk3_temp = xk1_temp, xk2_temp, xk3_temp
                    _, _, _, _, _, flag = compVOTGradients(yk1_temp, yk2_temp, yk3_temp, I0, I1, lambda_, gamma)

                xk1, xk2, xk3 = xk1_temp, xk2_temp, xk3_temp
                yk1, yk2, yk3 = yk1_temp, yk2_temp, yk3_temp

                yk1minus1, yk2minus1, yk3minus1 = [np.zeros_like(I0) for _ in range(3)]
                xk1minus1, xk2minus1, xk3minus1 = [np.zeros_like(I0) for _ in range(3)]
        
        if iter > 1:
            f1t, f2t, f3t, I0_recon, Ierror,_ = compVOTGradients(yk1minus1, yk2minus1, yk3minus1, I0, I1, lambda_, gamma)
            xk1_temp = yk1minus1 - step_size * f1t
            xk2_temp = yk2minus1 - step_size * f2t
            xk3_temp = yk3minus1 - step_size * f3t

            yk1_temp = xk1_temp + (iter - 2) / (iter + 1) * (xk1_temp - xk1minus1)
            yk2_temp = xk2_temp + (iter - 2) / (iter + 1) * (xk2_temp - xk2minus1)
            yk3_temp = xk3_temp + (iter - 2) / (iter + 1) * (xk3_temp - xk3minus1)

            if scale > 2:
                step_size = 10**(-(scale + 2.5)) / np.max(np.sqrt(f1**2 + f2**2 + f3**2))
            elif scale == 0:
                step_size = 10**(-(scale + 1.5)) / np.max(np.sqrt(f1**2 + f2**2 + f3**2))
            elif scale < 2:
                step_size = 10**(-(scale + 2.5)) / np.max(np.sqrt(f1**2 + f2**2 + f3**2))
            elif scale == 2:
                step_size = 10**(-(scale + 1.5)) / np.max(np.sqrt(f1**2 + f2**2 + f3**2))

            _, _, _, _, _, flag = compVOTGradients(yk1_temp, yk2_temp, yk3_temp, I0, I1, lambda_, gamma)
            while flag and not converged:
                step_size /= 2
                if step_size < 1e-8:
                    converged = True
                    step_size = 0
                    if iter == 2:
                        results.update({"f1": f1, "f2": f2, "f3": f3})

                xk1_temp = yk1minus1 - step_size * f1t
                xk2_temp = yk2minus1 - step_size * f2t
                xk3_temp = yk3minus1 - step_size * f3t

                yk1_temp = xk1_temp + (iter - 2) / (iter + 1) * (xk1_temp - xk1minus1)
                yk2_temp = xk2_temp + (iter - 2) / (iter + 1) * (xk2_temp - xk2minus1)
                yk3_temp = xk3_temp + (iter - 2) / (iter + 1) * (xk3_temp - xk3minus1)

                _, _, _, _, _, flag = compVOTGradients(yk1_temp, yk2_temp, yk3_temp, I0, I1, lambda_, gamma)

            xk1, xk2, xk3 = xk1_temp, xk2_temp, xk3_temp
            yk1, yk2, yk3 = yk1_temp, yk2_temp, yk3_temp

        if (not converged) or iter < 3:
            yk1minus2, yk2minus2, yk3minus2 = yk1minus1, yk2minus1, yk3minus1
            yk1minus1, yk2minus1, yk3minus1 = yk1, yk2, yk3
            xk1minus1, xk2minus1, xk3minus1 = xk1, xk2, xk3

        err1[iter-1] = (0.5 * np.sum(((Ierror.flatten() / I0.flatten()) * mask.flatten())**2) / np.count_nonzero(mask) ) 
        err2[iter-1] = (0.5 * np.sum((Ierror.flatten() / I0.flatten())**2) / I0.size ) 
        err3[iter-1] = ( np.mean((Ierror.flatten() / I0.flatten())**2) ) 
        err4[iter-1] = (0.5 * np.sum(Ierror.flatten()**2) ) 

        if err1[iter - 1] / err1[0] > level:
            if it == 0:
                it = iter
            lambda_ = penalty
        mass[iter-1] = np.sum(((yk1minus2 - X)**2 + (yk2minus2 - Y)**2 + (yk3minus2 - Z)**2) * I0)

        if iter == 1:
            C1, C2, C3 = curl(f1, f2, f3)
        else:
            C1, C2, C3 = curl(yk1minus2, yk2minus2, yk3minus2)

        C = np.sum(C1.flatten()**2 + C2.flatten()**2 + C3.flatten()**2)  # L2 norm
        errorcurl[iter-1] = (0.5 * C)

        objective = err4 + lambda_ * errorcurl

        if iter >= max_iters:
            max_iters *= 2
            new_err1, new_err2, new_err3, new_err4, new_mass, new_errorcurl = np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters)),np.zeros((max_iters))
            new_err1[:iter] = err1[:iter]
            new_err2[:iter] = err2[:iter]
            new_err3[:iter] = err3[:iter]
            new_err4[:iter] = err4[:iter]
            new_mass[:iter] = mass[:iter]
            new_errorcurl[:iter] = errorcurl[:iter]
            err1, err2, err3, err4, mass, errorcurl = new_err1, new_err2, new_err3, new_err4, new_mass, new_errorcurl 

        if iter > 50 and (objective[iter - 2] < objective[iter-1]):
            return results


        if (
            (converged and scale != 0) or (converged and scale == 0 and iter > 2) or
            (iter > 500 and scale == 1 and round(err3[iter-1]) - round(err3[iter - 2]) * 10**6 == 0) or
            (scale == 0 and round(err3[iter-1] * 10**3) / 10**3 <= cutoff  and iter > 2) or
            (scale != 0 and err3[iter-1] <= cutoff0)
        ): 
            return results

        # print(np.sum(I0_recon))
        I0_recon = I0_recon / np.sum(I0_recon) * 10**6
        results['iter'] = iter
        results["f1"] = yk1minus2
        results["f2"] = yk2minus2
        results["f3"] = yk3minus2
        results["I0_recon"] = I0_recon
        results["mass"] = mass
        results["MSE2"] = err2
        results["MSE1"] = err1
        results["MSE3"] = err3
        results["curl"] = 2 * np.asarray(errorcurl) / np.size(C)
        results["I0"] = I0
        results["I1"] = I1
        results["objective"] = objective

        iter += 1

if __name__ == '__main__':
    shape = (14,15,16)
    f1,f2,f3,I0 = np.ones(shape),np.ones(shape),np.ones(shape),np.zeros(shape)
