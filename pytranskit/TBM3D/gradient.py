# these are the functions for TBM software
# Reference: 
#   [1] Kundu, Shinjini, et al. "Discovery and visualization of structural biomarkers from MRI using transport-based morphometry." NeuroImage 167 (2018): 256-275.

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, map_coordinates
from scipy.linalg import svd, orth
from pytranskit.optrans.decomposition import PLDA

def compVOTGradients(f1, f2, f3, I0, I1, lambd=50, gamma=0.1):
    """
    Computes the gradients required in VOT et al's variation optimization approach.

    Parameters:
    - f1, f2, f3: Current deformation fields (3D arrays)
    - I0, I1: Original images (3D arrays)
    - lambd: Penalty for curl term
    - gamma: Penalty for mass transport term

    Returns:
    - f1t, f2t, f3t: Gradients to update in the next iteration
    - I0_recon: Reconstructed I0
    - Ierror: Intensity error
    - flag: 1 if the current deformation is not diffeomorphic
    """

    X, Y, Z = np.meshgrid(
        np.arange(f1.shape[0]), np.arange(f1.shape[1]), np.arange(f1.shape[2]), indexing='ij'
    )

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

    # Determinant of Jacobian
    detf = (
        f1x * f2y * f3z + f1y * f2z * f3x + f1z * f2x * f3y
        - f1x * f2z * f3y - f1y * f2x * f3z - f1z * f2y * f3x
    )

    # Check diffeomorphism
    flag = int(np.any(detf < 0))

    # Interpolate I1
    coordinates = np.array([f1.flatten(), f2.flatten(), f3.flatten()])
    It = map_coordinates(I1, coordinates, order=3, mode='constant', cval=np.min(I1), prefilter = True).reshape(f1.shape)
    # Taking absolute value
    It = np.abs(It)
    Ierror = detf * It - I0
    Itx, Ity, Itz = np.gradient(It)

    # Compute divD
    g11x, _, _ = np.gradient((f2y * f3z - f2z * f3y) * Ierror * It)
    _, g12y, _ = np.gradient(-(f2x * f3z - f2z * f3x) * Ierror * It)
    _, _, g13z = np.gradient((f2x * f3y - f2y * f3x) * Ierror * It)

    g21x, _, _ = np.gradient(-(f1y * f3z - f1z * f3y) * Ierror * It)
    _, g22y, _ = np.gradient((f1x * f3z - f1z * f3x) * Ierror * It)
    _, _, g23z = np.gradient(-(f1x * f3y - f1y * f3x) * Ierror * It)

    g31x, _, _ = np.gradient((f1y * f2z - f1z * f2y) * Ierror * It)
    _, g32y, _ = np.gradient(-(f1x * f2z - f1z * f2x) * Ierror * It)
    _, _, g33z = np.gradient((f1x * f2y - f1y * f2x) * Ierror * It)

    divD1 = g11x + g12y + g13z
    divD2 = g21x + g22y + g23z
    divD3 = g31x + g32y + g33z

    # Compute curlC
    curlC1 = f2xy - f1yy - f1zz + f3xz
    curlC2 = f3yz - f2zz - f2xx + f1yx
    curlC3 = f1zx - f3xx - f3yy + f2zy

    # Compute gradients
    f1t = detf * Itx * Ierror - divD1 + lambd * curlC1 - gamma * (X - f1) * I0
    f2t = detf * Ity * Ierror - divD2 + lambd * curlC2 - gamma * (Y - f2) * I0
    f3t = detf * Itz * Ierror - divD3 + lambd * curlC3 - gamma * (Z - f3) * I0
    
    # Zero out boundary derivatives
    Z = np.pad(np.ones((f1.shape[0] - 2, f1.shape[1] - 2, f1.shape[2] - 2)), 1, mode='constant')
    f1t *= Z
    f2t *= Z
    f3t *= Z

    # Reconstruct I0
    I0_recon = detf * It
    # print("it",np.max(It),np.min(It),np.sum(It))
    # print("detf",np.max(detf),np.min(detf),np.sum(detf))
    # print("I0",np.max(I0_recon),np.min(I0_recon),np.sum(I0_recon))
    
    return f1t, f2t, f3t, I0_recon, Ierror, flag

def projection_metric(A, B):
    """
    Projection metric based on principal angles between column spaces of A and B.
    
    Parameters
    ----------
    A : np.ndarray, shape (N, M)
        Orthonormal matrix.
    B : np.ndarray, shape (N, M)
        Orthonormal matrix.
    
    Returns
    -------
    d : float
        Distance between column spaces of A and B.
    """
    _, M = A.shape
    
    # Orthonormalize A and B (like MATLAB's orth)
    A = orth(A.astype(float))
    B = orth(B.astype(float))
    
    # Normalize each column (to match MATLAB loop)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    
    # SVD of A'B
    _, S, _ = svd(A.T @ B)

    costheta = np.diag(np.atleast_2d(S)) if S.ndim > 1 else S
    v = M - np.sum(costheta**2)
    if v < 0:
        v = 0
    
    d = np.sqrt(v)
    return d

def calculate_alpha(FINAL_FEATS, EIGENV1, labels):
    """
    Python translation of Calculate_Alpha.m
    
    Parameters
    ----------
    FINAL_FEATS : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    EIGENV1 : np.ndarray
        Eigenvectors, shape (n_features, n_features).
    labels : np.ndarray
        Class labels, shape (n_samples,).
    
    Returns
    -------
    Thresh : float
        The chosen alpha.
    error_subspace : list
        List of projection metric errors between consecutive subspaces.
    """
    
    x = np.arange(0.01, 100 + 7, 7)
    
    # Equivalent of MATLAB's VecPCA
    VecPCA = EIGENV1[:, :FINAL_FEATS.shape[1]].astype(float)
    # print(VecPCA)
    
    Vec = []
    error_subspace = []
    
    for Alpha in x:
        # Run PLDA
        plda=PLDA(alpha=Alpha,n_components=3)
        plda.fit(FINAL_FEATS, labels)
        # print(plda.components_.T )
        PLDA_directions = plda.components_.T 
        # print(PLDA_directions)
        
        Vec_curr = VecPCA.dot(PLDA_directions) 
        # print(Vec_curr)
        # break
        
        if len(Vec) > 0:
            # Align sign with previous basis
            align_matrix = np.diag(np.sign(np.diag(Vec_curr.T @ Vec[-1])))
            Vec_curr = Vec_curr.dot(align_matrix) 
            
            # Projection metric error
            err = projection_metric(Vec_curr, Vec[-1])
            error_subspace.append(err)
        
        Vec.append(Vec_curr)
    
    # Find cutoff index
    cutoff = 0.5  # (0.5% in MATLAB; note the factor *100 below)
    change = np.abs(np.gradient(np.array(error_subspace, dtype=float)))
    
    indices = np.where(change * 100 < cutoff)[0]
    if len(indices) > 0:
        Thresh = x[indices[0]]
    else:
        Thresh = x[-1]  # fallback
    
    return Thresh


if __name__ == '__main__':
    A, B, C = np.asarray([[1,1,1],[0,1,0],[2,1,-1],[-1,1,2]]), np.asarray([[1,0,0,0],[0,1,0,0],[-1,2,1,0],[2,-1,0,1]]), np.asarray([0,1,0,1])
    print(A.shape)
    print(calculate_alpha(A,B,C))