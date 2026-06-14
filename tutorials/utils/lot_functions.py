import os
import warnings

import numpy as np

from scipy.ndimage import convolve
from scipy.signal.windows import gaussian
from scipy.spatial import cKDTree

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label

from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import NullLocator


## ---------------------------------------------------------
def visualize_LOT(Data, Intensity, Nx, Ny, scale=1, crop=1):
    NG = 35
    I1 = np.zeros((scale * Nx, scale * Ny))

    loc = scale * np.round(np.reshape(Data, (int(len(Data) / 2), 2), order='F'))

    linearind = ((loc[:, 1] - 1) * (scale * Nx) + loc[:, 0] - 1).astype(int)

    i1 = np.reshape(I1, (scale * Nx * scale * Ny), order='F')
    i1[linearind] = Intensity
    I1 = np.reshape(i1, (scale * Nx, scale * Ny), order='F')

    h1 = gaussian(NG * scale, std=12).reshape(-1, 1)
    h = h1 @ h1.T
    h = h / np.sum(h)

    I1 = convolve(I1, h, mode='constant')
    I1 = I1 - I1.min()
    I1 = I1 / I1.max()

    return I1.T


## ---------------------------------------------------------
def img2pts_Lloyd_v2(img, Nmasses, seed=None):

    def fromInd2Coord(indices, Ny):
        indices = np.array(indices, dtype=float)+1
        x = (indices // Ny) + 1
        y = np.mod(indices, Ny)

        # Handle the case where mod returns 0
        y[y == 0] = Ny
        x[y == Ny] = x[y == Ny] - 1

        return np.vstack([y, x])

    def L2_distance(A, B):
        A = np.asarray(A)
        B = np.asarray(B)
        distances = np.linalg.norm(A - B, axis=0)
        return distances.reshape(-1, 1)

    stopLloyd = 0.5
    vis = 0

    img = img.astype(float)
    ny, nx = img.shape

    img_t = img / np.max(img)
    iuint=(img_t*255).astype(np.uint8); t256=threshold_otsu(iuint); level=0.22*float(t256)/255.0 # level = threshold_otsu(img_t) * 0.22

    BW = (img_t >= level).astype(float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        BW = remove_small_objects(BW.astype(bool), min_size=7).astype(float)

    labeled = label(BW.astype(int))
    props = regionprops(labeled)

    if len(props) == 0:
        raise RuntimeError("No connected component after thresholding.")

    bb = props[0].bbox
    sy = int(np.round(bb[0])) + 1  # +1 for MATLAB 1-based indexing
    sx = int(np.round(bb[1])) + 1
    Ny = int(np.round(bb[2] - bb[0]))
    Nx = int(np.round(bb[3] - bb[1]))

    img_zeroed = img.copy()
    img_zeroed[img_t < level] = 0

    ind = np.where((img_t >= level).T.flatten())[0]
    if Nmasses is None:
        Nmasses = int(len(ind) / 25) + 20

    useK = min(Nmasses, len(ind))

    rng = np.random.default_rng(seed)
    output_Index = rng.choice(ind, size=useK, replace=False)


    res_P = fromInd2Coord(output_Index, ny)  # 2×K
    res_c = np.ones(res_P.shape[1])          # K

    BW2 = np.zeros_like(img_t)
    img_x = img_zeroed / np.sum(img_zeroed)
    BW2[sy-1:sy+Ny-1, sx-1:sx+Nx-1] = img_x[sy-1:sy+Ny-1, sx-1:sx+Nx-1]
    row, col = np.where(BW2.T > 0)
    V = BW2.T[row, col].astype(float)

    Pl = np.vstack([col + 1, row + 1])  # 2×N

    if len(ind) < Nmasses: # not checked ---------
        res_P2 = fromInd2Coord(ind, ny)
        img_flat = img_zeroed.T.flatten()
        nlz = np.sum(img_flat[ind])
        res_c2 = img_flat[ind] / nlz
        var_out = np.zeros(len(ind))
        llerr = 0
        return res_P2, res_c2, llerr, Pl, V, var_out

    llerr = []
    cur = 1  # Start at 1
    differ = 1
    couunt_debug=0
    while differ > stopLloyd:
        couunt_debug=couunt_debug+1

        K = res_P.shape[1]

        _, neighbors_map = cKDTree(res_P.T).query(Pl.T, workers=1)
        # dists = ((Pl.T[:, None, :] - res_P.T[None, :, :]) ** 2).sum(axis=2)
        # neighbors_map2 = np.argmin(dists, axis=1)

        Vsum = np.bincount(neighbors_map, weights=V, minlength=K) + 1e-10
        cx = np.bincount(neighbors_map, weights=V * Pl[0], minlength=K) / Vsum
        cy = np.bincount(neighbors_map, weights=V * Pl[1], minlength=K) / Vsum

        diff0 = Pl[0] - cx[neighbors_map]
        diff1 = Pl[1] - cy[neighbors_map]
        errUB = np.bincount(neighbors_map, weights=V * (diff0**2 + diff1**2), minlength=K)

        res_P[:] = np.vstack([cx, cy])
        res_c = Vsum - 1e-10

        llerr.append(np.sum(errUB))

        if cur >= 4:
            differ = (llerr[cur-1] - llerr[cur-2]) / (llerr[cur-2] - llerr[cur-3])
        else:
            differ = 1
        cur += 1

    eps = 1e-10
    keep = np.where(res_c >= eps)[0]

    res_P = res_P[:, keep]
    res_c = res_c[keep]

    res_c = res_c / np.sum(res_c)
    res_P2 = res_P[[1, 0], :] 
    res_c2 = res_c.reshape(-1, 1) 

    X = res_P2.T
    a = np.squeeze(res_c2)

    return X, a

## ---------------------------------------------------------
def particleApproximation(img_array, Nmasses, seed=None):
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) # os.cpu_count() or 1
    print(f"Using {n_cores} CPU cores ...\n")

    results = Parallel(n_jobs=n_cores)(
        delayed(img2pts_Lloyd_v2)(img_array[i], Nmasses, seed=seed) 
        for i in tqdm(range(len(img_array)), desc="Particle Approximation")
    )

    X1_list, a1_list = zip(*results)
    return list(X1_list), list(a1_list)
    
## ---------------------------------------------------------
FACE           = ['#D8D8D8', '#FFA040']
EDGE           = ['#444444', '#7A3800']
EXTRA_color    = ['#D8D8D8', '#7A3800']
BG, AXES_BG, GRID_C, SPINE_C, TICK_C = '#13131a', '#1a1a24', '#22222e', '#333344', '#666677'
def draw_panel(proj, label, pat_id, n_std, point_size, point_alpha, title, fig, gs_col, img_top_text=None, blocks=None):                
    for dim in range(2):
        mu = proj[:, dim].mean()
        sigma = proj[:, dim].std(ddof=0)
        if sigma == 0:
            sigma = 1.0
        proj[:, dim] = (proj[:, dim] - mu) / sigma

    x, y = proj[:, 0], proj[:, 1]
    unique_labels, unique_patients = np.unique(label), np.unique(pat_id)
    li = {lbl: i for i, lbl in enumerate(unique_labels)}

    xlim = (-5.0, 5.0)
    ylim = (-5.0, 5.0)

    gs = gs_col.subgridspec(
        3, 3,
        # width_ratios=[1, 0.08, 0.25],
        # height_ratios=[0.25, 0.08, 1],
        width_ratios=[1, 0.15, 0.25],
        height_ratios=[0.25, 0.15, 1],
        hspace=0.02,
        wspace=0.02,
    )
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_img_h = fig.add_subplot(gs[1, 0])
    ax_sc    = fig.add_subplot(gs[2, 0])
    ax_img_v = fig.add_subplot(gs[2, 1])
    ax_right = fig.add_subplot(gs[2, 2])

    for ax in [ax_sc, ax_top, ax_right]:
        ax.set_facecolor(AXES_BG)
        for sp in ax.spines.values(): sp.set_color(SPINE_C)
        ax.tick_params(colors=TICK_C, labelcolor=TICK_C, length=4)

    for ax in [ax_img_h, ax_img_v]:
        ax.set_facecolor(BG)
        ax.axis('off')
    
    img_top=blocks[0]
    ax_img_h.imshow(img_top, cmap='gray', aspect='auto')
    ax_img_h.text(1.02, 0.5, img_top_text or '', color='#cccccc', fontsize=16, ha='left', va='center', transform=ax_img_h.transAxes)
    
    img_right=np.flipud(blocks[1].T)
    ax_img_v.imshow(img_right, cmap='gray', aspect='auto')

    ax_sc.set(xlim=xlim, ylim=ylim)
    ax_sc.set_xlabel('Direction 1', color=TICK_C, fontsize=12)
    ax_sc.set_ylabel('Direction 2', color=TICK_C, fontsize=12)
    ax_sc.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
    ax_sc.axhline(0, color='#2a2a38', linewidth=1.2, zorder=1)
    ax_sc.axvline(0, color='#2a2a38', linewidth=1.2, zorder=1)
    # ax_sc.set_title(title, color='#e0e0e0', fontsize=12, fontweight='bold', pad=8)

    sigma_ticks  = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    sigma_labels = [r'$-5\sigma$', r'$-4\sigma$', r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$', r'$0$', r'$+1\sigma$', r'$+2\sigma$', r'$+3\sigma$', r'$+4\sigma$', r'$+5\sigma$']
    ax_sc.set_xticks(sigma_ticks); ax_sc.set_xticklabels(sigma_labels)
    ax_sc.set_yticks(sigma_ticks); ax_sc.set_yticklabels(sigma_labels)

    for pid in unique_patients:
        pts = proj[pat_id == pid, :2]
        if len(pts) < 3: continue
        ci = li[label[pat_id == pid][0]]

    for lbl in unique_labels:
        ci, mask = li[lbl], label == lbl
        ax_sc.scatter(x[mask], y[mask], s=point_size, facecolors=FACE[ci], edgecolors=EDGE[ci], linewidths=0.6, alpha=point_alpha, zorder=4)

    bins_x = np.linspace(*xlim, 40)
    bins_y = np.linspace(*ylim, 40)
    ax_top.set_xlim(xlim); ax_top.grid(True, color=GRID_C, linewidth=0.6, axis='y'); ax_top.tick_params(labelbottom=False, bottom=False); ax_top.set_ylabel('count', color=TICK_C, fontsize=12)
    ax_right.set_ylim(ylim); ax_right.grid(True, color=GRID_C, linewidth=0.6, axis='x'); ax_right.tick_params(labelleft=False, left=False); ax_right.set_xlabel('count', color=TICK_C, fontsize=12)

    for lbl in unique_labels:
        ci, mask = li[lbl], label == lbl
        ax_top.hist(x[mask], bins=bins_x, color=FACE[ci], edgecolor=EDGE[ci], linewidth=0.5, alpha=0.7, zorder=3, density=True)
        ax_right.hist(y[mask], bins=bins_y, color=FACE[ci], edgecolor=EDGE[ci], linewidth=0.5, alpha=0.7, orientation='horizontal', zorder=3, density=True)
        ax_top.axvline(x[mask].mean(), color=EDGE[ci], linestyle='--', linewidth=2.0, alpha=0.9, zorder=5)
        ax_right.axhline(y[mask].mean(), color=EDGE[ci], linestyle='--', linewidth=2.0, alpha=0.9, zorder=5)

    ax_top.set_ylim(bottom=0); ax_right.set_xlim(left=0)

    class_handles = [mpatches.Patch(facecolor=FACE[li[lbl]], edgecolor=EDGE[li[lbl]], linewidth=0.8, label=f'{"Malignant" if lbl == max(unique_labels) else "Benign"}  (n={(label==lbl).sum()})') for lbl in unique_labels]
    leg = ax_sc.legend(handles=class_handles, loc='lower right', framealpha=0.25, facecolor='#222230', edgecolor=SPINE_C, labelcolor='#cccccc', fontsize=12)
    leg.get_title().set_color('#aaaaaa')

## ---------------------------------------------------------
def make_figure(proj_tr, label_tr, pat_label_tr, proj_te, label_te, pat_label_te, title_tr=None, title_te=None, show_contours=True, n_std=2, point_size=30, point_alpha=0.85, figsize=(18, 9), blocks=None):                              
    fig = plt.figure(figsize=figsize, facecolor=BG)
    outer = fig.add_gridspec(1, 2, wspace=0.12, left=0.06, right=0.97, top=0.93, bottom=0.09)

    for col, (proj, label, pat_id, title) in enumerate([(proj_tr, label_tr, pat_label_tr, title_tr), (proj_te, label_te, pat_label_te, title_te)]):
        gs = outer[col]                                                                 
        draw_panel(proj, label, pat_id, n_std, point_size, point_alpha, title, fig, gs, title, blocks=blocks)                         

    plt.show()
    return fig

## ---------------------------------------------------------
BG      = '#13131a'
AXES_BG = '#1a1a24'
SPINE_C = '#333344'
ACCENT  = '#FFA040'
LABEL_C = '#cccccc'
DIM_C   = '#444455'
def plot_feature_maps(blocks, cols=2):
    
    n    = len(blocks)
    rows = (n + 1) // cols

    fig = plt.figure(figsize=(18, rows * 1.2 + 1.2), facecolor=BG)
    fig.text(
        0.5, 1.05,
        'FEATURE MAP OVERVIEW',
        ha='center', va='top',
        color=ACCENT, fontsize=18, fontweight='bold',
        fontfamily='monospace'
    )
    fig.text(
        0.5, 0.99,
        f'{n} learned representations  ·  grayscale projection',
        ha='center', va='top',
        color=DIM_C, fontsize=14, fontfamily='monospace'
    )

    gs = gridspec.GridSpec(
        rows, cols,
        figure=fig,
        hspace=0.45, wspace=0.12,
        left=0.04, right=0.97,
        top=0.94, bottom=0.03
    )

    for i, img in enumerate(blocks):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(AXES_BG)

        ax.imshow(img, cmap='gray', aspect='auto')

        for sp in ax.spines.values():
            sp.set_color(SPINE_C)
            sp.set_linewidth(0.8)

        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

        ax.text(
            0.015, 0.97,
            f'{i+1:02d}',
            transform=ax.transAxes,
            color=ACCENT, fontsize=12, fontweight='bold',
            fontfamily='monospace',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#13131a', edgecolor='none', alpha=0.7)
        )

        ax.text(
            0.5, -0.045,
            f'Feature {i+1}',
            transform=ax.transAxes,
            color=LABEL_C, fontsize=14,
            fontfamily='monospace',
            va='top', ha='center'
        )

    if n % cols != 0:
        fig.add_subplot(gs[rows - 1, cols - 1]).set_visible(False)

    plt.show()
    return fig
    
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------
## ---------------------------------------------------------