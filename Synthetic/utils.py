import pandas as pd
import numpy as np
import glob, os, shutil, cv2, json, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.ndimage as ndimage
from tqdm import tqdm


def getFiles(path, limit=None, shuffle=False):
    target = sorted(glob.glob(os.path.join(path, '*')))
    if shuffle:
        np.random.shuffle(target) 
    return target[:limit]

def formatAxis(img):
    return np.transpose(img, (0, 2, 1))

def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def showTile(img, mask=False, save=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    mid_x = img.shape[0] // 2
    mid_y = img.shape[1] // 2
    mid_z = img.shape[2] // 2

    slices = [
        img[mid_x, :, :],  # Plano YZ (Corte ao longo do eixo X)
        img[:, mid_y, :],  # Plano XZ (Corte ao longo do eixo Y)
        img[:, :, mid_z]   # Plano XY (Corte ao longo do eixo Z)
    ]

    slices[0] = np.array(slices[0])
    arr_y = np.array(slices[1])
    arr_z = np.array(slices[2])
    slices[1] = np.rot90(arr_z, -1)
    slices[2] = arr_y

    cmap_config = ListedColormap(['black', 'red', 'green', 'blue']) if mask else 'gray'
    vmin, vmax  = (0, 3) if mask else (None, None)
    titles = [f'Slice X={mid_x}', f'Slice Y={mid_y}', f'Slice Z={mid_z}']
    
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap=cmap_config, vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        return plt.close(fig)

    plt.show() 


def show3DCube(ax, volume, label):
    from matplotlib.colors import Normalize as MplNormalize

    nx, ny, nz = volume.shape
    cmap = plt.cm.gray
    vlo, vhi = np.percentile(volume, [2, 98])
    norm = MplNormalize(vmin=vlo, vmax=vhi)

    xs = np.arange(nx)
    ys = np.arange(ny)
    zs = np.arange(nz)

    midX = nx // 2
    midY = ny // 2
    surfKwargs = dict(shade=False, antialiased=False, linewidth=0, rcount=nx, ccount=nz)

    # Left wall: x=midX plane (inline section)
    sliceLeft = volume[midX, :, :]
    yL, zL = np.meshgrid(ys, zs, indexing='ij')
    xL = np.full_like(yL, midX)
    ax.plot_surface(xL, yL, nz - 1 - zL, facecolors=cmap(norm(sliceLeft)), **surfKwargs)

    # Right wall: y=midY plane (crossline section)
    sliceRight = volume[:, midY, :]
    xR, zR = np.meshgrid(xs, zs, indexing='ij')
    yR = np.full_like(xR, midY)
    ax.plot_surface(xR, yR, nz - 1 - zR, facecolors=cmap(norm(sliceRight)), **surfKwargs)

    # Floor: bottom depth slice
    sliceFloor = volume[:, :, -1]
    xF, yF = np.meshgrid(xs, ys, indexing='ij')
    zF = np.zeros_like(xF)
    ax.plot_surface(xF, yF, zF, facecolors=cmap(norm(sliceFloor)), **surfKwargs)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.view_init(elev=20, azim=315)


def showSteps(steps, save=None):
    fig = plt.figure(figsize=(18, 12))
    for i, (volume, label) in enumerate(steps):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        show3DCube(ax, volume, label)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)

    plt.show()