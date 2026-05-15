import pandas as pd
import numpy as np
import glob, os, shutil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize as MplNormalize


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


def show3DCube(ax, volume, label, x_ratio=0.1, y_ratio=0.9, z_ratio=0.9, stride=1):
    """
    Plota fatias ortogonais se cruzando.
    x_ratio: controla a posição esquerda/direita do plano YZ.
    y_ratio: controla a posição frente/trás do plano XZ (0.8 empurra para trás).
    z_ratio: controla a posição cima/baixo do plano XY (chão).
    stride: Aumente para 2 ou 3 para renderizar muito mais rápido (sacrificando um pouco de resolução).
    """
    nx, ny, nz = volume.shape
    pos_x, pos_y, pos_z = int(nx * x_ratio), int(ny * y_ratio), int(nz * z_ratio)

    cmap = plt.cm.gray
    norm = MplNormalize(vmin=volume.min(), vmax=volume.max())

    def plot_plane(axis_to_fix, fixed_pos):
        if axis_to_fix == 'y':    # Plano XZ
            ranges_dim1 = [(0, pos_x + 1), (pos_x, nx)]
            ranges_dim2 = [(0, pos_z + 1), (pos_z, nz)]
        elif axis_to_fix == 'x':  # Plano YZ
            ranges_dim1 = [(0, pos_y + 1), (pos_y, ny)]
            ranges_dim2 = [(0, pos_z + 1), (pos_z, nz)]
        else:                     # Plano XY (z)
            ranges_dim1 = [(0, pos_x + 1), (pos_x, nx)]
            ranges_dim2 = [(0, pos_y + 1), (pos_y, ny)]

        for start1, end1 in ranges_dim1:
            for start2, end2 in ranges_dim2:
                arr1, arr2 = np.arange(start1, end1), np.arange(start2, end2)

                if axis_to_fix == 'y':
                    X, Z = np.meshgrid(arr1, arr2, indexing='ij')
                    Y = np.full_like(X, fixed_pos)
                    Z_plot = nz - Z
                    data = volume[start1:end1, fixed_pos, start2:end2]
                elif axis_to_fix == 'x':
                    Y, Z = np.meshgrid(arr1, arr2, indexing='ij')
                    X = np.full_like(Y, fixed_pos)
                    Z_plot = nz - Z
                    data = volume[fixed_pos, start1:end1, start2:end2]
                else:
                    X, Y = np.meshgrid(arr1, arr2, indexing='ij')
                    Z_plot = np.full_like(X, nz - fixed_pos)
                    data = volume[start1:end1, start2:end2, fixed_pos]

                ax.plot_surface(X, Y, Z_plot, facecolors=cmap(norm(data)), shade=False, antialiased=False, linewidth=0, rstride=stride, cstride=stride)

    plot_plane('y', pos_y) # Parede XZ
    plot_plane('x', pos_x) # Parede YZ
    plot_plane('z', pos_z) # Chão XY

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(label, fontsize=14, fontweight='bold', loc='left')
    ax.view_init(elev=20, azim=-45)


def showSteps(steps, save=None):
    fig = plt.figure(figsize=(18, 12))
    for i, (volume, label) in enumerate(steps):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        show3DCube(ax, volume, label)
        
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)

    plt.show()