import glob, os, shutil, cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap


def getFiles(path, limit=None, shuffle=False):
    target = sorted(glob.glob(os.path.join(path, '*')))
    if shuffle:
        np.random.shuffle(target) 
    return target[:limit]

def getAllFiles(base):
    return [os.path.join(root, file) for root, dirs, files in os.walk(base) for file in files]

def getFile(path, index):
    return getFiles(path)[index]

def discretize(img, thresh=127):
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def pasteMask(img, mask, alpha=0.5, threshold=0.5, color=(0, 0, 255)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid = img.shape[0] // 2
    
    slices = [
        (img[mid, :, :], mask[mid, :, :], 'Slice X (Sagittal)'),
        (img[:, mid, :], mask[:, mid, :], 'Slice Y (Coronal)'),
        (img[:, :, mid], mask[:, :, mid], 'Slice Z (Axial)')
    ]
    
    for i, (img_slice, mask_slice, title) in enumerate(slices):
        img_norm = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        
        overlay = img_rgb.copy()
        
        condition = mask_slice > threshold
        overlay[condition] = color 
        blended = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
        
        axes[i].imshow(blended)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def showTile(img, mask=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid       = img.shape[0] // 2

    slices = [
        img[mid, :, :],  # Plano YZ
        img[:, mid, :],  # Plano XZ
        img[:, :, mid]   # Plano XY
    ]
    
    cmap_config = 'gray'

    if mask:
        cmap_config = ListedColormap(['black', 'red', 'green', 'blue'])
        vmin, vmax  = 0, 3 
    else:
        vmin, vmax = (None, None)

    titles = ['Slice X', 'Slice Y', 'Slice Z']
    
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap=cmap_config, vmin=vmin, vmax=vmax)
        ax.set_title(f'{titles[i]}={mid}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()