import glob, os, shutil, cv2
import numpy as np
import matplotlib.pyplot as plt


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


    h, w  = img.shape[:2]
    scale = size / max(h, w)

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    top    = (size - new_h) // 2
    bottom = size - new_h - top
    left   = (size - new_w) // 2
    right  = size - new_w - left
    return (new_h, new_w, top, bottom, left, right, scale)

def pasteMask(img, mask, alpha=0.5, threshold=0.5, color=(255, 0, 0)):
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

def showTile(img):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = img.shape[0] // 2
    
    slice_x = img[mid, :, :]  # Plano YZ
    slice_y = img[:, mid, :]  # Plano XZ  
    slice_z = img[:, :, mid]  # Plano XY

    axes[0].imshow(slice_x, cmap='gray')
    axes[0].set_title('Slice X=64')

    axes[1].imshow(slice_y, cmap='gray')
    axes[1].set_title('Slice Y=64')

    axes[2].imshow(slice_z, cmap='gray')
    axes[2].set_title('Slice Z=64')
    plt.tight_layout()
    plt.show()