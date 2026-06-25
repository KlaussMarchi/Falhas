import os
import shutil
import glob

lucas_dir = 'Lucas'
models = [d for d in os.listdir(lucas_dir) if os.path.isdir(os.path.join(lucas_dir, d))]

for model in models:
    marlim_dir = os.path.join(lucas_dir, model, 'marlim')
    if not os.path.exists(marlim_dir):
        continue
        
    print(f"Migrating {model}...")
    
    # Target directory
    target_masks = os.path.join(marlim_dir, 'patch_1200', 'masks')
    os.makedirs(target_masks, exist_ok=True)
    
    # Source .dat masks
    old_masks_dat = os.path.join(marlim_dir, 'masks', 'dat')
    if os.path.exists(old_masks_dat):
        for dat_file in glob.glob(os.path.join(old_masks_dat, '*.dat')):
            # Move to new target
            filename = os.path.basename(dat_file)
            shutil.move(dat_file, os.path.join(target_masks, filename))
            
    # Now delete legacy folders and files in marlim/
    legacy_items = ['images', 'masks', 'predicted_crop_il-0-63_xl-0-2239_z-0-1600.npy']
    for item in legacy_items:
        path = os.path.join(marlim_dir, item)
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
    # Also remove any .npy files in marlim_dir
    for npy_file in glob.glob(os.path.join(marlim_dir, '*.npy')):
        os.remove(npy_file)
        
print("Migration completed!")
