# MESMA ESCALA DE PIXEL: SEÇÕES 128x128 DO dataset_74 E DO dataset_wu (COM RÓTULO) AO LADO DE TILES REAIS DO MARLIM (COM O ESPECIALISTA)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R = '/home/grva-mint/Projects/Falhas'
rng = np.random.default_rng(3)
rows = []

for name in ('dataset_74', 'dataset_wu'):
    df = pd.read_csv(f'{R}/Dataset/{name}/DataBase.csv')
    for i in rng.choice(len(df), 4, replace=False):
        img, msk = np.load(df.img_path.iloc[i]), np.load(df.mask_path.iloc[i])
        rows.append((f'{name} {df.id.iloc[i]}', img[64], msk[64]))

db = pd.read_csv(f'{R}/Synthetic/Regions/Marlim/files/DataBase.csv')
faulted = db[db.region == 'faulted']
for i in rng.choice(len(faulted), 8, replace=False):
    row  = faulted.iloc[i]
    tile = np.load(row.img_path)
    gt   = np.array(Image.open(f'{R}/Blocks/Marlim/files/patches/{row.patch}/{row.patch}_interpretado.png').convert('L')) > 128
    rows.append((f'Marlim {row.patch} z{row.z0} x{row.x0}', tile[int(row.inline)], gt[row.z0:row.z0 + 128, row.x0:row.x0 + 128]))

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for ax, (title, img, msk) in zip(axes.ravel(), rows):
    ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    my, mx = np.nonzero(msk)
    ax.scatter(mx, my, s=0.4, c='red', alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('synreal.png', dpi=55)
print('ok')
