# UM DATASET DE EXPERIMENTO AO LADO DO MARLIM NA MESMA ESCALA DE PIXEL: 8 SEÇÕES INLINE CENTRAIS (COM O RÓTULO) E 4 TILES REAIS DE DEV
# uso: python dataset_view.py <tag> [<tag2> ...]
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/home/grva-mint/Projects/Falhas'
EXP  = f'{ROOT}/Synthetic/files/experiments'
REAL = [('1200', 'patch_d000_h012_w008.dat'), ('1300', 'patch_d000_h013_w020.dat'), ('1200', 'patch_d000_h014_w027.dat'), ('1300', 'patch_d000_h008_w030.dat')]

for tag in sys.argv[1:]:
    df   = pd.read_csv(f'{EXP}/datasets/{tag}/DataBase.csv')
    pick = np.random.default_rng(1).choice(len(df), 8, replace=False)
    rows = [(f'{tag} {df.id.iloc[i]} rótulo {df.msk_frac.iloc[i] * 100:.1f}%', np.load(df.img_path.iloc[i])[64], np.load(df.mask_path.iloc[i])[64]) for i in pick]
    rows += [(f'Marlim {p} {n[6:-4]}', np.fromfile(f'{ROOT}/Blocks/Marlim/files/patches/{p}/tiles/{n}', np.float32).reshape(128, 128, 128)[64], None) for p, n in REAL]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for ax, (title, sec, lab) in zip(axes.ravel(), rows):
        ax.imshow(sec, cmap='gray', vmin=np.percentile(sec, 1), vmax=np.percentile(sec, 99))
        if lab is not None:
            zz, yy = np.nonzero(lab)
            ax.scatter(yy, zz, s=0.3, c='red', alpha=0.35)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(f'{EXP}/checks/view_{tag}.png', dpi=50); plt.close()
    print('ok', tag)
