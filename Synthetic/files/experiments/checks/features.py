# CONFERÊNCIA VISUAL DAS ETAPAS NOVAS DO GERADOR: SEÇÃO INLINE CENTRAL (z × y) DE TILES COM CADA RECURSO LIGADO, NA ESCALA
# DA REDE (MIN-MAX DO VOLUME), AO LADO DE TILES REAIS DO MARLIM DE DEV NA MESMA ESCALA DE PIXEL, COM O RÓTULO EM VERMELHO
import sys, json, time
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
ROOT = '/home/grva-mint/Projects/Falhas'
sys.path.insert(0, ROOT)
from Synthetic.index import SyntheticGenerator

FEATURES = {
    'padrão (dataset_74)': {},
    'corpos (1-3)': {'bodyCount': (1, 4)},
    'água + zona morta + hachura': {'seafloorProb': 1.0, 'deadZoneProb': 1.0, 'seafloorDepth': (0.1, 0.3), 'deadZoneDepth': (0.5, 0.8), 'deadZoneHatch': (0.1, 0.2)},
    'campo de ganho 0,6': {'gainField': 0.6},
    'fase 90°': {'waveletPhase': (90.0, 90.0)},
    'período longo (f 33-55)': {'waveletFreq': (33, 55)},
}
SEEDS = (11, 12)
real  = [('1200', 'patch_d000_h012_w008.dat'), ('1300', 'patch_d000_h013_w020.dat'), ('1200', 'patch_d000_h014_w027.dat'), ('1300', 'patch_d000_h011_w012.dat')]

rows = []
for name, params in FEATURES.items():
    for seed in SEEDS:
        g = SyntheticGenerator(seed=seed)
        g.set(params)
        t0 = time.time()
        img, msk = g.get()
        img = (img - img.min()) / (img.max() - img.min())
        rows.append((f'{name} s{seed} ({time.time() - t0:.0f}s)', img[64].T, msk[64].T))
        print(rows[-1][0], flush=True)

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for ax, (title, sec, lab) in zip(axes.ravel(), rows + [(f'Marlim {p} {n[6:-4]}', np.fromfile(f'{ROOT}/Blocks/Marlim/files/patches/{p}/tiles/{n}', np.float32).reshape(128, 128, 128)[64], None) for p, n in real]):
    ax.imshow(sec, cmap='gray', vmin=np.percentile(sec, 1), vmax=np.percentile(sec, 99))
    if lab is not None:
        zz, yy = np.nonzero(lab)
        ax.scatter(yy, zz, s=0.3, c='red', alpha=0.4)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f'{ROOT}/Synthetic/files/experiments/checks/features.png', dpi=55)
print('ok')
