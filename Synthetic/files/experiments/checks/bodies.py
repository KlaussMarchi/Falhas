# CONFERÊNCIA DOS CORPOS: 3 TILES COM 2 CORPOS, RUÍDO PADRÃO E RUÍDO BAIXO, SEÇÕES INLINE 32/64/96 COM O RÓTULO
import sys, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(6)
sys.path.insert(0, '/home/grva-mint/Projects/Falhas')
from Synthetic.index import SyntheticGenerator
rows = []
for noise in ((0.015, 0.618), (0.0, 0.15)):
    for seed in (21, 22, 23):
        g = SyntheticGenerator(seed=seed)
        g.set({'bodyCount': (2, 3), 'noiseLevel': noise})
        img, msk = g.get()
        img = (img - img.min()) / (img.max() - img.min())
        for i in (32, 64, 96):
            rows.append((f'ruído {noise} s{seed} x={i}', img[i].T, msk[i].T))
fig, axes = plt.subplots(6, 3, figsize=(15, 30))
for ax, (t, s, m) in zip(axes.ravel(), rows):
    ax.imshow(s, cmap='gray', vmin=np.percentile(s, 1), vmax=np.percentile(s, 99))
    zz, yy = np.nonzero(m); ax.scatter(yy, zz, s=0.3, c='red', alpha=0.3)
    ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig('/home/grva-mint/Projects/Falhas/Synthetic/files/experiments/checks/bodies.png', dpi=45)
print('ok')
