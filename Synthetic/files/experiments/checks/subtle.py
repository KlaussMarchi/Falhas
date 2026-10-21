# CONFERÊNCIA DAS FALHAS SUTIS (H4): O MESMO TILE COM O REJEITO PADRÃO, COM REJEITO EM PERÍODOS (0,2-1,5) E RÓTULO SÓ ONDE VISÍVEL,
# E COM ARRASTO; SEÇÃO INLINE CENTRAL E O RÓTULO
import sys, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(6)
sys.path.insert(0, '/home/grva-mint/Projects/Falhas')
from Synthetic.index import SyntheticGenerator
CASES = {
    'padrão': {'noiseLevel': (0.0, 0.15)},
    'rejeito 0,2-1,5 período, rótulo ≥ 0,15 período': {'noiseLevel': (0.0, 0.15), 'faultThrowPeriod': (0.2, 1.5), 'faultVisible': 0.15},
    'idem + arrasto 0,3-0,9': {'noiseLevel': (0.0, 0.15), 'faultThrowPeriod': (0.2, 1.5), 'faultVisible': 0.15, 'faultDrag': (0.3, 0.9)},
}
rows = []
for name, params in CASES.items():
    for seed in (51, 52):
        g = SyntheticGenerator(seed=seed)
        g.set({**params, 'waveletLog': True, 'waveletFreq': (33, 208), 'faultCount': (2, 4)})
        img, msk = g.get()
        img = (img - img.min()) / (img.max() - img.min())
        rows.append((f'{name} s{seed} rótulo {msk.mean() * 100:.1f}%', img[64].T, msk[64].T))
        print(rows[-1][0], flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for k, (t, s, m) in enumerate(rows):
    r, c = k // 2, (k % 2) * 2
    for j, ax in enumerate(axes[r, c:c + 2]):
        ax.imshow(s, cmap='gray', vmin=np.percentile(s, 1), vmax=np.percentile(s, 99))
        if j == 1:
            zz, yy = np.nonzero(m); ax.scatter(yy, zz, s=0.3, c='red', alpha=0.4)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig('/home/grva-mint/Projects/Falhas/Synthetic/files/experiments/checks/subtle.png', dpi=50)
print('ok')
