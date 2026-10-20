# AUDITORIA VISUAL: 16 STICKS DA dbrnet DE REFERÊNCIA SEM PAR NO ESPECIALISTA (z 450-1200, PATCHES 1200 E 1300), CADA UM NUM RECORTE DA SÍSMICA
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
import marlim
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sections')
R = '/home/grva-mint/Projects/Falhas'
picked = []
for pid in ('1200', '1300'):
    sec = np.load(f'{S}/art5_dbrnet_{pid}.npy').astype(np.float32)
    _, cmp = marlim.stickMetrics(pid, sec)
    seis, gt = cmp.background, cmp.sticks('interpretado')
    loose = [s for s in cmp.sticks('rede') if cmp.covered(s, gt) < 0.2 and 450 <= (s['p0'][1] + s['p1'][1]) / 2 < 1200 and s['len'] >= 150]
    rng = np.random.default_rng(int(pid))
    for s in rng.choice(loose, min(8, len(loose)), replace=False):
        picked.append((pid, s, seis, sec))
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for ax, (pid, s, seis, sec) in zip(axes.ravel(), picked):
    cx, cz = (s['p0'][0] + s['p1'][0]) / 2, (s['p0'][1] + s['p1'][1]) / 2
    half = max(110, s['len'] / 2 + 30)
    x0, x1 = int(max(0, cx - half)), int(min(seis.shape[1], cx + half))
    z0, z1 = int(max(0, cz - half)), int(min(seis.shape[0], cz + half))
    ax.imshow(seis[z0:z1, x0:x1], cmap='gray', vmin=0, vmax=1, extent=(x0, x1, z1, z0))
    ax.plot([s['p0'][0], s['p1'][0]], [s['p0'][1], s['p1'][1]], c='red', lw=1.0, alpha=0.6)
    ax.set_title(f"{pid} z{int(cz)} x{int(cx)} len {int(s['len'])} dip {s['dip']:.0f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('audit.png', dpi=50)
print(len(picked), 'sticks')
