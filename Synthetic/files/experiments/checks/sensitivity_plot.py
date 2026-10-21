# FIGURA DA SONDA DE SENSIBILIDADE: SEÇÃO (z × x) DE UM TILE DE CADA CASO COM A PROBABILIDADE ≥ 0,5 DA dbrnet DE REFERÊNCIA
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments/checks'
df = pd.read_csv(f'{D}/sensitivity.csv')
cases = list(dict.fromkeys(df.caso))
fig, axes = plt.subplots(2, len(cases), figsize=(4 * len(cases), 8.5))
for j, case in enumerate(cases):
    k = int(df.index[df.caso == case][0])
    img, prob, msk = np.load(f'{D}/sens_{k:02d}.npy')
    for i, ax in enumerate(axes[:, j]):
        ax.imshow(img.T, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        if i == 1:
            ax.imshow(np.ma.masked_less(prob.T, 0.5), cmap='autumn', vmin=0.5, vmax=1, alpha=0.7)
            zz, xx = np.nonzero(msk.T)
            ax.scatter(xx, zz, s=0.2, c='cyan', alpha=0.4)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, j].set_title(f"{case}\nmarcado {df[df.caso == case].marcado.mean() * 100:.2f}% do volume", fontsize=10)
plt.tight_layout(); plt.savefig(f'{D}/sensitivity.png', dpi=50)
print('ok')
