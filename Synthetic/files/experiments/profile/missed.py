# FALHAS DO ESPECIALISTA QUE A dbrnet DE REFERÊNCIA PERDE (COBERTURA < 0,5) EM z 450-1200 NOS PATCHES DE DEV
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = '/home/grva-mint/Projects/Falhas'
store = pickle.load(open('profile/art5_sticks_dev.pkl', 'rb'))
df = pd.read_pickle('profile/art5_gt_dev_table.pkl').reset_index()
miss = df[(df['cov'] < 0.5) & (df.zc >= 450) & (df.zc < 1200)]
seis = {pid: np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) for pid in ('1200', '1300')}
prob = {pid: np.load(f'analise_19-10/sections/art5_dbrnet_{pid}.npy').astype(np.float32) for pid in ('1200', '1300')}
for f, start in enumerate(range(0, len(miss), 6)):
    fig, axes = plt.subplots(2, 6, figsize=(30, 11))
    for col, (_, row) in enumerate(miss.iloc[start:start + 6].iterrows()):
        k = row['index'] - df.index[df.patch == row.patch][0]
        g = store[row.patch]['gt'][k]
        S, P = seis[row.patch], prob[row.patch]
        half = max(110, g['len'] / 2 + 40)
        x0, x1 = int(max(0, row.xc - half)), int(min(S.shape[1], row.xc + half))
        z0, z1 = int(max(0, row.zc - half)), int(min(S.shape[0], row.zc + half))
        crop = S[z0:z1, x0:x1]; lo, hi = np.percentile(crop, [1, 99])
        for r, ax in enumerate(axes[:, col]):
            ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi, extent=(x0, x1, z1, z0), interpolation='none')
            p0, p1 = np.array(g['p0']), np.array(g['p1'])
            n = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (np.linalg.norm(p1 - p0) + 1e-9) * 14
            for sgn in (1, -1):
                ax.plot([p0[0] + sgn * n[0], p1[0] + sgn * n[0]], [p0[1] + sgn * n[1], p1[1] + sgn * n[1]], c='lime', lw=1.0, alpha=0.8)
            if r == 1:
                ax.imshow(np.ma.masked_less(P[z0:z1, x0:x1], 0.5), cmap='autumn', vmin=0.5, vmax=1, alpha=0.6, extent=(x0, x1, z1, z0))
            ax.set_xlim(x0, x1); ax.set_ylim(z1, z0); ax.set_xticks([]); ax.set_yticks([])
        axes[0, col].set_title(f"#{start + col} {row.patch} z{int(row.zc)} x{int(row.xc)} len{int(g['len'])} dip{g['dip']:.0f} cov{row['cov']:.2f}", fontsize=12)
    plt.tight_layout(); plt.savefig(f'profile/audit/missed_{f}.png', dpi=50); plt.close()
print(len(miss))
