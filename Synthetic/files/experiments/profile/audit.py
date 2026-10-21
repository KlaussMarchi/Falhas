# AUDITORIA VISUAL DOS STICKS DA dbrnet DE REFERÊNCIA SEM PAR NO ESPECIALISTA (DEV: 1200 E 1300), 12 POR FIGURA, RECORTE NATIVO
import pickle, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = '/home/grva-mint/Projects/Falhas'
store = pickle.load(open('profile/art5_sticks_dev.pkl', 'rb'))
table = pd.read_pickle('profile/art5_sticks_dev_table.pkl')
loose = table[table['cov'] < 0.2].sample(frac=1, random_state=0).reset_index()
seis  = {pid: np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) for pid in ('1200', '1300')}
for f0 in range(0, 48, 6):
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    for ax, (k, row) in zip(axes.ravel(), loose.iloc[f0:f0 + 6].iterrows()):
        s = store[row.patch]['mine'][row['index'] - table.index[table.patch == row.patch][0]]
        S = seis[row.patch]
        cx, cz = row.xc, row.zc
        half = max(110, s['len'] / 2 + 40)
        x0, x1 = int(max(0, cx - half)), int(min(S.shape[1], cx + half))
        z0, z1 = int(max(0, cz - half)), int(min(S.shape[0], cz + half))
        crop = S[z0:z1, x0:x1]
        lo, hi = np.percentile(crop, [1, 99])
        ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi, extent=(x0, x1, z1, z0), interpolation='none')
        p0, p1 = np.array(s['p0']), np.array(s['p1'])
        n = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (np.linalg.norm(p1 - p0) + 1e-9) * 14
        for sgn in (1, -1):
            ax.plot([p0[0] + sgn * n[0], p1[0] + sgn * n[0]], [p0[1] + sgn * n[1], p1[1] + sgn * n[1]], c='red', lw=1.0, alpha=0.7)
        for g in store[row.patch]['gt']:
            ax.plot([g['p0'][0], g['p1'][0]], [g['p0'][1], g['p1'][1]], c='lime', lw=0.8, ls=':', alpha=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)
        ax.set_title(f"#{f0 + list(loose.index[f0:f0+6]).index(k)} {row.patch} z{int(cz)} x{int(cx)} len{int(s['len'])} dip{s['dip']:.0f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(f'profile/audit/audit6_{f0 // 6:02d}.png', dpi=60); plt.close()
loose[['patch', 'zc', 'xc', 'len', 'dip', 'cov']].to_csv('profile/audit/sticks.csv')
print(len(loose), 'sticks em', (len(loose) + 11) // 12, 'figuras')
