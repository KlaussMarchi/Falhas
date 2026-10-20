# RECORTES NATIVOS DA INLINE ANOTADA: SÍSMICA, ESPECIALISTA E A MÁSCARA DA dbrnet DE REFERÊNCIA (Artigo/model_5)
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R = '/home/grva-mint/Projects/Falhas'
index = json.load(open(f'{R}/Blocks/Marlim/files/cache/inline_index.json'))

def section(pid):
    seis = np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy')
    gt   = np.array(Image.open(f'{R}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    prob = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sections', f'art5_dbrnet_{pid}.npy'))
    return seis, gt, np.asarray(prob)

for pid, windows in {'1200': [(250, 650, 900, 1500), (600, 1000, 300, 900), (900, 1300, 1200, 1800)], '2600': [(300, 700, 1300, 1900), (700, 1100, 700, 1300), (1050, 1450, 1500, 2100)]}.items():
    seis, gt, prob = section(pid)
    fig, axes = plt.subplots(len(windows), 3, figsize=(18, 6 * len(windows)))
    for row, (z0, z1, x0, x1) in zip(axes, windows):
        s, g, p = seis[z0:z1, x0:x1], gt[z0:z1, x0:x1], prob[z0:z1, x0:x1]
        for ax in row:
            ax.imshow(s, cmap='gray', vmin=0, vmax=1, extent=(x0, x1, z1, z0))
        gz, gx = np.nonzero(g)
        row[1].scatter(gx + x0, gz + z0, s=0.3, c='lime')
        mask = np.ma.masked_where(p < 0.5, p)
        row[2].imshow(mask, cmap='autumn', vmin=0.5, vmax=1, alpha=0.6, extent=(x0, x1, z1, z0))
        row[2].scatter(gx + x0, gz + z0, s=0.2, c='cyan')
        row[0].set_title(f'patch {pid}  z {z0}-{z1}  x {x0}-{x1}  σ={s.std():.3f}')
        row[1].set_title('especialista (verde)')
        row[2].set_title(f'dbrnet ref ≥0.5 (laranja) + especialista (ciano)  máscara={float((p >= 0.5).mean()):.3f}')
    plt.tight_layout()
    plt.savefig(f'crops_{pid}.png', dpi=60)
    plt.close()
    print('ok', pid)
