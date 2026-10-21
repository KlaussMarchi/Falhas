# FIGURA DE COMPARAÇÃO: AS SEÇÕES PREDITAS (sections/<tag>_<patch>_fast.npy) DE VÁRIAS RODADAS NOS MESMOS RECORTES DA INLINE
# ANOTADA, COM O ESPECIALISTA POR CIMA. uso: python compare_sections.py <patch> <saída.png> <tag> [<tag> ...]
# COM TOP=<fração> NO AMBIENTE, MOSTRA O q MAIS PROVÁVEL DA SEÇÃO VIVA (O PONTO DE OPERAÇÃO IGUALADO) NO LUGAR DO LIMIAR 0,5
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT  = '/home/grva-mint/Projects/Falhas'
EXP   = f'{ROOT}/Synthetic/files/experiments'
CROPS = {'1200': ((600, 1000, 300, 900), (900, 1300, 1200, 1800)), '1300': ((600, 1000, 400, 1000), (850, 1250, 1500, 2100))}


def show(pid, out, tags):
    seismic = np.load(f'{ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
    gt      = np.array(Image.open(f'{ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    crops   = CROPS[pid]
    fig, axes = plt.subplots(len(crops), len(tags), figsize=(5 * len(tags), 4.2 * len(crops)), squeeze=False)
    for c, tag in enumerate(tags):
        sec = np.load(f'{EXP}/sections/{tag}_{pid}_fast.npy').astype(np.float32)
        if os.environ.get('TOP'):
            live = np.load(f'{EXP}/cache/live_{pid}.npy') if os.path.exists(f'{EXP}/cache/live_{pid}.npy') else np.ones(sec.shape, bool)
            sec  = (sec >= np.quantile(sec[live], 1 - float(os.environ['TOP']))).astype(np.float32) * live
        for r, (z0, z1, x0, x1) in enumerate(crops):
            ax = axes[r, c]
            ax.imshow(seismic[z0:z1, x0:x1], cmap='gray', vmin=0.2, vmax=0.8, extent=(x0, x1, z1, z0))
            ax.imshow(np.ma.masked_less(sec[z0:z1, x0:x1], 0.5), cmap='autumn', alpha=0.7, vmin=0.5, vmax=1, extent=(x0, x1, z1, z0))
            ax.contour(np.arange(x0, x1), np.arange(z0, z1), gt[z0:z1, x0:x1], levels=[0.5], colors='cyan', linewidths=0.6)
            ax.set_title(f'{tag} — patch {pid} z {z0}-{z1}, máscara {(sec >= 0.5).mean():.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=60)


if __name__ == '__main__':
    show(sys.argv[1], sys.argv[2], sys.argv[3:])
