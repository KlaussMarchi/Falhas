# MEDIDAS SUAVES DE TSTR, CALCULADAS DAS SEÇÕES SALVAS (sections/<tag>_<pid>_fast.npy): EM VEZ DOS STICKS (NÃO LINEARES, LIMIAR 0,5 E EXTRATOR),
# O RANKING DOS PIXELS VIVOS PELA PROBABILIDADE CONTRA A INTERPRETAÇÃO — AUC E AP DE "PERTO DE FALHA DO ESPECIALISTA" (≤ NEAR px DO ESQUELETO) CONTRA
# "LONGE" (> FAR px), E A PRECISÃO/COBERTURA/F DE PIXEL COM A MÁSCARA IGUALADA EM q% DA SEÇÃO VIVA (O MESMO PONTO DE OPERAÇÃO DO F1@q).
# A ANOTAÇÃO É PARCIAL: FALHA REAL NÃO ANOTADA CONTA COMO "LONGE" PARA TODO MODELO IGUAL. top2_<faixa>: FRAÇÃO DO 2% MAIS PROVÁVEL EM CADA FAIXA DE z (SEM RÓTULO).
# uso: python smooth.py   -> profile/smooth.csv (uma linha por tag × patch)
import os, sys, glob, re
import numpy as np, pandas as pd
import scipy.ndimage as ndi
from multiprocessing import get_context

EXP   = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
OUT   = f'{EXP}/profile/smooth.csv'
NEAR  = 8
FAR   = 30
TOL   = 12
FRACS = (0.01, 0.02, 0.04)
BANDS = {'z<450': (0, 450), 'z450-700': (450, 700), 'z700-950': (700, 950), 'z950-1200': (950, 1200), 'z>1200': (1200, 10 ** 6)}    # ONDE CAI O 2% MAIS PROVÁVEL
sys.path.insert(0, f'{EXP}/harness')


def reference(pid):
    from PIL import Image
    from skimage.morphology import skeletonize
    import marlim
    gt   = np.array(Image.open(f'{marlim.ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    sk   = skeletonize(gt)
    return sk, ndi.distance_transform_edt(~sk), marlim.liveMask(pid)


def ranking(pos, neg):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def measure(path):
    tag, pid = re.match(r'(.+)_(\d{4})_fast\.npy', os.path.basename(path)).groups()
    sec      = np.load(path).astype(np.float32)
    sk, dist, live = reference(pid)
    auc, ap  = ranking(sec[live & (dist <= NEAR)], sec[live & (dist > FAR)][::7])
    row      = {'tag': tag, 'patch': pid, 'auc_perto': auc, 'ap_perto': ap}
    top = (sec >= np.quantile(sec[live], 0.98)) & live
    z   = np.nonzero(top)[0]
    row.update({f'top2_{name}': float(((z >= lo) & (z < hi)).mean()) for name, (lo, hi) in BANDS.items()})
    for frac in FRACS:
        mask   = (sec >= np.quantile(sec[live], 1 - frac)) & live
        near   = ndi.distance_transform_edt(~mask)
        p      = float((dist[mask] <= TOL).mean()) if mask.any() else 0.0
        r      = float((near[sk] <= TOL).mean())
        q      = f'{frac * 100:g}'
        row.update({f'pp@{q}': p, f'pr@{q}': r, f'pf@{q}': 2 * p * r / (p + r + 1e-9)})
    return row


if __name__ == '__main__':
    done  = pd.read_csv(OUT, dtype={'patch': str}) if os.path.exists(OUT) else pd.DataFrame(columns=['tag', 'patch'])
    seen  = set(zip(done.tag, done.patch.astype(str)))
    paths = [p for p in sorted(glob.glob(f'{EXP}/sections/*_fast.npy')) if tuple(re.match(r'(.+)_(\d{4})_fast\.npy', os.path.basename(p)).groups()) not in seen]
    with get_context('fork').Pool(6) as pool:
        rows = pool.map(measure, paths)
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(table.groupby('tag')[[c for c in table if c not in ('tag', 'patch')]].mean().round(3).to_string())
