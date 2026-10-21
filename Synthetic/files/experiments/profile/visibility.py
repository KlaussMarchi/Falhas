# VISIBILIDADE DO RÓTULO SINTÉTICO (TRAVA DE APRENDIBILIDADE DO OBJETIVO, SEM RÓTULO REAL): EM CADA SEÇÃO, A AUC DA
# DESCONTINUIDADE (1 - SEMBLANCE DE 5 TRAÇOS) ENTRE O PIXEL ROTULADO E O PIXEL VIVO LONGE DO RÓTULO. 0,5 = RÓTULO INVISÍVEL.
# uso: python visibility.py <dataset...>   -> profile/visibility.csv (uma linha por dataset: mediana e quantis por tile)
import os, sys, glob
import numpy as np, pandas as pd
import scipy.ndimage as ndi
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler import TileRuler, gauss

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = '/home/grva-mint/Projects/Falhas'
SLICES = (32, 64, 96)
FAR    = 6    # PIXEL "LONGE DO RÓTULO" FICA A MAIS DISSO DE QUALQUER VOXEL ROTULADO


def auc(pos, neg):
    if len(pos) < 20 or len(neg) < 20:
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tile(paths):
    img, msk = np.load(paths[0]).astype(np.float32), np.load(paths[1]) > 0
    values   = []
    for i in SLICES:
        sec, lab = img[i], msk[i]
        if lab.sum() < 20:
            continue
        a    = sec - np.median(sec)
        live = np.sqrt(gauss(a * a, 12)) > TileRuler.ALIVE
        zs   = a / (float(a[live].std()) + 1e-8) if live.any() else a
        disc = 1 - TileRuler.semblance(zs, 2)
        far  = ndi.distance_transform_edt(~lab) > FAR
        values.append(auc(disc[lab & live], disc[far & live]))
    return {'name': os.path.basename(paths[0]), 'vis': float(np.nanmean(values)) if values else np.nan, 'label': float(msk.mean())}


def folder(tag):
    for base in (f'{ROOT}/Dataset/{tag}', f'{ROOT}/Synthetic/files/experiments/datasets/{tag}'):
        if os.path.exists(f'{base}/DataBase.csv'):
            return base


if __name__ == '__main__':
    out  = f'{HERE}/visibility.csv'
    done = pd.read_csv(out, index_col=0) if os.path.exists(out) else pd.DataFrame()
    for tag in sys.argv[1:]:
        db    = pd.read_csv(f'{folder(tag)}/DataBase.csv')
        tasks = list(zip(db.img_path, db.mask_path))
        with get_context('fork').Pool(8) as pool:
            rows = pd.DataFrame(pool.map(tile, tasks, chunksize=4))
        v = rows.vis.dropna()
        done.loc[tag, ['vis_p10', 'vis_med', 'vis_p90', 'sem_rotulo', 'rotulo']] = [v.quantile(0.1), v.median(), v.quantile(0.9), float(rows.vis.isna().mean()), rows.label.mean()]
        print(tag, done.loc[tag].astype(float).round(3).to_dict(), flush=True)
    done.to_csv(out)
