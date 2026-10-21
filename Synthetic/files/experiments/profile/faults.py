# PERFIL DAS FALHAS: MERGULHO, COMPRIMENTO, ESPAÇAMENTO, DENSIDADE E REJEITO/PERÍODO DOS STICKS DO ESPECIALISTA NOS 4 PATCHES E DOS RÓTULOS DOS DATASETS
# uso: python faults.py [especialista | modelo:<tag> | iracema:<tag>:<inline> | dataset ...] -> profile/faults_<nome>.pkl
# modelo e iracema = STICKS DA MÁSCARA IGUALADA EM 1% DE UM MODELO DE BASE (SEM RÓTULO, PORTÁVEL)
import os, sys, glob
import numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'harness'))
import marlim
from offset import slope, evidence

HERE    = os.path.dirname(os.path.abspath(__file__))
PATCHES = ('1200', '1300', '1400', '2600')
WATER   = 0.50263


def measure(section, dip, sticks):
    rows = []
    for s in sticks:
        (x0, z0), (x1, z1) = s['p0'], s['p1']
        ev  = evidence(section, dip, s)
        hit = ev[(ev.rho > 0.6) & (ev.rho - ev.rho0 > 0.1)] if len(ev) else ev
        rows.append({'zc': (z0 + z1) / 2, 'xc': (x0 + x1) / 2, 'len': s['len'], 'dip': float(np.degrees(np.arctan2(abs(z1 - z0), abs(x1 - x0) + 1e-9))), 'rejeito': float(hit.lag.abs().median()) if len(hit) else np.nan, 'periodo': float(ev.per.median()) if len(ev) else np.nan, 'visivel': float(len(hit) / max(len(ev), 1)), 'pontos': len(ev), 'acertos': len(hit)})
    table = pd.DataFrame(rows)
    if len(table):
        table['razao'] = table.rejeito / table.periodo
        table['espaco'] = [spacing(table, i) for i in range(len(table))]
    return table


def spacing(table, i, band=20):
    row  = table.iloc[i]
    near = table[(table.index != table.index[i]) & ((table.zc - row.zc).abs() < band)]
    return float((near.xc - row.xc).abs().min()) if len(near) else np.nan


def expert():
    out = []
    for pid in PATCHES:
        section = np.load(f'{marlim.ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) - WATER
        raster  = np.array(Image.open(f'{marlim.ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
        sticks  = marlim.NB['FaultStickExtractor'].fromRaster(raster)
        table   = measure(section, slope(section), sticks).assign(patch=pid)
        live    = marlim.liveMask(pid)
        table.attrs[pid] = float(table.len.sum() / live.sum())
        out.append(table.assign(densidade=table.len.sum() / live.sum()))
    return pd.concat(out, ignore_index=True)


def model(tag, frac=0.01):
    out = []
    for pid in PATCHES:
        prob    = np.load(f'{marlim.ROOT}/Synthetic/files/experiments/sections/{tag}_{pid}_fast.npy').astype(np.float32)
        seismic = np.load(f'{marlim.ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
        sticks  = marlim.NB['FaultStickExtractor'](0.80, 140, 59).extract(marlim.equalize(pid, prob, frac)[0], seismic=seismic)
        section = seismic - WATER
        table   = measure(section, slope(section), sticks).assign(patch=pid)
        out.append(table.assign(densidade=table.len.sum() / marlim.liveMask(pid).sum()))
    return pd.concat(out, ignore_index=True)


def iracema(tag, inline, frac=0.01, water=0.49599):
    import iracema as block
    prob    = np.load(f'{HERE}/../sections/{tag}_iracema{inline}.npy').astype(np.float32)
    seismic = np.load(f'{HERE}/../sections/iracema{inline}_seismic.npy')
    live    = block.liveMask(seismic)
    sticks  = marlim.NB['FaultStickExtractor'](0.80, 140, 59).extract(block.equalize(prob, live, frac), seismic=seismic)
    table   = measure(seismic - water, slope(seismic - water), sticks)
    return table.assign(densidade=table.len.sum() / live.sum())


def synthetic(name, nTiles=60, seed=0):
    folder = f'{marlim.ROOT}/Dataset/{name}' if os.path.isdir(f'{marlim.ROOT}/Dataset/{name}') else f'{marlim.ROOT}/Synthetic/files/experiments/datasets/{name}'
    db     = pd.read_csv(f'{folder}/DataBase.csv').sample(nTiles, random_state=seed)
    out    = []
    for _, row in db.iterrows():
        img, msk = np.load(row.img_path).astype(np.float32), np.load(row.mask_path) > 0
        for x in (32, 64, 96):
            section = img[x] - np.median(img[x])
            sticks  = marlim.NB['FaultStickExtractor'].fromRaster(msk[x])
            table   = measure(section, slope(section), sticks)
            out.append(table.assign(tile=row.id, densidade=table.len.sum() / section.size if len(table) else 0.0))
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['especialista']:
        kind  = name.split(':')
        table = expert() if name == 'especialista' else model(kind[1]) if kind[0] == 'modelo' else iracema(kind[1], int(kind[2])) if kind[0] == 'iracema' else synthetic(name)
        name  = name.replace(':', '_')
        table.to_pickle(f'{HERE}/faults_{name}.pkl')
        print(name, len(table), 'sticks')
        print(table[['len', 'dip', 'rejeito', 'periodo', 'razao', 'visivel', 'espaco']].describe(percentiles=[0.1, 0.5, 0.9]).round(2).loc[['count', '10%', '50%', '90%']].to_string())
