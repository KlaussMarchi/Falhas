# PERFIL DO IRACEMA COM A MESMA RÉGUA: 1500 TILES SORTEADOS DO BLOCO INTEIRO (A ÁGUA DELE É -vmin/(vmax-vmin) = 0.495994)
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler import TileRuler, BlockSampler
ROOT   = '/home/grva-mint/Projects/Falhas'
FOLDER = f'{ROOT}/Blocks/Iracema/files/patches'
META   = f'{ROOT}/Blocks/Iracema/files/patch_metadata.json'
info   = json.load(open(META))
WATER  = -info['vmin'] / (info['vmax'] - info['vmin'])

def row(name):
    s = BlockSampler(FOLDER, META)
    d, z0, x0 = s.core(name)
    return {'name': name, 'd0': d, 'z0': z0, 'x0': x0, **TileRuler(WATER).get(s.get(name))}

if __name__ == '__main__':
    names = BlockSampler(FOLDER, META).names()
    pick  = list(np.random.default_rng(0).choice(names, 1500, replace=False))
    t0 = time.time()
    with get_context('fork').Pool(10) as pool:
        rows = pool.map(row, pick, chunksize=4)
    pd.DataFrame(rows).to_pickle('iracema_1500.pkl')
    print(len(rows), f'{time.time() - t0:.0f}s', 'água', WATER)
