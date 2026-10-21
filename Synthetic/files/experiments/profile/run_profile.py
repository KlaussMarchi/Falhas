# PERFIL DOS TILES REAIS (PATCHES DE DEV) E DOS DATASETS-ÂNCORA COM A MESMA RÉGUA -> profile/<nome>.pkl
import os, sys, glob, time
import numpy as np
import pandas as pd
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler import TileRuler, BlockSampler

ROOT    = '/home/grva-mint/Projects/Falhas'
OUT     = os.path.dirname(os.path.abspath(__file__))
WATER   = 0.50263
WORKERS = 10


def realRow(task):
    folder, name = task
    sampler = BlockSampler(folder)
    d, z0, x0 = sampler.core(name)
    return {'name': name, 'z0': z0, 'x0': x0, **TileRuler(WATER).get(sampler.get(name))}


def synthRow(path):
    return {'name': os.path.basename(path), **TileRuler().get(np.load(path))}


def run(name, func, tasks):
    out = f'{OUT}/{name}.pkl'
    if os.path.exists(out):
        return print(name, 'já existe')
    t0 = time.time()
    with get_context('fork').Pool(WORKERS) as pool:
        rows = pool.map(func, tasks, chunksize=4)
    pd.DataFrame(rows).to_pickle(out)
    print(name, len(rows), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    for pid in sys.argv[1:] or ['1200', '1300']:
        if pid.isdigit():
            folder = f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles'
            run(f'marlim_{pid}', realRow, [(folder, n) for n in BlockSampler(folder).names()])
        else:
            folder = f'{ROOT}/Dataset/{pid}' if os.path.isdir(f'{ROOT}/Dataset/{pid}') else f'{ROOT}/Synthetic/files/experiments/datasets/{pid}'
            paths  = pd.read_csv(f'{folder}/DataBase.csv').img_path.tolist() if os.path.exists(f'{folder}/DataBase.csv') else sorted(glob.glob(f'{folder}/images/*.npy'))
            run(pid, synthRow, paths)
