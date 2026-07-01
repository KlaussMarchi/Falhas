"""Worker importável para geração paralela de tiles estratigráficos.
Definir o worker num módulo (não numa célula do notebook) garante que ele seja
picklável pelo ProcessPoolExecutor tanto em 'fork' quanto em 'spawn'."""
import os
import numpy as np
from index import SyntheticGenerator


def strat_tile_worker(args):
    """args = (i, cfg_dict, seed, img_dir, msk_dir). Gera e salva 1 tile."""
    i, cfg, seed, img_dir, msk_dir = args
    np.random.seed(seed)
    g = SyntheticGenerator()
    g.set(cfg)
    img, msk = g.get()
    img = np.transpose(img, (0, 2, 1))
    msk = np.transpose(msk, (0, 2, 1))
    os.makedirs(img_dir, exist_ok=True)          # resiliente a diretório sumir mid-run
    os.makedirs(msk_dir, exist_ok=True)
    np.save(os.path.join(img_dir, f'img_{i:04d}.npy'), img)
    np.save(os.path.join(msk_dir, f'img_{i:04d}.npy'), msk)
    return i
