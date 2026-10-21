# DATASET DE EXPERIMENTO: SyntheticGenerator.dataset() + O Format 'normalize' (MIN-MAX POR VOLUME), SEM TOCAR NO Task/info.json
# uso: python make_dataset.py <tag>   (as opções vêm de datasets/specs.json[tag]; grava datasets/<tag>/{images,masks,DataBase.csv,synthetic.json})
import os, sys, json, glob, shutil, time
import numpy as np
import pandas as pd
import torch

ROOT = '/home/grva-mint/Projects/Falhas'
EXP  = f'{ROOT}/Synthetic/files/experiments'
sys.path.insert(0, ROOT)
from Synthetic.index import SyntheticGenerator
from Dataset.index import Normalization


def main(tag):
    spec    = json.load(open(f'{EXP}/datasets/specs.json'))[tag]
    folder  = f'{EXP}/datasets/{tag}'
    options = {'directory': f'{folder}/original', 'seed': spec['seed'], 'regions': spec['regions']}

    if os.path.exists(f'{folder}/DataBase.csv'):
        return print(tag, 'já existe')

    torch.set_num_threads(int(os.environ.get('THREADS', 8)))
    t0 = time.time()
    SyntheticGenerator().dataset(options)
    json.dump({**options, 'directory': 'original'}, open(f'{folder}/synthetic.json', 'w'), indent=4)
    print(tag, f'gerado em {time.time() - t0:.0f}s', flush=True)
    format(folder)


# O Format.ipynb DO dataset_74 NO MODO 'normalize': MIN-MAX DE CADA VOLUME EM float32, MÁSCARA COMO SAIU, DataBase.csv COM CAMINHO ABSOLUTO
def format(folder):
    normalize = Normalization('normalize')
    os.makedirs(f'{folder}/images', exist_ok=True)
    os.makedirs(f'{folder}/masks', exist_ok=True)
    rows = []

    for img_path in sorted(glob.glob(f'{folder}/original/*/images/*.npy')):
        name = os.path.basename(img_path)
        msk  = np.load(img_path.replace('/images/', '/masks/'))
        img  = normalize(np.load(img_path))
        np.save(f'{folder}/images/{name}', img)
        np.save(f'{folder}/masks/{name}', msk)
        rows.append({'id': name, 'img_min': np.min(img), 'img_max': np.max(img), 'img_mean': np.mean(img), 'img_std': np.std(img), 'msk_min': np.min(msk), 'msk_max': np.max(msk), 'msk_frac': float((msk > 0).mean()), 'shape': img.shape, 'img_path': f'{folder}/images/{name}', 'mask_path': f'{folder}/masks/{name}', 'scaling': 'normalize'})

    pd.DataFrame(rows).to_csv(f'{folder}/DataBase.csv', index=None)
    shutil.rmtree(f'{folder}/original')
    print(os.path.basename(folder), len(rows), 'tiles formatados, máscara média', round(float(np.mean([r['msk_frac'] for r in rows])), 4), flush=True)


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        main(tag)
