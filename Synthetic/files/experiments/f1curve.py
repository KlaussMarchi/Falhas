# F1 DE STICKS COM A MÁSCARA IGUALADA EM VÁRIAS FRAÇÕES DA SEÇÃO VIVA (1–5%), CALCULADO DEPOIS DAS SEÇÕES SALVAS (sections/<tag>_<pid>_fast.npy):
# A SONDA CURTA MARCA ~10% E A RECEITA COMPLETA 2,6–5%, ENTÃO A COMPARAÇÃO JUSTA É NO MESMO PONTO DE OPERAÇÃO; A MÉDIA DAS FRAÇÕES (f1_curva)
# TIRA O ACASO DE UMA FRAÇÃO SÓ. MESMO EXTRATOR E MESMO COMPARADOR DO NOTEBOOK.
# uso: python f1curve.py   -> profile/f1curve.csv (uma linha por tag × patch)
import os, sys, glob, re
import numpy as np, pandas as pd
from multiprocessing import get_context

EXP   = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
OUT   = f'{EXP}/profile/f1curve.csv'
FRACS = (0.01, 0.02, 0.03, 0.04, 0.05)
sys.path.insert(0, f'{EXP}/harness')


def measure(path):
    import marlim
    tag, pid = re.match(r'(.+)_(\d{4})_fast\.npy', os.path.basename(path)).groups()
    sec = np.load(path).astype(np.float32)
    row = {'tag': tag, 'patch': pid}
    for frac in FRACS:
        eq, _ = marlim.equalize(pid, sec, frac)
        m, _  = marlim.stickMetrics(pid, eq)
        row.update({f'f1@{frac * 100:g}': m['f1'], f'p@{frac * 100:g}': m['precisao'], f'r@{frac * 100:g}': m['recall']})
    row['f1_curva'] = float(np.mean([row[f'f1@{f * 100:g}'] for f in FRACS]))
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
    print(table.groupby('tag')[[c for c in table if c.startswith('f1')]].mean().round(3).to_string())
