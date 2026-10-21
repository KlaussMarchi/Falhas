# RESUMO DAS RODADAS NA RECEITA COMPLETA NOS 4 PATCHES: F1 NATIVO, MARCAÇÃO NO LIMIAR 0,5, F1 COM A MÁSCARA IGUALADA EM VOLTA DO
# ÓTIMO (A ORDENAÇÃO, SEM A CALIBRAÇÃO) E A RAZÃO MARCAÇÃO/RÓTULO QUE CALIBRA A DENSIDADE DA PRÓXIMA RODADA -> profile/full.csv
# uso: python full.py <tag> [<tag> ...]
import os, sys, json, glob
import numpy as np, pandas as pd
from multiprocessing import get_context

EXP   = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
OUT   = f'{EXP}/profile/full.csv'
FRACS = (0.008, 0.01, 0.0125, 0.015)
sys.path.insert(0, f'{EXP}/harness')


def measure(job):
    import marlim
    tag, pid = job
    sec = np.load(f'{EXP}/sections/{tag}_{pid}_fast.npy').astype(np.float32)
    row = {'tag': tag, 'patch': pid, 'marca': float((sec[marlim.liveMask(pid)] >= 0.5).mean())}
    row.update(marlim.stickMetrics(pid, sec)[0])
    for frac in FRACS:
        row[f'f1@{frac * 100:g}'] = marlim.stickMetrics(pid, marlim.equalize(pid, sec, frac)[0])[0]['f1']
    return row


def label(dataset):
    for folder in (f'{EXP}/datasets/{dataset}', f'/home/grva-mint/Projects/Falhas/Dataset/{dataset}'):
        if os.path.exists(f'{folder}/DataBase.csv'):
            db = pd.read_csv(f'{folder}/DataBase.csv')
            return float(np.mean([(np.load(p) > 0).mean() for p in db.mask_path.sample(min(60, len(db)), random_state=0)]))
    return np.nan


if __name__ == '__main__':
    done  = pd.read_csv(OUT, dtype={'patch': str}) if os.path.exists(OUT) else pd.DataFrame(columns=['tag', 'patch'])
    seen  = set(zip(done.tag, done.patch))
    jobs  = [(tag, pid) for tag in sys.argv[1:] for pid in ('1200', '1300', '1400', '2600') if (tag, pid) not in seen and os.path.exists(f'{EXP}/sections/{tag}_{pid}_fast.npy')]
    with get_context('fork').Pool(8) as pool:
        rows = pool.map(measure, jobs)
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(OUT, index=False)

    results  = pd.DataFrame([json.loads(line) for line in open(f'{EXP}/results.jsonl')])
    datasets = results.groupby('tag').dataset.first()
    table    = table[table.tag.isin(sys.argv[1:])]
    full     = table.groupby('tag').patch.nunique() == 4
    summary  = table[table.tag.isin(full[full].index)].groupby('tag')[['f1', 'precisao', 'recall', 'marca', 'sticks'] + [f'f1@{f * 100:g}' for f in FRACS]].mean()
    summary['ordem'] = summary[[f'f1@{f * 100:g}' for f in FRACS]].max(axis=1)
    summary['rotulo'] = [label(datasets.get(tag, '')) for tag in summary.index]
    summary['marca/rotulo'] = summary.marca / summary.rotulo
    pd.set_option('display.width', 250)
    print(summary.sort_values('f1', ascending=False).round(3).to_string())
