# SONDA × RECEITA COMPLETA NOS DATASETS COM ÂNCORA: AS MESMAS MÉTRICAS E O ρ DE SPEARMAN ENTRE AS DUAS ORDENS. uso: python proxy.py [sonda]
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

EXP     = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
ANCHORS = {'anc_art2_wu': 'dataset_wu', 'anc_mar2_wu': 'dataset_wu', 'anc_art5_74wu': 'dataset_74_wu', 'anc_mar3_74wu440': 'dataset_74_wu', 'anc_mar1_74': 'dataset_74', 'anc_mar5_74mar': 'mix_74_mar', 'anc_mar6_wumar': 'mix_wu_mar'}
METRICS = ['f1', 'f1@1', 'f1@2', 'f1_curva', 'auc_perto', 'ap_perto', 'pp@1', 'pf@1', 'pf@2', 'pf@4', 'mask_frac']


# SÓ AS RODADAS MEDIDAS EM TODOS OS PATCHES PEDIDOS (PATCHES=1200,1300 NO AMBIENTE ESCOLHE OUTRO CONJUNTO)
def complete(rows):
    wanted = os.environ.get('PATCHES', '1200,1300,1400,2600').split(',')
    rows   = rows[rows.patch.astype(str).isin(wanted)]
    full   = rows.groupby('tag').patch.nunique()
    return rows[rows.tag.isin(full[full == len(wanted)].index)]


def table():
    rows   = pd.DataFrame([json.loads(l) for l in open(f'{EXP}/results.jsonl')])
    rows   = complete(rows[rows.metodo == 'rapido'])[['tag', 'patch', 'dataset', 'f1', 'f1@1', 'f1@2', 'mask_frac']]
    curve  = pd.read_csv(f'{EXP}/profile/f1curve.csv', dtype={'patch': str})[['tag', 'patch', 'f1_curva']]
    smooth = pd.read_csv(f'{EXP}/profile/smooth.csv', dtype={'patch': str})
    rows   = rows.merge(curve, on=['tag', 'patch'], how='left').merge(smooth, on=['tag', 'patch'], how='left')
    return rows.groupby('tag').agg({**{m: 'mean' for m in METRICS}, 'dataset': 'first'})


if __name__ == '__main__':
    probe = sys.argv[1] if len(sys.argv) > 1 else 'Q20'
    per   = table()
    full  = per.loc[[t for t in ANCHORS if t in per.index]].assign(dataset=lambda t: t.index.map(ANCHORS)).groupby('dataset')[METRICS].mean()
    short = per[per.index.str.startswith(probe + '_')].groupby('dataset')[METRICS].mean()
    keys  = full.index.intersection(short.index)
    pd.set_option('display.width', 250)
    print(f'{len(keys)} datasets com âncora e sonda {probe}: {list(keys)}')
    print(pd.concat({'completa': full.loc[keys], probe: short.loc[keys]}, axis=1).round(3).T.to_string())
    print()
    print(pd.Series({m: round(spearmanr(full.loc[keys, m], short.loc[keys, m]).correlation, 2) if len(keys) >= 3 else np.nan for m in METRICS}, name='ρ completa × sonda').to_string())
