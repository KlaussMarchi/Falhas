# GAP TABLE DA FASE B: DESCRITORES DOS TILES REAIS (DEV, POR CLASSE E FAIXA DE z) CONTRA OS DATASETS-ÂNCORA
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
BANDS = [(0, 450), (450, 700), (700, 950), (950, 1200), (1200, 1601)]
real = pd.concat([pd.read_pickle(f'marlim_{p}.pkl').assign(patch=p) for p in ('1200', '1300')], ignore_index=True)
real['zc'] = real.z0 + 32
real['band'] = pd.cut(real.zc, [b[0] for b in BANDS] + [1601], right=False, labels=[f'{a}-{b}' for a, b in BANDS])
real['cls'] = np.select([real.water > 0.5, real.live < 0.1, real.live > 0.9], ['água', 'morto', 'refletivo'], 'misto')
print('COMPOSIÇÃO DOS TILES REAIS (dev, 1820 tiles):')
print(pd.crosstab(real.band, real.cls, margins=True))
cols = ['sigma', 'sat', 'fine', 'period', 'bandwidth', 'coherence', 'dipP50', 'dipP90', 'dipP90_il', 'dipP90_xl', 'disc5', 'disc17', 'disc5_il', 'disc5_xl', 'steep90', 'steep90_il', 'steep90_xl', 'flat90', 'lag4', 'lag16', 'kurt', 'trend']
syn = {n: pd.read_pickle(f'{n}.pkl') for n in ('dataset_74', 'dataset_wu', 'marlim_opt', 'dataset_regions')}
q = lambda s: f'{s.quantile(0.1):.3g}/{s.median():.3g}/{s.quantile(0.9):.3g}'
rows = {}
live = real[real.live > 0.5]
for band, g in live.groupby('band', observed=True):
    rows[f'real {band} (n={len(g)})'] = {c: q(g[c].dropna()) for c in cols}
for n, df in syn.items():
    rows[n] = {c: q(df[c].dropna()) for c in cols}
print('\nDESCRITORES p10/p50/p90 (real: tiles com live > 0.5)')
print(pd.DataFrame(rows).to_string())
print('\nCOBERTURA: fração dos tiles reais (live>0.5) dentro do [p2.5, p97.5] do sintético')
cov = {}
for n, df in syn.items():
    cov[n] = {c: f'{((live[c] >= df[c].quantile(0.025)) & (live[c] <= df[c].quantile(0.975))).mean():.2f}' for c in cols}
union = pd.concat(syn.values())
cov['união 4'] = {c: f'{((live[c] >= union[c].quantile(0.025)) & (live[c] <= union[c].quantile(0.975))).mean():.2f}' for c in cols}
print(pd.DataFrame(cov).T.to_string())
