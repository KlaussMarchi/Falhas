# META-VALIDAÇÃO (M2): ORDENA OS DATASETS PELO TSTR MEDIDO (MÉDIA DOS PATCHES DE DEV E DAS SEMENTES, POR SONDA) E COMPARA COM
# CADA OBJETIVO SEM RÓTULO CANDIDATO: ρ DE SPEARMAN COM INTERVALO POR BOOTSTRAP (REAMOSTRANDO DATASETS) E A FRAÇÃO DOS PARES QUE
# DIFEREM ALÉM DO RUÍDO E ESTÃO NA ORDEM CERTA.
# uso: python meta.py [sonda: Q20|P20] [métrica: f1|f1@1]
#   E1 = profile/objective_e1.csv (régua × tiles reais, sem treino); trava = profile/visibility.csv (AUC do rótulo sintético)
#   E2 = estatísticas SEM RÓTULO da predição da própria sonda no Marlim (máscara, fragmentos, horizontal, água, borda) e a bateria
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

EXP     = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
FREE    = ['mask_frac', 'water_frac', 'frag_curtos', 'horizontal', 'no_mergulho', 'paralelo_camada', 'borda_x0', 'fragmentos']
TRAVA   = 0.75    # MEDIANA DA AUC DE VISIBILIDADE DO RÓTULO ABAIXO DISSO PERDE O QUE FALTA


# SÓ AS RODADAS MEDIDAS EM TODOS OS PATCHES PEDIDOS (PATCHES=1200,1300 NO AMBIENTE ESCOLHE OUTRO CONJUNTO)
def complete(rows):
    wanted = os.environ.get('PATCHES', '1200,1300,1400,2600').split(',')
    rows   = rows[rows.patch.astype(str).isin(wanted)]
    full   = rows.groupby('tag').patch.nunique()
    return rows[rows.tag.isin(full[full == len(wanted)].index)]


def runs(probe):
    rows = pd.DataFrame([json.loads(l) for l in open(f'{EXP}/results.jsonl')])
    rows = rows[rows.tag.str.startswith(probe + '_')]
    return rows


# TSTR POR DATASET: MÉDIA DOS PATCHES EM CADA RODADA, DEPOIS MÉDIA E DESVIO ENTRE AS RODADAS (SEMENTES)
def tstr(probe, metric):
    rows = runs(probe)
    rows = complete(rows[rows.metodo == 'rapido'])
    # AS MÉTRICAS DO f1curve.py (F1 COM A MÁSCARA IGUALADA EM 1–5% E A MÉDIA f1_curva) VÊM DO CSV, CALCULADAS DAS SEÇÕES SALVAS
    if metric not in rows or metric in ('f1@3', 'f1@4', 'f1@5', 'f1_curva'):
        source = 'smooth' if metric.startswith(('pp@', 'pr@', 'pf@', 'auc_', 'ap_')) else 'f1curve'
        curve  = pd.read_csv(f'{EXP}/profile/{source}.csv', dtype={'patch': str})[['tag', 'patch', metric]]
        rows   = rows.drop(columns=[metric], errors='ignore').merge(curve, on=['tag', 'patch'], how='inner')
    per  = rows.groupby(['dataset', 'tag'])[[metric] + FREE].mean().reset_index()
    out  = per.groupby('dataset')[[metric] + FREE].mean()
    out['sd']    = per.groupby('dataset')[metric].std()
    out['n']     = per.groupby('dataset')[metric].count()
    return out


def battery(probe):
    rows = runs(probe)
    bat  = rows[rows.metodo == 'bateria'].dropna(axis=1, how='all')
    if not len(bat):
        return pd.DataFrame()
    tags = runs(probe)[['tag', 'dataset']].dropna().drop_duplicates().set_index('tag').dataset
    bat  = bat.assign(dataset=bat.tag.map(tags))
    cols = [c for c in bat if c.startswith('bat_') and c not in ('bat_falhas', 'bat_recall_falhas')]
    out  = bat.groupby('dataset')[cols + ['bat_recall_falhas']].mean()
    out['bat_fp'] = out[cols].mean(axis=1)
    return out


def injection(probe):
    rows = runs(probe)
    inj  = rows[rows.metodo == 'injecao'].dropna(axis=1, how='all')
    if not len(inj):
        return pd.DataFrame()
    tags = rows[['tag', 'dataset']].dropna().drop_duplicates().set_index('tag').dataset
    return inj.assign(dataset=inj.tag.map(tags)).groupby('dataset')[[c for c in inj if c.startswith('inj_')]].mean()


# E2+ (e2plus.py): STICKS SEM RÓTULO DA MÁSCARA IGUALADA EM 1% NA SEÇÃO REAL, MÉDIA DOS PATCHES E DAS SEMENTES POR DATASET
def e2plus(probe):
    import os
    path = f'{EXP}/profile/e2plus.csv'
    if not os.path.exists(path):
        return pd.DataFrame()
    e2p  = pd.read_csv(path)
    tags = runs(probe)[['tag', 'dataset']].dropna().drop_duplicates().set_index('tag').dataset
    e2p  = e2p[e2p.tag.isin(tags.index)]
    return e2p.assign(dataset=e2p.tag.map(tags)).groupby('dataset')[[c for c in e2p if c.startswith('e2_')]].mean()


# O RUÍDO DE SEMENTE: DESVIO MÉDIO ENTRE RODADAS DO MESMO DATASET (SÓ OS QUE TÊM ≥ 2)
def noise(table):
    sd = table.loc[table.n >= 2, 'sd']
    return float(np.sqrt((sd ** 2).mean())) if len(sd) else np.nan


def compare(score, objective, sigma, n=2000, seed=0):
    keys = score.dropna().index.intersection(objective.dropna().index)
    a, b = score.loc[keys].to_numpy(float), objective.loc[keys].to_numpy(float)
    if len(keys) < 4:
        return {'n': len(keys)}
    rho  = spearmanr(a, b).correlation
    rng  = np.random.default_rng(seed)
    boot = [spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, len(a), len(a)) for _ in range(n)) if len(set(i)) > 3]
    pairs = [(i, j) for i in range(len(a)) for j in range(i + 1, len(a)) if abs(a[i] - a[j]) > 2 * sigma]
    good  = np.mean([np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]) for i, j in pairs]) if pairs else np.nan
    return {'n': len(keys), 'rho': round(rho, 3), 'ic95': (round(np.nanpercentile(boot, 2.5), 2), round(np.nanpercentile(boot, 97.5), 2)), 'pares': len(pairs), 'ordem_certa': round(good, 3)}


def candidates(table, probe):
    e1  = pd.read_csv(f'{EXP}/profile/objective_e1.csv', index_col=0)
    vis = pd.read_csv(f'{EXP}/profile/visibility.csv', index_col=0)
    out = {f'E1 {c}': e1[c] for c in e1}
    gap = np.maximum(TRAVA - vis.vis_med, 0)
    for c in ('marg', 'cob_grupos', 'cob_x_den', 'energia', 'composicao'):
        out[f'E1 {c} − trava'] = e1[c] - 10 * gap.reindex(e1.index).fillna(0) * e1[c].abs().median()
    out['trava (vis_med)'] = vis.vis_med
    for c in FREE:
        out[f'E2 −{c}'] = -table[c]
    out['E2 no_mergulho − frag − horiz'] = table.no_mergulho - table.frag_curtos - table.horizontal
    inj = injection(probe)
    if len(inj):
        out.update({f'E3 {c}' if c.startswith('inj_recall') else f'E3 −{c}': inj[c] if c.startswith('inj_recall') else -inj[c] for c in inj})
        out['E3 recall − 10·fundo'] = inj.inj_recall - 10 * inj.inj_fundo
        out['E3 recall × (1 − 10·fundo)'] = inj.inj_recall * (1 - 10 * inj.inj_fundo)
    e2p = e2plus(probe)
    for c in e2p:
        out[f'E2+ {c}'] = e2p[c]
        out[f'E2+ −{c}'] = -e2p[c]
    bat = battery(probe)
    for c in bat:
        out[f'E2 −{c}' if c != 'bat_recall_falhas' else f'E2 {c}'] = -bat[c] if c != 'bat_recall_falhas' else bat[c]
    return out


if __name__ == '__main__':
    probe, metric = (sys.argv[1:] + ['Q20', 'f1@1'])[:2] if len(sys.argv) > 1 else ('Q20', 'f1@1')
    table = tstr(probe, metric)
    sigma = noise(table)
    pd.set_option('display.width', 250)
    print(f'TSTR {probe} {metric} (dev, média dos patches), ruído de semente σ = {sigma:.4f}')
    print(table[[metric, 'sd', 'n']].sort_values(metric, ascending=False).round(3).to_string())
    rows = {name: compare(table[metric], obj, sigma if np.isfinite(sigma) else 0.02) for name, obj in candidates(table, probe).items()}
    print()
    print(pd.DataFrame(rows).T.sort_values('rho', ascending=False).to_string())
