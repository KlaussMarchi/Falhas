# PORTÃO C: CADA ABLAÇÃO CONTRA O SEU CONTROLE NA MESMA SONDA (A*/B* CONTRA O A0, C1_* CONTRA O C0_v1, C0_v2 CONTRA O C0_v1),
# MÉDIA DOS PATCHES DE DEV E DAS SEMENTES; Δ MARCADO COM * QUANDO PASSA DE 2σ DO RUÍDO DA PRÓPRIA SONDA (SEMENTE DE TREINO ⊕ AMOSTRA DO GERADOR).
# uso: python gate_c.py [sonda: Q20|P20]
import sys, json
import numpy as np, pandas as pd

EXP     = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
METRICS = ['f1', 'f1@1', 'recall', 'precisao', 'deteccao', 'mask_frac', 'perto_gt', 'cobertura', 'cob_z450-700', 'cob_z700-950', 'cob_z950-1200', 'frag_curtos', 'horizontal', 'water_frac', 'borda_x0']
SMOOTH  = ['pf@1', 'pf@2', 'pp@1', 'auc_perto', 'ap_perto', 'top2_z700-950', 'top2_z>1200']
CONTROL = {'C1_': 'C0_v1', 'C0_v2': 'C0_v1', 'C0_v1': 'A0_ctrl', 'A': 'A0_ctrl', 'B': 'A0_ctrl', 'dataset_74': 'A0_ctrl'}


def control(dataset):
    return next((c for prefix, c in CONTROL.items() if dataset.startswith(prefix)), None)


def table(probe):
    rows = pd.DataFrame([json.loads(l) for l in open(f'{EXP}/results.jsonl')])
    rows = rows[rows.tag.str.startswith(probe + '_')]
    tags = rows[['tag', 'dataset']].dropna().drop_duplicates().set_index('tag').dataset
    fast = rows[rows.metodo == 'rapido']
    # AS MEDIDAS SUAVES (smooth.py) E O F1 IGUALADO EM 1–5% (f1curve.py), CALCULADOS DAS SEÇÕES SALVAS
    for name, cols in (('smooth', SMOOTH), ('f1curve', ['f1_curva'])):
        extra_ = pd.read_csv(f'{EXP}/profile/{name}.csv', dtype={'patch': str})
        fast   = fast.merge(extra_[['tag', 'patch'] + [c for c in cols if c in extra_]], on=['tag', 'patch'], how='left')
    per  = fast.groupby(['dataset', 'tag'])[[m for m in METRICS + SMOOTH + ['f1_curva'] if m in fast]].mean()
    extra = []
    for kind in ('bateria', 'injecao'):
        part = rows[rows.metodo == kind].dropna(axis=1, how='all')
        if len(part):
            cols = [c for c in part if c.startswith(('bat_', 'inj_'))]
            extra.append(part.assign(dataset=part.tag.map(tags)).groupby(['dataset', 'tag'])[cols].mean())
    return pd.concat([per] + extra, axis=1) if extra else per


if __name__ == '__main__':
    probe = sys.argv[1] if len(sys.argv) > 1 else 'Q20'
    per   = table(probe)
    mean  = per.groupby('dataset').mean()
    sd    = per.groupby('dataset').std()
    n     = per.groupby('dataset').size()
    noise = np.sqrt((sd[n >= 2] ** 2).mean())
    # O RUÍDO DE AMOSTRA DO GERADOR: dataset_74 E A0_ctrl SÃO A MESMA CONFIGURAÇÃO COM SEMENTES DE GERAÇÃO DIFERENTES
    if {'dataset_74', 'A0_ctrl'} <= set(mean.index):
        sample = (mean.loc['dataset_74'] - mean.loc['A0_ctrl']).abs() / np.sqrt(2)
        noise  = np.sqrt(noise.fillna(0) ** 2 + sample.fillna(0) ** 2)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print(f'{probe}: datasets {len(mean)}, rodadas por dataset {n.to_dict()}')
    print('ruído (σ): semente de treino (datasets com ≥ 2 rodadas) ⊕ amostra do gerador (|74 − A0| / √2):')
    print(noise.dropna().round(4).to_string())
    print()
    rows = []
    for dataset in mean.index:
        ref = control(dataset)
        if ref is None or ref == dataset or ref not in mean.index:
            continue
        delta = mean.loc[dataset] - mean.loc[ref]
        mark  = {m: f'{delta[m]:+.3f}{"*" if np.isfinite(noise.get(m, np.nan)) and abs(delta[m]) > 2 * noise[m] else ""}' for m in delta.index if np.isfinite(delta[m])}
        rows.append({'dataset': dataset, 'controle': ref, 'n': int(n[dataset]), **mark})
    print(pd.DataFrame(rows).set_index('dataset').to_string())
