# ONDE A MÉTRICA OFICIAL PERDE: STICKS PREDITOS SEM PAR NO ESPECIALISTA (PRECISÃO) E STICKS DO ESPECIALISTA NÃO COBERTOS (RECALL), POR PROFUNDIDADE
import os, sys, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness'))
import marlim
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sections')
BANDS = [(0, 450), (450, 700), (700, 950), (950, 1200), (1200, 1601)]
rows = []
for tag in ('art5_dbrnet', 'ms7_nru'):
    for pid in ('1200', '1300', '1400', '2600'):
        sec = np.load(f'{S}/{tag}_{pid}.npy').astype(np.float32)
        metrics, cmp = marlim.stickMetrics(pid, sec)
        mine, gt = cmp.sticks('rede'), cmp.sticks('interpretado')
        for kind, items, others in (('predito', mine, gt), ('especialista', gt, mine)):
            for s in items:
                zc = (s['p0'][1] + s['p1'][1]) / 2
                band = next(f'{a}-{b}' for a, b in BANDS if a <= zc < b)
                rows.append({'rede': tag, 'patch': pid, 'tipo': kind, 'faixa_z': band, 'len': s['len'], 'coberto': cmp.covered(s, others), 'mergulho': s['dip']})
df = pd.DataFrame(rows)
df['len_coberto'] = df.len * df.coberto
df['len_perdido'] = df.len * (1 - df.coberto)
pd.set_option('display.width', 200)
for kind, label in (('predito', 'PRECISÃO: comprimento predito SEM par no especialista, por faixa de z (fração do total perdido)'), ('especialista', 'RECALL: comprimento do especialista NÃO coberto, por faixa de z (fração do total perdido)')):
    part = df[df.tipo == kind]
    table = part.groupby(['rede', 'faixa_z']).agg(sticks=('len', 'size'), len_total=('len', 'sum'), len_perdido=('len_perdido', 'sum'))
    table['frac_do_perdido'] = table.len_perdido / table.groupby(level=0).len_perdido.transform('sum')
    table['taxa_perda_na_faixa'] = table.len_perdido / table.len_total
    print(f'\n{label}'); print(table.round(3).to_string())
print('\nmergulho dos sticks preditos sem par (p10/p50/p90):', df[(df.tipo == 'predito') & (df.coberto < 0.5)].mergulho.quantile([0.1, 0.5, 0.9]).round(1).tolist(), '| com par:', df[(df.tipo == 'predito') & (df.coberto >= 0.5)].mergulho.quantile([0.1, 0.5, 0.9]).round(1).tolist())
