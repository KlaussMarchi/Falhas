# ONDE A REFERÊNCIA ERRA, POR CONTEXTO (ANÁLISE, USA O RÓTULO SÓ PARA ENTENDER, NÃO PARA CALIBRAR): EM CADA TILE DE DEV, A FRAÇÃO
# DA INLINE ANOTADA (NÚCLEO 64×64 DO TILE) MARCADA PELA dbrnet DE REFERÊNCIA LONGE (> 12 px) DE QUALQUER FALHA DO ESPECIALISTA,
# CONTRA OS DESCRITORES SEM RÓTULO DO TILE (PERFIL DA RÉGUA). CORRELAÇÃO DE SPEARMAN E MÉDIA POR QUARTIL.
import numpy as np, pandas as pd
import scipy.ndimage as ndi
from scipy.stats import spearmanr
from PIL import Image

ROOT = '/home/grva-mint/Projects/Falhas'
EXP  = f'{ROOT}/Synthetic/files/experiments'
rows = []
for pid in ('1200', '1300'):
    sec  = np.load(f'{EXP}/analise_19-10/sections/art5_dbrnet_{pid}.npy').astype(np.float32) >= 0.5
    gt   = np.array(Image.open(f'{ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    far  = ndi.distance_transform_edt(~gt) > 12
    prof = pd.read_pickle(f'{EXP}/profile/marlim_{pid}.pkl')
    for _, t in prof.iterrows():
        z0, x0 = int(t.z0), int(t.x0)
        cell   = (slice(z0, z0 + 64), slice(x0, x0 + 64))
        rows.append({'patch': pid, 'z': z0 + 32, 'mascara': float(sec[cell].mean()), 'fp': float((sec & far)[cell].mean()), 'perto': float((sec & ~far)[cell].mean()), 'gt': float(gt[cell].mean()), **{c: t[c] for c in ('live', 'period', 'sigmaLive', 'sat', 'fine', 'coherence', 'dipP90', 'disc5', 'disc17', 'lag16', 'steep90', 'flat90', 'kurt', 'trend')}})
df   = pd.DataFrame(rows)
live = df[(df.live > 0.5) & (df.z >= 450) & (df.z < 1200)]
print(f'tiles vivos de dev em z 450–1200: {len(live)}; FP médio {live.fp.mean():.4f}, máscara {live.mascara.mean():.4f}')
out = {c: {'ρ com FP': round(spearmanr(live[c], live.fp, nan_policy="omit").correlation, 3), 'ρ com máscara perto do GT': round(spearmanr(live[c], live.perto, nan_policy="omit").correlation, 3)} for c in ('period', 'sigmaLive', 'sat', 'fine', 'coherence', 'dipP90', 'disc5', 'disc17', 'lag16', 'steep90', 'flat90', 'kurt', 'trend', 'gt')}
print(pd.DataFrame(out).T.sort_values('ρ com FP').to_string())
print()
for c in ('disc17', 'steep90', 'period', 'sat', 'coherence', 'sigmaLive'):
    q = pd.qcut(live[c], 4, duplicates='drop')
    print(c, live.groupby(q, observed=True).fp.mean().round(4).to_dict())
df.to_pickle(f'{EXP}/profile/context_art5.pkl')
