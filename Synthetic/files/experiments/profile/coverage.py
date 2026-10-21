# OBJETIVO SEM RÓTULO CANDIDATO (E1): COBERTURA E DENSIDADE DE NAEEM ET AL. (2020) DOS TILES REAIS PELOS SINTÉTICOS NO ESPAÇO
# DOS DESCRITORES DA RÉGUA, PADRONIZADO PELO REAL (MEDIANA E IQR DOS TILES VIVOS). PARA A META-VALIDAÇÃO (M2) CONTRA O TSTR.
import numpy as np, pandas as pd

FEATURES = ['period', 'bandwidth', 'coherence', 'dipP90', 'disc5', 'disc17', 'lag4', 'fine', 'kurt', 'trend', 'steep90', 'flat90']


def standardize(real, synth, cols):
    mid   = real[cols].median()
    scale = (real[cols].quantile(0.75) - real[cols].quantile(0.25)).replace(0, 1)
    return ((real[cols] - mid) / scale).to_numpy(float), ((synth[cols] - mid) / scale).to_numpy(float)


# COBERTURA = FRAÇÃO DOS REAIS COM UM SINTÉTICO DENTRO DO RAIO DO k-ÉSIMO VIZINHO REAL; DENSIDADE = MÉDIA DE BOLAS REAIS POR SINTÉTICO / k
def naeem(real, synth, cols=FEATURES, k=5):
    real  = real.dropna(subset=cols)
    synth = synth.dropna(subset=cols)
    R, S  = standardize(real, synth, cols)
    dRR   = np.linalg.norm(R[:, None] - R[None], axis=2)
    np.fill_diagonal(dRR, np.inf)
    radius = np.sort(dRR, axis=1)[:, k - 1]
    dRS   = np.linalg.norm(R[:, None] - S[None], axis=2)
    inside = dRS <= radius[:, None]
    return {'cobertura': float(inside.any(1).mean()), 'densidade': float(inside.sum(0).mean() / k), 'n_real': len(R), 'n_sint': len(S)}
