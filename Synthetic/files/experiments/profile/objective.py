# OBJETIVOS SEM RÓTULO CANDIDATOS DA FASE E (FAMÍLIA E1): O QUANTO OS TILES DE UM DATASET SINTÉTICO COBREM O PERFIL DOS TILES
# REAIS DE DEV PELA RÉGUA (profile/<nome>.pkl), SEM TREINO E SEM RÓTULO. A META-VALIDAÇÃO (meta.py) DIZ QUAL ORDENA COMO O TSTR.
# uso: python objective.py [datasets...]   (sem argumento: todos os profile/*.pkl sintéticos)
import os, sys, glob
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

HERE     = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['period', 'bandwidth', 'coherence', 'dipP90', 'disc5', 'disc17', 'lag4', 'lag16', 'fine', 'kurt', 'trend', 'steep90', 'flat90', 'sat']
GROUPS   = {'escala': ['period', 'bandwidth'], 'continuidade': ['disc5', 'disc17', 'lag4', 'lag16'], 'textura': ['fine', 'kurt', 'coherence', 'sat'], 'estrutura': ['dipP90', 'steep90', 'flat90', 'trend']}
REAL     = ('marlim_1200', 'marlim_1300')
NOT_SYN = ('iracema', 'marlim_1', 'marlim_2', 'art5_', 'offset_', 'objective', 'context_')


def realTable():
    real = pd.concat([pd.read_pickle(f'{HERE}/{n}.pkl') for n in REAL], ignore_index=True)
    return real


def classes(table):
    return pd.Series(np.select([table.water > 0.5, table.live < 0.1, table.live > 0.9], ['agua', 'morto', 'refletivo'], 'misto'), index=table.index)


def standardize(real, synth, cols):
    mid   = real[cols].median()
    scale = (real[cols].quantile(0.75) - real[cols].quantile(0.25)).replace(0, 1)
    return ((real[cols] - mid) / scale).to_numpy(float), ((synth[cols] - mid) / scale).to_numpy(float)


# COBERTURA E DENSIDADE DE NAEEM ET AL. (2020): REAL COBERTO = TEM UM SINTÉTICO DENTRO DO RAIO DO k-ÉSIMO VIZINHO REAL
def naeem(R, S, k=5):
    dRR = np.linalg.norm(R[:, None] - R[None], axis=2)
    np.fill_diagonal(dRR, np.inf)
    radius = np.sort(dRR, axis=1)[:, k - 1]
    inside = np.linalg.norm(R[:, None] - S[None], axis=2) <= radius[:, None]
    return float(inside.any(1).mean()), float(inside.sum(0).mean() / k)


# DISTÂNCIA DE ENERGIA (RIZZO & SZÉKELY) ENTRE AS DUAS NUVENS PADRONIZADAS
def energy(R, S):
    d = lambda A, B: np.linalg.norm(A[:, None] - B[None], axis=2).mean()
    return float(2 * d(R, S) - d(R, R) - d(S, S))


# DISTÂNCIA A PROXY (BEN-DAVID ET AL. 2010): ERRO DE UM CLASSIFICADOR REAL × SINTÉTICO EM VALIDAÇÃO CRUZADA, CLASSES BALANCEADAS
def pad(R, S, seed=0):
    rng = np.random.default_rng(seed)
    n   = min(len(R), len(S))
    X   = np.vstack([R[rng.choice(len(R), n, replace=False)], S[rng.choice(len(S), n, replace=False)]])
    y   = np.r_[np.zeros(n), np.ones(n)]
    acc = cross_val_score(HistGradientBoostingClassifier(max_iter=100, random_state=seed), X, y, cv=5).mean()
    return float(2 * (2 * acc - 1))


def scores(real, synth):
    rl, sl = real[real.live > 0.5].dropna(subset=FEATURES), synth[synth.live > 0.5].dropna(subset=FEATURES)
    R, S   = standardize(rl, sl, FEATURES)
    lo, hi = sl[FEATURES].quantile(0.025), sl[FEATURES].quantile(0.975)
    rlo, rhi = rl[FEATURES].quantile(0.025), rl[FEATURES].quantile(0.975)
    out = {
        'marg': float(((rl[FEATURES] >= lo) & (rl[FEATURES] <= hi)).mean().mean()),
        'marg_prec': float(((sl[FEATURES] >= rlo) & (sl[FEATURES] <= rhi)).mean().mean()),
        'energia': -energy(R, S),
        'pad': -pad(R, S) if min(len(R), len(S)) >= 20 else np.nan,
    }
    for name, cols in GROUPS.items():
        r, s = standardize(rl, sl, cols)
        out[f'cob_{name}'], out[f'den_{name}'] = naeem(r, s)
    out['cob_grupos'] = float(np.mean([out[f'cob_{n}'] for n in GROUPS]))
    out['den_grupos'] = float(np.mean([min(out[f'den_{n}'], 1.0) for n in GROUPS]))
    out['cob_x_den']  = out['cob_grupos'] * out['den_grupos']
    pr, ps = classes(real).value_counts(normalize=True), classes(synth).value_counts(normalize=True)
    out['composicao'] = float(1 - 0.5 * pr.sub(ps, fill_value=0).abs().sum())
    return out


def synthetic():
    names = [os.path.basename(p)[:-4] for p in glob.glob(f'{HERE}/*.pkl')]
    return sorted(n for n in names if not n.startswith(NOT_SYN))


if __name__ == '__main__':
    real  = realTable()
    names = sys.argv[1:] or synthetic()
    table = pd.DataFrame({n: scores(real, pd.read_pickle(f'{HERE}/{n}.pkl')) for n in names}).T
    old   = pd.read_csv(f'{HERE}/objective_e1.csv', index_col=0) if os.path.exists(f'{HERE}/objective_e1.csv') else pd.DataFrame()
    pd.concat([old.drop(index=table.index, errors='ignore'), table]).to_csv(f'{HERE}/objective_e1.csv')
    pd.set_option('display.width', 250)
    print(table.round(3).to_string())
