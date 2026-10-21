# CALIBRAÇÃO DO GERADOR POR AG (PROTÓTIPO DA FASE E, NO MOLDE DA Calibration DO Regions/Synthetic): GENOMA EM CENTRO + LOG DA
# LARGURA DE CADA INTERVALO, CAIXA PELA PLAUSIBILIDADE FÍSICA EM VOLTA DA REFERÊNCIA, SEMENTES COMUNS POR GERAÇÃO (A RETOMADA
# REPETE OS SORTEIOS), CMA-ES DO Synthetic/Nature COM PARTIDA NA REFERÊNCIA (mean) E ESCOLHA FINAL EM SEMENTES NOVAS.
# O OBJETIVO É PLUGÁVEL: QUALQUER OBJETO COM getScores(candidatos, semente) -> [{'objetivo': ..., ...}] (VER E1Objective).
import os, sys, json, hashlib
import numpy as np, pandas as pd

ROOT = '/home/grva-mint/Projects/Falhas'
EXP  = f'{ROOT}/Synthetic/files/experiments'
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
sys.path.insert(0, f'{EXP}/harness')
sys.path.insert(0, f'{EXP}/profile')
from Synthetic.index import SyntheticGenerator
from Synthetic.Nature.index import NatureSelector
from Synthetic.Nature.Processing.Memory.index import Memory


# UM TILE DA BUSCA NA ESCALA DOS DATASETS ('normalize': MIN-MAX DO VOLUME) E NOS EIXOS (x, z, y), COM A MÁSCARA
def getTile(options, seed, fast=True):
    generator = SyntheticGenerator(seed=seed)
    generator.set({**options, 'fast': fast})
    image, mask = generator.get()
    image = np.transpose(generator.applyGain(image), (0, 2, 1))
    return (image - image.min()) / (image.max() - image.min()), np.transpose(mask, (0, 2, 1))


# A PARTE MEDIDA (M4): O PERFIL SEM RÓTULO DO BLOCO (1 - Profile) VIRA A MISTURA DE REGIÕES DO dataset() — UMA REGIÃO POR FAIXA DE
# PROFUNDIDADE COM PELO MENOS minShare DOS TILES VIVOS, COM ESSE PESO E A FREQUÊNCIA QUE DÁ O PERÍODO p10–p90 DA FAIXA PELO MAPA
# f → PERÍODO MEDIDO PELA MESMA RÉGUA (profile/period_map.csv; FORA DO MAPA, A BORDA). O base VALE PARA TODAS AS REGIÕES.
def getRegions(profile, periodMap, nImages, base=None, band=250, minShare=0.05):
    live   = profile[(profile.water <= 0.8) & (profile.live >= 0.8) & profile.period.notna()]
    curve  = periodMap.groupby('f').period.median().sort_values()
    toFreq = lambda p: float(np.exp(np.interp(np.log(p), np.log(curve.values), np.log(curve.index.values))))
    groups = {z: g for z, g in live.groupby((live.z // band) * band) if len(g) >= minShare * len(live)}
    total  = sum(len(g) for g in groups.values())
    counts = {z: int(round(nImages * len(g) / total)) for z, g in groups.items()}
    counts[max(counts, key=counts.get)] += nImages - sum(counts.values())
    regions = {}
    for z, g in groups.items():
        low, high = g.period.quantile([0.1, 0.9])
        regions[f'z{int(z)}'] = {'n_images': counts[z], 'output': f'z{int(z)}', 'params': {**(base or {}), 'waveletLog': True, 'waveletFreq': (round(toFreq(high), 1), round(toFreq(low), 1))}}
    return regions


# OBJETIVO E1: COBERTURA DO PERFIL REAL PELOS TILES DO CANDIDATO NA RÉGUA, MENOS O QUE FALTA À VISIBILIDADE DO RÓTULO PARA O PISO
class E1Objective:
    TERM  = 'marg'
    FLOOR = 0.75

    def __init__(self, real, nTiles=40, fast=True):
        from ruler import TileRuler
        import objective, visibility
        self.real, self.nTiles, self.fast = real, nTiles, fast
        self.ruler, self.objective, self.visibility = TileRuler(), objective, visibility

    def getScores(self, candidates, seed):
        out = []
        for params in candidates:
            tiles = [getTile(params, seed + i, self.fast) for i in range(self.nTiles)]
            table = pd.DataFrame([self.ruler.get(image) for image, _ in tiles])
            vis   = float(np.nanmedian([self.getVisibility(image, mask) for image, mask in tiles]))
            parts = self.objective.scores(self.real, table)
            gap   = max(self.FLOOR - vis, 0) if np.isfinite(vis) else self.FLOOR
            out.append({'objetivo': parts[self.TERM] - 10 * gap, 'visibilidade': vis, **parts})
        return out

    def getVisibility(self, image, mask):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            np.save(f'{folder}/i.npy', image.astype(np.float32))
            np.save(f'{folder}/m.npy', mask)
            return self.visibility.tile((f'{folder}/i.npy', f'{folder}/m.npy'))['vis']


# OBJETIVO POR SONDA: CADA CANDIDATO VIRA UM DATASET PEQUENO GERADO NA PLACA, A SONDA (A Q20: dbrnet nf 16, 20 ÉPOCAS) TREINA NELE
# PELO train.py E O measure DÁ A NOTA: 'tstr' = F1@1 MÉDIO NOS PATCHES DE DEV (USA A INTERPRETAÇÃO, SÓ O MARLIM TEM), 'injecao' =
# RECALL DA FALHA INJETADA MENOS O PESO × A MARCAÇÃO DO TILE REAL SEM INJEÇÃO (SEM RÓTULO, PORTÁVEL), OU UMA FUNÇÃO (modelo, tag) ->
# {'objetivo': ...}. O DATASET DE CADA AVALIAÇÃO É APAGADO DEPOIS DO TREINO; O MODELO FICA EM models/calib_<semente>_<k> E A NOTA
# EM calib/<semente>_<k>.json, QUE A RETOMADA REAPROVEITA.
class ProbeObjective:
    INFO   = {'network': 'dbrnet', 'img_size': None, 'scaling': 'normalize', 'lr': 0.001, 'loss': 'dice_focal', 'batch_size': 2, 'scheduler': 'plateau', 'dropout': 0.1, 'num_filters': 16, 'ema': True, 'epochs': 20, 'patience': 15, 'n_trials': 1, 'trial': 0}
    WEIGHT = 10.0

    def __init__(self, measure='tstr', nTiles=100, fast=True, info=None, patches=('1200', '1300'), workers=1):
        self.measure, self.nTiles, self.fast, self.patches, self.workers = measure, nTiles, fast, list(patches), workers
        self.info = {**self.INFO, **(info or {})}

    # TRÊS ETAPAS: GERA OS DATASETS DA GERAÇÃO EM SÉRIE NA PLACA, TREINA workers SONDAS AO MESMO TEMPO (CADA UMA NUM PROCESSO) E MEDE
    def getScores(self, candidates, seed):
        from concurrent.futures import ThreadPoolExecutor
        names = [self.prepare(params, seed, k) for k, params in enumerate(candidates)]
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(self.train, [name for name in names if name]))
        return [self.getScore(seed, k) for k in range(len(candidates))]

    def prepare(self, params, seed, k):
        import torch, shutil, glob, make_dataset
        name, data, run = f'calib_{seed}_{k}', f'{EXP}/calib/{seed}_{k}', f'{EXP}/models/calib_{seed}_{k}'
        if os.path.exists(f'{EXP}/calib/{seed}_{k}.json') or glob.glob(f'{run}/model_*/model.pth'):
            return None
        shutil.rmtree(data, ignore_errors=True)
        SyntheticGenerator().dataset({'directory': f'{data}/original', 'seed': seed, 'regions': {'tiles': {'n_images': self.nTiles, 'output': 'tiles', 'params': {**params, 'fast': self.fast}}}})
        make_dataset.format(data)
        torch.cuda.empty_cache()
        os.makedirs(run, exist_ok=True)
        json.dump({'info': {**self.info, 'dataset': name}, 'database': f'{data}/DataBase.csv', 'params': params, 'seed': seed}, open(f'{run}/config.json', 'w'), indent=1, default=float)
        return name

    def train(self, name):
        import shutil, subprocess
        subprocess.run([sys.executable, f'{EXP}/train.py', name], check=True, stdout=open(f'{EXP}/models/{name}/train.log', 'a'), stderr=subprocess.STDOUT)
        shutil.rmtree(f'{EXP}/calib/{name.removeprefix("calib_")}')

    def getScore(self, seed, k):
        import glob
        path = f'{EXP}/calib/{seed}_{k}.json'
        if os.path.exists(path):
            return json.load(open(path))
        model = sorted(glob.glob(f'{EXP}/models/calib_{seed}_{k}/model_*'))[-1]
        score = {**self.getMeasure(model, f'calib_{seed}_{k}'), 'semente': seed, 'k': k}
        json.dump(score, open(path, 'w'), default=float)
        return score

    def getMeasure(self, model, tag):
        import marlim, injection
        results = f'{EXP}/calib/results.jsonl'
        if callable(self.measure):
            return self.measure(model, tag)
        if self.measure == 'tstr':
            rows = marlim.run(model, self.patches, tag, fast=True, results=results, save=f'{EXP}/calib/sections')
            keys = ('f1@1', 'f1', 'recall', 'precisao', 'mask_frac')
            return {'objetivo': float(np.mean([row['f1@1'] for row in rows])), **{k: float(np.mean([row[k] for row in rows])) for k in keys}}
        out = injection.run(model, tag, results=results)
        return {'objetivo': out['inj_recall'] - self.WEIGHT * out['inj_fundo'], **{k: v for k, v in out.items() if k.startswith('inj_')}}


class Calibration:
    SPREAD      = 0.25    # MEIA-LARGURA DA CAIXA DO CENTRO EM FRAÇÃO DA FAIXA DE CADA PARÂMETRO
    MIN_WIDTH   = 0.01    # LARGURA MÍNIMA DE UM INTERVALO EM FRAÇÃO DA FAIXA
    POPULATION  = 10
    GENERATIONS = 30
    N_CHECK     = 4
    SEED        = 20261021

    def __init__(self, objective, reference, limits, fixed=None, name='marlim'):
        self.objective = objective
        self.reference = reference
        self.limits    = limits
        self.fixed     = fixed or {}
        self.box       = {key: self.getBox(key, value) for key, value in self.getGenome(reference).items()}
        key            = hashlib.md5(json.dumps([self.box, self.fixed, type(objective).__name__, getattr(objective, 'nTiles', None), self.SEED], sort_keys=True, default=str).encode()).hexdigest()[:8]
        self.memory    = f'{ROOT}/Synthetic/files/memory/{name}_{key}'
        Memory.EVERY   = 1

    def update(self):
        problem = {
            'objective': self.getObjective,
            'maximize': True,
            'seed': 42,
            'verbose': True,
            'backend': 'vector',
            'variables': {key: {'bounds': (0.0, 1.0)} for key in self.box},
            'mean': self.encode(self.reference),
            'sigma': 0.25,
            'population': self.POPULATION,
            'generations': 0 if os.path.exists(f'{self.memory}/best.json') else self.GENERATIONS,
            'patience': None,
        }
        self.search = NatureSelector('genetic', problem, memory=self.memory)
        self.search.update()
        return self.search

    # OS INDIVÍDUOS DE UMA GERAÇÃO DIVIDEM A SEMENTE, QUE SAI DA PRÓPRIA POPULAÇÃO: A RETOMADA REPETE OS MESMOS SORTEIOS
    def getObjective(self, X):
        seed = self.SEED + int(hashlib.md5(np.asarray(X, float).tobytes()).hexdigest()[:7], 16) % 10 ** 7
        rows = self.objective.getScores([self.getParams(u) for u in X], seed)
        self.log(X, rows, seed)
        return [row['objetivo'] for row in rows]

    def log(self, X, rows, seed):
        os.makedirs(self.memory, exist_ok=True)
        with open(f'{self.memory}/evaluations.jsonl', 'a') as file:
            for u, row in zip(X, rows):
                file.write(json.dumps({'seed': seed, 'genome': [float(v) for v in u], **{k: v for k, v in row.items() if np.isscalar(v)}}, default=float) + '\n')

    # A ESCOLHA FINAL EM SEMENTES QUE A BUSCA NUNCA USOU; A REFERÊNCIA SÓ PERDE PARA UMA VANTAGEM MAIOR QUE O ERRO PADRÃO
    def evaluate(self):
        candidates = {'referência': self.reference, 'busca': self.decode(list(self.search.best.values())), 'média da busca': self.decode(self.search.optimizer.m)}
        seeds      = [self.SEED + 10 ** 8 + 1000 * k for k in range(self.N_CHECK)]
        runs       = pd.DataFrame([{'candidato': name, 'semente': seed, **score} for seed in seeds for name, score in zip(candidates, self.objective.getScores([{**params, **self.fixed} for params in candidates.values()], seed))])
        scores     = runs.pivot(index='semente', columns='candidato', values='objetivo')[list(candidates)]
        gain       = scores.sub(scores['referência'], axis=0)
        summary    = pd.DataFrame({'objetivo': scores.mean(), 'desvio': scores.std(), 'ganho': gain.mean(), 'erro': gain.std() / np.sqrt(self.N_CHECK)})
        chosen     = summary.objetivo.idxmax()
        self.winner = candidates[chosen if summary.loc[chosen, 'ganho'] > summary.loc[chosen, 'erro'] else 'referência']
        return summary

    def getParams(self, u):
        return {**self.decode(u), **self.fixed}

    # OS PARÂMETROS PLANOS DO GENOMA: CADA INTERVALO VIRA O CENTRO E O LOG DA LARGURA
    def getGenome(self, params):
        genome = {}
        for key, item in params.items():
            if isinstance(item, (tuple, list)):
                genome[f'{key}Center'], genome[f'{key}Width'] = (item[0] + item[1]) / 2, np.log(max(item[1] - item[0], self.getMinimum(key)))
            else:
                genome[key] = item
        return genome

    # O CENTRO ANDA SPREAD DA FAIXA EM VOLTA DA REFERÊNCIA, NO MÍNIMO 1 NOS INTEIROS; A LARGURA VAI DA MÍNIMA À FAIXA INTEIRA
    def getBox(self, name, value):
        key       = name.removesuffix('Center').removesuffix('Width')
        low, high = self.limits[key]
        half      = max(self.SPREAD * (high - low), 1 if isinstance(low, int) else 0)
        return (float(np.log(self.getMinimum(key))), float(np.log(high - low))) if name.endswith('Width') else (max(low, value - half), min(high, value + half))

    def getMinimum(self, key):
        low, high = self.limits[key]
        return max(self.MIN_WIDTH * (high - low), 1 if isinstance(low, int) else 0)

    def encode(self, params):
        genome = self.getGenome(params)
        return [float(np.clip((genome[name] - low) / (high - low), 0, 1)) for name, (low, high) in self.box.items()]

    # GENOMA NORMALIZADO -> PARÂMETROS: O INTERVALO EM VOLTA DO CENTRO, PRESO NA FAIXA E, NOS INTEIROS, COM O TETO EXCLUSIVO ACIMA DO PISO
    def decode(self, u):
        genome = {name: low + value * (high - low) for (name, (low, high)), value in zip(self.box.items(), u)}
        params = {}
        for key, (low, high) in self.limits.items():
            if key in genome:
                params[key] = int(round(genome[key])) if isinstance(low, int) else float(genome[key])
                continue
            if f'{key}Center' not in genome:
                continue
            half = np.exp(genome[f'{key}Width']) / 2
            a, b = max(low, genome[f'{key}Center'] - half), min(high, genome[f'{key}Center'] + half)
            params[key] = (int(round(a)), max(int(round(b)), int(round(a)) + 1)) if isinstance(low, int) else (float(a), float(b))
        return params
