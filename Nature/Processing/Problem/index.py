import math
import warnings
import numpy as np


# TRADUZ O PROBLEMA DO USUÁRIO (VARIÁVEIS, BOUNDS, RESTRIÇÕES) EM GENOMA NUMÉRICO.
class Problem:
    TYPES = ('real', 'int', 'cat', 'bool')
    WORST = 1e12

    def __init__(self, objective, variables, maximize=True, constraints=None, backend='thread'):
        self.objective = objective

        if isinstance(maximize, (list, tuple)):
            raise ValueError('maximize deve ser bool: o framework é mono-objetivo.')
        
        # todo modelo maximiza raw * weight: minimizar é só inverter o sinal
        self.weight = 1.0 if maximize else -1.0

        if not isinstance(variables, dict) or not variables:
            raise ValueError('variables deve ser um dict não vazio.')

        self.names = list(variables.keys())
        self.nVars = len(self.names)
        self.genes = [self.gene(n, variables[n]) for n in self.names]
        self.low   = np.array([g['low'] for g in self.genes])

        # teto seguro: evita divisão por zero em variáveis fixas (low == up)
        self.up    = np.array([max(g['up'], g['low'] + 1e-9) for g in self.genes])
        self.constraints = [constraints] if callable(constraints) else list(constraints or [])
        self.penalty     = (-self.WORST if self.weight > 0 else self.WORST,)
        self.warned      = False

        if backend not in ('thread', 'process', 'vector'):
            raise ValueError("backend deve ser 'thread', 'process' ou 'vector'.")
        
        if backend == 'vector' and any(g['type'] != 'real' for g in self.genes):
            raise ValueError("backend='vector' só suporta variáveis reais.")
        
        self.backend = backend

    def gene(self, name, cfg):
        t = cfg.get('type', 'real')

        if t not in self.TYPES:
            raise ValueError(f"Variável '{name}': type='{t}' inválido {self.TYPES}.")
        
        g = {'name': name, 'type': t}
        if t == 'bool':
            g['low'], g['up'] = 0.0, 1.0
            return g
        if t == 'cat':
            g['values'] = list(cfg.get('values', []))
            if not g['values']:
                raise ValueError(f"Variável '{name}' (cat) exige 'values' não vazio.")
            g['low'], g['up'] = 0.0, float(len(g['values']) - 1)
            return g
        if 'bounds' not in cfg:
            raise ValueError(f"Variável '{name}' ({t}) exige 'bounds'.")
        lo, hi = cfg['bounds']
        if hi < lo:
            raise ValueError(f"Variável '{name}': bounds invertidos {cfg['bounds']}.")
        if t == 'int':
            lo, hi = math.ceil(lo), math.floor(hi)
            if lo > hi:
                raise ValueError(f"Variável '{name}' (int): nenhum inteiro em {cfg['bounds']}.")
        g['low'], g['up'] = float(lo), float(hi)
        return g

    def decode(self, genome):
        out = {}
        for i, g in enumerate(self.genes):
            v, t = min(max(genome[i], g['low']), g['up']), g['type']
            if t == 'real': out[g['name']] = float(v)
            if t == 'int':  out[g['name']] = int(round(v))
            if t == 'bool': out[g['name']] = bool(round(v))
            if t == 'cat':  out[g['name']] = g['values'][int(round(v))]
        return out

    # O Portrait PRECISA DO GENOMA PARA FIXAR AS OUTRAS VARIÁVEIS AO FATIAR O RELEVO
    def encode(self, best):
        return np.array([g['values'].index(best[g['name']]) if g['type'] == 'cat' else float(best[g['name']]) for g in self.genes])

    # nan/inf SEQUESTRAM O argmax E VIRAM "MELHOR SOLUÇÃO" PARA SEMPRE: ENTRAM COMO PENALIDADE
    def valid(self, score):
        if all(math.isfinite(s) for s in score):
            return score
        if not self.warned:
            self.warned = True
            warnings.warn('objective devolveu valor não finito (nan/inf); tratado como penalidade.', stacklevel=3)
        return self.penalty

    def evaluate(self, genome):
        d = self.decode(genome)
        if self.constraints and not all(c(d) for c in self.constraints):
            return self.penalty
        score = self.objective(d)
        if isinstance(score, (list, tuple, np.ndarray)):
            score = float(np.asarray(score).ravel()[0])
        return self.valid((float(score),))

    def score(self, X, mapFunc):
        if self.backend == 'vector':
            out = np.asarray(self.objective(np.asarray(X, float)), float).reshape(len(X), -1)
            # o caminho vetorizado não passa pelo evaluate: checa restrições aqui
            if self.constraints:
                for i, row in enumerate(X):
                    if not all(c(self.decode(row)) for c in self.constraints):
                        out[i] = self.penalty
            return out if np.isfinite(out).all() else np.array([self.valid(tuple(r)) for r in out], float)
        return np.array(list(mapFunc(self.evaluate, [list(r) for r in X])), float)
