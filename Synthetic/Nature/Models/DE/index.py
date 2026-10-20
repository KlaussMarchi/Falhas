import numpy as np
from ...Processing.Engine.index import Engine


# EVOLUÇÃO DIFERENCIAL (Storn & Price 1997): v = base + F·Σ(xr - xs), crossover bin/exp, seleção gulosa
# 1-a-1. strategy = 'base/diffs/cx'; o default current-to-best/1/bin ganha do rand/1/bin canônico na medição.
class DifferentialEvolution(Engine):
    MODEL      = 'de'
    DESC       = 'Evolução Diferencial'
    ARRAYS     = ('X', 'raw')
    BASES      = ('rand', 'best', 'current-to-best')
    CROSSOVERS = ('bin', 'exp')

    def __init__(self, objective, variables, maximize=True, population=50, generations=200, F=(0.5, 1.0),
                 CR=0.9, strategy='current-to-best/1/bin', patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        parts = strategy.split('/')
        if len(parts) != 3 or parts[0] not in self.BASES or parts[1] not in ('1', '2') or parts[2] not in self.CROSSOVERS:
            raise ValueError(f"strategy inválida '{strategy}'; use 'base/diffs/cx' com base {self.BASES}, diffs 1|2, cx {self.CROSSOVERS}.")
        self.base, self.diffs, self.cx = parts[0], int(parts[1]), parts[2]

        # rand precisa do vetor-base aleatório; best/current-to-best não. Cada diferença gasta 2 índices.
        needed = (self.base == 'rand') + 2 * self.diffs
        if population < needed + 1:
            raise ValueError(f"population deve ser >= {needed + 1} para a estratégia '{strategy}'.")

        self.F = (float(F), float(F)) if isinstance(F, (int, float)) else tuple(sorted(map(float, F)))
        if not 0.0 <= self.F[0] <= self.F[1] <= 2.0:
            raise ValueError('F deve estar em [0, 2] (escalar ou (min, max) para dither).')
        if not 0.0 <= CR <= 1.0:
            raise ValueError('CR deve estar em [0, 1].')
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)
        self.size     = int(population)
        self.span     = int(generations)
        self.CR       = float(CR)
        self.strategy = strategy

    def update(self):
        rng = self.open()

        try:
            arrays = self.resume('X')
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw = self.score(X)
                gen0 = 0
            else:
                X, raw, gen0 = arrays['X'], arrays['raw'], self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw}
            wf        = raw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = self.begin(gen0, X, raw, self.size, float(np.max(wf)), self.span)

            for gen in self.track(gen0 + 1):
                f = rng.uniform(*self.F)  # dither: um F por geração amplia a robustez (Storn & Price)
                U = self.cross(X, self.repair(X, self.donor(X, wf, f, rng)), rng)

                rawU = self.score(U)
                wfU  = rawU[:, 0] * self.problem.weight
                win  = wfU >= wf  # aceitar empate move a população por platôs e regiões inviáveis
                X[win], raw[win], wf[win] = U[win], rawU[win], wfU[win]

                if self.tick(gen, self.size, X, raw, float(np.max(wf))):
                    break
        finally:
            self.pool.stop()

        winner = int(np.argmax(wf))
        return self.finish(X[winner], raw[winner][0], self.stopped)

    def config(self):
        return {'population': self.size, 'generations': self.span, 'F': list(self.F), 'CR': self.CR, 'strategy': self.strategy, 'patience': self.patience, 'target': self.target}

    def donor(self, X, wf, f, rng):
        best = X[np.argmax(wf)]
        if self.base == 'rand':
            r = self.samples(1 + 2 * self.diffs, rng)
            return X[r[:, 0]] + f * self.delta(X, r[:, 1:])
        if self.base == 'best':
            return best + f * self.delta(X, self.samples(2 * self.diffs, rng))
        return X + f * (best - X) + f * self.delta(X, self.samples(2 * self.diffs, rng))

    # SORTEIO POR REJEIÇÃO, O(n·count) DE RAM — PERMUTAR UMA MATRIZ (n, n-1) POR GERAÇÃO É O(n²)
    def samples(self, count, rng):
        rows = np.arange(self.size)
        r    = self.pick(self.size, rows, rng)[:, None]
        for _ in range(count - 1):
            new = self.pick(self.size, rows, rng)
            while True:
                clash = (r == new[:, None]).any(1)
                if not clash.any():
                    break
                new[clash] = self.pick(int(clash.sum()), rows[clash], rng)
            r = np.hstack([r, new[:, None]])
        return r

    def delta(self, X, pairs):
        return sum(X[pairs[:, 2 * k]] - X[pairs[:, 2 * k + 1]] for k in range(self.diffs))

    def cross(self, X, V, rng):
        n, d = X.shape
        if self.cx == 'bin':
            mask = rng.random((n, d)) < self.CR
            mask[np.arange(n), rng.integers(d, size=n)] = True
            return np.where(mask, V, X)
        # exp: bloco contíguo com wrap a partir de um gene sorteado, enquanto rand < CR
        streak  = np.cumprod(np.hstack([np.ones((n, 1)), rng.random((n, d - 1)) < self.CR]), axis=1).astype(bool)
        offsets = (rng.integers(d, size=n)[:, None] + np.arange(d)) % d
        mask    = np.zeros((n, d), bool)
        mask[np.repeat(np.arange(n), d), offsets.ravel()] = streak.ravel()
        return np.where(mask, V, X)
