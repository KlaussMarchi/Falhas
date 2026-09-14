import numpy as np
from ...Processing.Engine.index import Engine


# ENXAME DE PARTÍCULAS (Kennedy & Eberhart 1995) COM INÉRCIA DECRESCENTE (Shi & Eberhart 1998) E
# COEFICIENTES VARIÁVEIS NO TEMPO (TVAC, Ratnaweera 2004): COMEÇA NA PRÓPRIA MEMÓRIA, TERMINA NO LÍDER.
class PSO(Engine):
    MODEL     = 'pso'
    DESC      = 'Enxame PSO'
    UNIT      = 'it'
    XLABEL    = 'Iteração'
    ARRAYS    = ('X', 'raw', 'V', 'pbest', 'pbestRaw')
    VMAX      = 0.5
    NEIGHBORS = 2

    def __init__(self, objective, variables, maximize=True, particles=50, iterations=200,
                 inertia=(0.9, 0.4), cognitive=(2.5, 0.5), social=(0.5, 2.5), topology='global',
                 patience=None, target=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if particles < 2:
            raise ValueError('particles deve ser >= 2.')
        if topology not in ('global', 'ring'):
            raise ValueError("topology deve ser 'global' ou 'ring'.")
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)
        self.size      = int(particles)
        self.span      = int(iterations)
        self.inertia   = self.pair(inertia)
        self.cognitive = self.pair(cognitive)
        self.social    = self.pair(social)
        self.topology  = topology

    def setup(self):
        self.vmax = self.VMAX * (self.problem.up - self.problem.low)
        offsets   = np.arange(-self.NEIGHBORS, self.NEIGHBORS + 1)
        self.ring = (np.arange(self.size)[:, None] + offsets) % self.size

    def update(self):
        self.setup()
        rng = self.open()

        try:
            arrays = self.resume('X')
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw = self.score(X)
                V   = rng.uniform(-self.vmax, self.vmax, X.shape)
                pbest, pbestRaw, it0 = X.copy(), raw.copy(), 0
            else:
                X, raw, V, pbest, pbestRaw = (arrays[k] for k in self.ARRAYS)
                it0 = self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw, 'V': V, 'pbest': pbest, 'pbestRaw': pbestRaw}
            pbestWf   = pbestRaw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = self.begin(it0, X, raw, self.size, float(np.max(pbestWf)), self.span)

            for it in self.track(it0 + 1):
                w, c1, c2 = (a + (b - a) * self.phase(it) for a, b in (self.inertia, self.cognitive, self.social))
                r1, r2 = rng.random((2,) + X.shape)
                V   = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (self.leaders(pbest, pbestWf) - X)
                V   = np.clip(V, -self.vmax, self.vmax)
                X   = X + V
                out = (X < self.problem.low) | (X > self.problem.up)
                X   = np.clip(X, self.problem.low, self.problem.up)
                V[out] = 0.0   # fronteira absorvente: evita partículas coladas no limite

                raw    = self.score(X)
                wf     = raw[:, 0] * self.problem.weight
                better = wf > pbestWf
                pbest[better], pbestRaw[better], pbestWf[better] = X[better], raw[better], wf[better]

                if self.tick(it, self.size, X, raw, float(np.max(pbestWf))):
                    break
        finally:
            self.pool.stop()

        winner = int(np.argmax(pbestWf))
        return self.finish(pbest[winner], pbestRaw[winner][0], self.stopped)

    def config(self):
        return {'particles': self.size, 'iterations': self.span, 'inertia': list(self.inertia),
                'cognitive': list(self.cognitive), 'social': list(self.social), 'topology': self.topology, 'patience': self.patience, 'target': self.target}

    def leaders(self, pbest, pbestWf):
        if self.topology == 'global':
            return pbest[np.argmax(pbestWf)][None, :]
        return pbest[self.ring[np.arange(self.size), np.argmax(pbestWf[self.ring], axis=1)]]
