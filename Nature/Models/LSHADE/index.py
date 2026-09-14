import numpy as np
from ...Processing.Engine.index import Engine


# L-SHADE (Tanabe & Fukunaga 2014), VENCEDORA DO CEC'2014: DE current-to-pbest/1/bin COM ARQUIVO DE
# DERROTADOS (Zhang & Sanderson 2009), MEMÓRIA DE H SUCESSOS PARA F/CR (Tanabe & Fukunaga 2013) E LPSR,
# QUE ENCOLHE A POPULAÇÃO LINEARMENTE AO LONGO DO ORÇAMENTO — EXPLORA COM MUITOS, REFINA COM POUCOS.
class LSHADE(Engine):
    MODEL  = 'lshade'
    DESC   = 'L-SHADE'
    ARRAYS = ('X', 'raw', 'arch', 'mF', 'mCR')
    H      = 6
    N_MIN  = 4

    def __init__(self, objective, variables, maximize=True, population=None, generations=300, pbest=0.11,
                 archive=2.6, patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        if not 0.0 < pbest <= 1.0:
            raise ValueError('pbest deve estar em (0, 1].')

        if archive < 0.0:
            raise ValueError('archive (taxa do arquivo) deve ser >= 0.')

        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)

        self.size = 18 * self.problem.nVars if population is None else int(population)
        if self.size < 8:
            raise ValueError('population deve ser >= 8 (ou None para 18·nVars).')

        self.span    = int(generations)
        self.maxNfe  = self.size * self.span
        self.pbest   = float(pbest)
        self.archive = float(archive)

    def update(self):
        rng = self.open()

        try:
            arrays = self.resume()
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw, self.nfe = self.score(X), self.size
                mF, mCR = np.full(self.H, 0.5), np.full(self.H, 0.5)
                arch    = np.empty((0, self.problem.nVars))
                self.k  = gen = 0
            else:
                X, raw, arch, mF, mCR = (arrays[k] for k in self.ARRAYS)
                self.nfe, self.k, gen = self.meta['nfe'], self.meta['k'], self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw, 'arch': arch, 'mF': mF, 'mCR': mCR}

            done = self.nfe   # origem do ciclo antes de reinflar, senão a extensão ganha crédito

            # extensão de campanha fechada: sem reinflar, o ciclo novo rodaria inteiro com N_MIN indivíduos
            if self.memory.origin(done) and len(X) < self.size:
                fresh     = rng.uniform(self.problem.low, self.problem.up, (self.size - len(X), self.problem.nVars))
                X, raw    = np.vstack([X, fresh]), np.vstack([raw, self.score(fresh)])
                self.nfe += len(fresh)

            wf    = raw[:, 0] * self.problem.weight
            crown = int(np.argmax(wf))
            elite = (X[crown].copy(), raw[crown].copy())
            spent = self.nfe

            # o orçamento aqui é avaliação, não geração: a L-SHADE encolhe a população e faz mais gerações
            limit    = self.begin(done, X, raw, len(X), float(wf[crown]), self.maxNfe)
            self.bar = self.cycle(limit, self.nfe)

            while self.nfe < limit:
                gen += 1
                idx = rng.integers(self.H, size=len(X))
                F   = self.scale(mF[idx], rng)
                CR  = np.clip(mCR[idx] + 0.1 * rng.standard_normal(len(X)), 0.0, 1.0)
                U   = self.cross(X, self.repair(X, self.mutate(X, wf, F, arch, rng)), CR, rng)

                rawU      = self.score(U)
                wfU       = rawU[:, 0] * self.problem.weight
                self.nfe += len(X)
                gain      = wfU - wf
                better    = gain > 0    # sucesso estrito: alimenta arquivo e adaptação de F/CR
                replace   = gain >= 0   # aceitar empate move a população por platôs/regiões inviáveis

                if better.any():
                    arch = np.vstack([arch, X[better]]) if len(arch) else X[better].copy()
                    self.remember(mF, mCR, F[better], CR[better], gain[better])

                X[replace], raw[replace], wf[replace] = U[replace], rawU[replace], wfU[replace]
                top = int(np.argmax(wf))

                if wf[top] > elite[1][0] * self.problem.weight:
                    elite = (X[top].copy(), raw[top].copy())
                X, raw, wf, arch = self.reduce(X, raw, wf, arch, self.nfe - self.origin, rng)
                signal = float(elite[1][0] * self.problem.weight)

                if self.bar is not None:
                    self.bar.update(len(U))
                # o gasto vem do contador, não do tamanho da população: o LPSR muda a população no caminho
                used, spent = self.nfe - spent, self.nfe
                if self.tick(gen, used, X, raw, signal, pop=len(X)):
                    break
            self.stopped = gen   # o open() deixou self.span aqui, que não é o nº de gerações do ciclo
            self.shut()
        finally:
            self.pool.stop()

        return self.finish(elite[0], elite[1][0], gen)

    # O TETO É AVALIAÇÃO, NÃO GERAÇÃO: O LPSR ENCOLHE A POPULAÇÃO E O NÚMERO DE GERAÇÕES NÃO É FIXO
    def budget(self):
        return self.maxNfe, 'ev'

    def config(self):
        return {'population': self.size, 'generations': self.span, 'pbest': self.pbest,
                'archive': self.archive, 'patience': self.patience, 'target': self.target, 'maxNfe': self.maxNfe}

    # ALÉM DA GERAÇÃO, O CHECKPOINT CARREGA O ORÇAMENTO GASTO E O PONTEIRO DA MEMÓRIA DE SUCESSO
    def mark(self, gen):
        return {**super().mark(gen), 'nfe': self.nfe, 'k': self.k}

    def mutate(self, X, wf, F, arch, rng):
        n      = len(X)
        pn     = max(2, round(self.pbest * n))
        order  = np.argsort(-wf)
        xpbest = X[order[(rng.random(n) * pn).astype(int)]]   # cada alvo puxa para um pbest do seu topo
        XA     = np.vstack([X, arch]) if len(arch) else X     # r2 vem de população ∪ arquivo
        r1     = self.distinct(n, n, rng, np.arange(n))
        r2     = self.distinct(n, len(XA), rng, np.arange(n), r1)
        return X + F[:, None] * (xpbest - X) + F[:, None] * (X[r1] - XA[r2])

    def distinct(self, n, high, rng, *avoid):
        return self.apart(lambda m: rng.integers(high, size=m), n, *avoid)

    # F_i ~ Cauchy(m, 0.1): REAMOSTRA OS <= 0 E SATURA EM 1 (Tanabe & Fukunaga)
    def scale(self, m, rng):
        F = m + 0.1 * rng.standard_cauchy(len(m))
        while (F <= 0).any():
            bad = F <= 0
            F[bad] = m[bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        return np.minimum(F, 1.0)

    def cross(self, X, V, CR, rng):
        return np.where(self.mask(X.shape, CR, rng), V, X)

    # O PESO DO SUCESSO É O GANHO DE APTIDÃO, E AS DUAS CÉLULAS ENTRAM PELA MÉDIA DE LEHMER. O CR SÓ PASSOU À
    # LEHMER NA L-SHADE (2014) — CONTRA A MÉDIA ARITMÉTICA DA SHADE ISSO VALE TRÊS VEZES O ERRO DA F9.
    def remember(self, mF, mCR, sF, sCR, gains):
        w = gains / gains.sum()
        mF[self.k], mCR[self.k] = self.lehmer(w, sF), self.lehmer(w, sCR)
        self.k = (self.k + 1) % self.H

    # MÉDIA DE LEHMER PONDERADA. SOMA ZERO SÓ ACONTECE NO CR, E MARCA A CÉLULA COMO TERMINAL: O -1 ATRAVESSA
    # O CLIP DO SORTEIO E QUEM SORTEAR ESSA CÉLULA CRUZA COM CR = 0, UM GENE SÓ VINDO DO MUTANTE.
    def lehmer(self, w, s):
        low = float(np.sum(w * s))
        return -1.0 if low == 0.0 else float(np.sum(w * s ** 2)) / low

    def reduce(self, X, raw, wf, arch, nfe, rng):
        # LPSR (Tanabe & Fukunaga 2014, eq. 8): N cai linearmente de size a N_MIN ao longo do ciclo
        n = max(self.N_MIN, int(round((self.N_MIN - self.size) / self.maxNfe * nfe + self.size)))
        if n < len(X):
            keep = np.argsort(-wf)[:n]
            X, raw, wf = X[keep], raw[keep], wf[keep]
        cap = int(round(self.archive * len(X)))
        if len(arch) > cap:
            arch = arch[rng.choice(len(arch), cap, replace=False)] if cap else np.empty((0, X.shape[1]))
        return X, raw, wf, arch
