import numpy as np
from ...Processing.Engine.index import Engine


# O SLOT 'genetic' DO SELETOR: CMA-ES (Hansen & Ostermeier 2001) COM REINÍCIO IPOP (Auger & Hansen 2005), NO
# LUGAR DO AG DE SBX + MUTAÇÃO NÃO-UNIFORME. NO LUGAR DE CRUZAR INDIVÍDUOS, A CADA GERAÇÃO ELA REAPRENDE A
# GAUSSIANA QUE OS GEROU — MÉDIA, PASSO E MATRIZ DE COVARIÂNCIA — ENTÃO A NUVEM SE ALONGA SOZINHA NA DIREÇÃO
# DO VALE E FICA INVARIANTE A ROTAÇÃO E A CONDICIONAMENTO, QUE É ONDE O AG PERDIA FEIO. O IPOP DOBRA A
# POPULAÇÃO A CADA COLAPSO, E É ISSO QUE A SEGURA NAS MULTIMODAIS: SEM ELE A CMA-ES AFUNDA NO PRIMEIRO
# MÍNIMO LOCAL E NÃO SAI MAIS.
class CMAES(Engine):
    MODEL  = 'genetic'
    DESC   = 'CMA-ES'
    ARRAYS = ('X', 'raw', 'm', 'C', 'pc', 'ps', 'trail', 'elite', 'eliteRaw')
    SIGMA0 = 0.3     # passo inicial em frações da aresta da caixa (Hansen)
    TOL_X  = 1e-12   # nuvem colapsada: maior eixo abaixo desta fração da aresta
    TOL_F  = 1e-12   # platô: melhor da geração preso nesta faixa relativa
    WINDOW = 20      # gerações olhadas para decidir platô
    IPOP   = 2       # fator de crescimento da população a cada reinício

    def __init__(self, objective, variables, maximize=True, population=None, generations=300, sigma=None,
                 patience=None, target=None, constraints=None, seed=None, memory=None, workers=1,
                 backend='thread', verbose=True):
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)

        # λ = 4 + 3·ln(d) é o default de Hansen, calibrado para gastar o mínimo de avaliações por geração
        self.size = 4 + int(3 * np.log(self.problem.nVars)) if population is None else int(population)
        if self.size < 4:
            raise ValueError('population deve ser >= 4 (ou None para 4 + 3·ln(nVars)).')

        self.span   = int(generations)
        self.maxNfe = self.size * self.span
        self.range  = float(np.mean(self.problem.up - self.problem.low))
        self.sigma0 = self.SIGMA0 * self.range if sigma is None else float(sigma)

    def update(self):
        rng      = self.open()
        self.nfe = 0

        try:
            arrays = self.resume()
            if arrays is None:
                self.born(self.size, rng)
                X, gen   = self.ask(rng), 0
                raw      = self.score(X)
                self.nfe = self.lam
                wf       = raw[:, 0] * self.problem.weight
                crown    = int(np.argmax(wf))
                elite    = (X[crown].copy(), raw[crown].copy())
            else:
                self.born(int(self.meta['lam']), rng)
                self.m, self.C, self.pc, self.ps = (arrays[k] for k in ('m', 'C', 'pc', 'ps'))
                X, raw                 = arrays['X'], arrays['raw']
                self.sigma, self.count = self.meta['sigma'], self.meta['count']
                self.trail             = list(arrays['trail'])
                self.nfe, gen          = self.meta['nfe'], self.meta['done']
                elite                  = (arrays['elite'], arrays['eliteRaw'])
                wf                     = raw[:, 0] * self.problem.weight
                self.decompose(True)   # B e Dv não vão ao npz: saem do C gravado
            self.pack = lambda: {'X': X, 'raw': raw, 'm': self.m, 'C': self.C, 'pc': self.pc, 'ps': self.ps,
                                 'trail': np.array(self.trail), 'elite': elite[0], 'eliteRaw': elite[1]}

            signal   = float(elite[1][0] * self.problem.weight)
            limit    = self.begin(self.nfe, X, raw, self.lam, signal, self.maxNfe)
            self.bar = self.cycle(limit, self.nfe)

            # a geração custa lam inteiro: entrar com menos que isso no caixa estoura o MaxFES do enunciado
            while self.nfe + self.lam <= limit:
                gen += 1
                # o reinício vem antes do tell: a amostra na mão saiu da distribuição velha e envenenaria a nova
                if self.stalled():
                    self.born(min(self.IPOP * self.lam, limit - self.nfe), rng)
                else:
                    self.tell(X, wf)
                    self.decompose()

                X         = self.ask(rng)
                raw       = self.score(X)
                wf        = raw[:, 0] * self.problem.weight
                self.nfe += self.lam
                top       = int(np.argmax(wf))
                self.trail.append(float(wf[top]))

                if wf[top] > elite[1][0] * self.problem.weight:
                    elite = (X[top].copy(), raw[top].copy())
                signal = float(elite[1][0] * self.problem.weight)

                if self.bar is not None:
                    self.bar.update(self.lam)
                if self.tick(gen, self.lam, X, raw, signal, pop=self.lam):
                    break
            self.stopped = gen   # o open() deixou self.span aqui, que não é o nº de gerações do ciclo
            self.shut()
        finally:
            self.pool.stop()

        return self.finish(elite[0], elite[1][0], gen)

    # O TETO É AVALIAÇÃO, NÃO GERAÇÃO: O IPOP DOBRA O lam A CADA REINÍCIO E A GERAÇÃO MUDA DE PREÇO
    def budget(self):
        return self.maxNfe, 'ev'

    def config(self):
        return {'population': self.size, 'generations': self.span, 'sigma': self.sigma0,
                'patience': self.patience, 'target': self.target, 'maxNfe': self.maxNfe}

    # O CHECKPOINT LEVA OS ESCALARES DA DISTRIBUIÇÃO; AS MATRIZES VÃO PELO pack()
    def mark(self, gen):
        return {**super().mark(gen), 'nfe': self.nfe, 'sigma': float(self.sigma),
                'lam': int(self.lam), 'count': int(self.count)}

    # DISTRIBUIÇÃO ZERADA. O IPOP CHAMA DE NOVO COM lam MAIOR, E COMO TODA TAXA DEPENDE DE lam, TUDO É
    # RECALCULADO AQUI — INCLUSIVE OS PESOS, QUE MUDAM DE TAMANHO JUNTO COM μ.
    def born(self, lam, rng):
        d   = self.problem.nVars
        mu  = lam // 2
        w   = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w  /= w.sum()
        eff = 1.0 / float(np.sum(w ** 2))

        self.lam, self.mu, self.w, self.mueff = lam, mu, w, eff
        self.cc    = (4.0 + eff / d) / (d + 4.0 + 2.0 * eff / d)
        self.cs    = (eff + 2.0) / (d + eff + 5.0)
        self.c1    = 2.0 / ((d + 1.3) ** 2 + eff)
        self.cmu   = min(1.0 - self.c1, 2.0 * (eff - 2.0 + 1.0 / eff) / ((d + 2.0) ** 2 + eff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((eff - 1.0) / (d + 1.0)) - 1.0) + self.cs
        self.chiN  = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

        self.m     = rng.uniform(self.problem.low, self.problem.up)
        self.sigma = self.sigma0
        self.C     = np.eye(d)
        self.B     = np.eye(d)
        self.Dv    = np.ones(d)
        self.pc    = np.zeros(d)
        self.ps    = np.zeros(d)
        self.count = 0
        self.eigen = self.nfe
        self.trail = []

    # ponytail: fora da caixa é grampeado e o ponto grampeado é o que alimenta o tell — o ótimo do CEC'14 é
    # interior, então basta; ótimo colado na borda pediria a penalidade de contorno do Hansen.
    def ask(self, rng):
        z = rng.standard_normal((self.lam, self.problem.nVars))
        return np.clip(self.m + self.sigma * (z * self.Dv) @ self.B.T, self.problem.low, self.problem.up)

    # OS μ MELHORES REPOSICIONAM A MÉDIA; O CAMINHO ps AJUSTA O PASSO E O CAMINHO pc ALONGA A COVARIÂNCIA NA
    # DIREÇÃO QUE VEM DANDO CERTO — É O TERMO DE POSTO 1 QUE APRENDE O VALE, E O DE POSTO μ QUE APRENDE A FORMA.
    def tell(self, X, wf):
        self.count += 1
        pick        = np.argsort(-wf)[:self.mu]
        old, self.m = self.m, self.w @ X[pick]
        yw          = (self.m - old) / self.sigma

        # C^(-1/2)·yw sai da própria decomposição: B·diag(1/D)·Bᵀ·yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ ((self.B.T @ yw) / self.Dv))
        walk    = float(np.linalg.norm(self.ps))
        # freio hsig: logo após um reinício o caminho ainda é curto, e sem o freio σ inflaria de uma vez
        hs      = walk / np.sqrt(1 - (1 - self.cs) ** (2 * self.count)) < (1.4 + 2.0 / (self.problem.nVars + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw

        Y      = (X[pick] - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (not hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= float(np.exp((self.cs / self.damps) * (walk / self.chiN - 1)))

    # O eigh É O(d³) E A ATUALIZAÇÃO DE C É O(μd²): NA CADÊNCIA DO Hansen ELE SE DILUI E O RUN VOLTA A SER O(d²)
    def decompose(self, force=False):
        if not force and self.nfe - self.eigen <= self.lam / (10.0 * self.problem.nVars * (self.c1 + self.cmu)):
            return
        self.eigen   = self.nfe
        self.C       = np.triu(self.C) + np.triu(self.C, 1).T   # a atualização acumula assimetria numérica
        vals, self.B = np.linalg.eigh(self.C)
        self.Dv      = np.sqrt(np.maximum(vals, 1e-20))         # autovalor negativo é só arredondamento

    # GATILHO DO IPOP, AS DUAS PARADAS DE Hansen QUE PEGAM QUASE TODO TRAVAMENTO: OU A NUVEM ENCOLHEU ABAIXO
    # DA PRECISÃO ÚTIL, OU O MELHOR DA GERAÇÃO NÃO SAI DE UMA FAIXA RELATIVA HÁ WINDOW GERAÇÕES.
    def stalled(self):
        if self.sigma * float(self.Dv.max()) < self.TOL_X * self.range:
            return True
        if len(self.trail) < self.WINDOW:
            return False
        span = self.trail[-self.WINDOW:]
        return max(span) - min(span) <= self.TOL_F * max(abs(max(span)), 1.0)
