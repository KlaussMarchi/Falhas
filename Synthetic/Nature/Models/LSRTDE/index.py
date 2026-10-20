import numpy as np
from ...Processing.Engine.index import Engine


# L-SRTDE (Stanovov & Semenkin 2024), VENCEDORA DO CEC'2024. DUAS POPULAÇÕES DO MESMO TAMANHO (L-NTADE,
# Stanovov et al. 2022): a *newest*, que guarda quem venceu há pouco, e a *top*, um anel de elite reescrito
# em círculo pelos sucessos. O ALVO NÃO É O i-ÉSIMO INDIVÍDUO E SIM UM SORTEADO DO ANEL, ENTÃO A BUSCA NÃO
# FICA PRESA NUM PAREAMENTO FIXO. A NOVIDADE DE 2024 É TROCAR A MEMÓRIA DE SUCESSO DO F PELA *TAXA* DE
# SUCESSO: mF = 0,4 + 0,25·tanh(5·SR) E O RECORTE pbest = 0,7·N·e^(-7·SR) SAEM DIRETO DELA — QUANDO A GERAÇÃO
# RENDE, O PASSO CRESCE E A GANÂNCIA CAI; QUANDO TRAVA, ENCOLHE O PASSO E APERTA A ELITE. SÓ O CR MANTÉM
# MEMÓRIA HISTÓRICA, E ELA GUARDA A TAXA DE CRUZAMENTO *REALIZADA*, NÃO A SORTEADA.
#
# O C++ DE REFERÊNCIA É ASSÍNCRONO INDIVÍDUO A INDIVÍDUO — UM SUCESSO JÁ ENTRA NO ANEL E PODE SER PAI AINDA
# NA MESMA GERAÇÃO. UM LAÇO 1 A 1 EM PYTHON NÃO FECHA NO ORÇAMENTO DESTE TRABALHO, ENTÃO A GERAÇÃO ANDA EM
# BLOCOS DE CHUNK: VETORIZADA POR DENTRO, ASSÍNCRONA POR FORA. FAZER A GERAÇÃO INTEIRA DE UMA VEZ CUSTA A F2
# EM D=50 (0 EM 15 CORRIDAS NO ALVO CONTRA 15 EM 15), QUE É EXATAMENTE ONDE A PROPAGAÇÃO DENTRO DA GERAÇÃO
# MAIS RENDE: UNIMODAL, POPULAÇÃO GRANDE E PROGRESSO CONSTANTE.
class LSRTDE(Engine):
    MODEL    = 'lsrtde'
    DESC     = 'L-SRTDE'
    ARRAYS   = ('X', 'raw', 'T', 'rawT', 'mCR', 'elite', 'eliteRaw')
    H        = 5
    N_MIN    = 4
    SIGMA_F  = 0.02    # o F sai de uma normal estreita: a taxa de sucesso é que move a média
    SIGMA_CR = 0.05
    MCR0     = 1.0     # memória do CR começa saturada; ela só desce quando o sucesso pedir
    SR0      = 0.5     # taxa de sucesso suposta antes da primeira geração
    RANK     = 3.0     # decaimento do peso por posto na escolha do vetor de diferença vindo do anel
    CHUNK    = 50      # indivíduos por bloco; o C++ propaga sucesso de 1 em 1, e cada bloco recupera um passo
    GREED    = (0.7, 7.0)   # recorte pbest = GREED[0]·N·e^(-GREED[1]·SR)

    def __init__(self, objective, variables, maximize=True, population=None, generations=300,
                 patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)

        self.size = 20 * self.problem.nVars if population is None else int(population)
        if self.size < 8:
            raise ValueError('population deve ser >= 8 (ou None para 20·nVars).')

        self.span   = int(generations)
        self.maxNfe = self.size * self.span

    def update(self):
        rng = self.open()

        try:
            arrays = self.resume()
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw, self.nfe = self.score(X), self.size
                seat    = np.argsort(-(raw[:, 0] * self.problem.weight))   # o anel nasce ordenado
                T, rawT = X[seat].copy(), raw[seat].copy()
                mCR     = np.full(self.H, self.MCR0)
                elite   = (T[0].copy(), rawT[0].copy())
                self.k  = self.pf = gen = 0
                self.sr = self.SR0
            else:
                X, raw, T, rawT, mCR, best, bestRaw = (arrays[k] for k in self.ARRAYS)
                elite    = (best, bestRaw)
                self.nfe, self.k, self.pf = self.meta['nfe'], self.meta['k'], self.meta['pf']
                self.sr, gen             = self.meta['sr'], self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw, 'T': T, 'rawT': rawT, 'mCR': mCR,
                                 'elite': elite[0], 'eliteRaw': elite[1]}

            done  = self.nfe
            spent = self.nfe
            limit = self.begin(done, T, rawT, len(T), float(elite[1][0] * self.problem.weight), self.maxNfe)
            self.bar = self.cycle(limit, self.nfe)

            while self.nfe + len(T) <= limit:
                gen  += 1
                n     = len(T)
                wfX   = raw[:, 0] * self.problem.weight
                ix    = np.argsort(-wfX)                                  # newest, melhor primeiro
                it    = np.argsort(-(rawT[:, 0] * self.problem.weight))   # top, melhor primeiro

                # mF sobe com a taxa de sucesso e o recorte da elite cai com ela: os dois botões da L-SRTDE
                meanF = 0.4 + 0.25 * np.tanh(5.0 * self.sr)
                pn    = max(2, int(n * self.GREED[0] * np.exp(-self.GREED[1] * self.sr)))
                w     = np.exp(-self.RANK * np.arange(n) / n)
                w    /= w.sum()

                seat  = rng.integers(n, size=n)                                              # alvo: um do anel
                prand = self.apart(lambda m: ix[rng.integers(pn, size=m)], n, seat)          # elite do newest
                r1    = self.apart(lambda m: it[rng.choice(n, size=m, p=w)], n, prand)       # anel, por posto
                r2    = self.apart(lambda m: ix[rng.integers(n, size=m)], n, prand, r1)      # newest, uniforme

                F    = self.scale(meanF, n, rng)
                CR   = np.clip(mCR[rng.integers(self.H, size=n)] + self.SIGMA_CR * rng.standard_normal(n), 0.0, 1.0)
                mask = self.mask((n, self.problem.nVars), CR, rng)
                cr, gain = [], []

                # a geração anda em blocos: dentro do bloco tudo é vetorizado, e entre blocos o anel já vem
                # atualizado pelos sucessos anteriores. É assim que a assincronia do C++ entra sem laço 1 a 1.
                for lo in range(0, n, self.CHUNK):
                    blk  = np.arange(lo, min(lo + self.CHUNK, n))
                    pai  = T[seat[blk]]
                    V    = pai + F[blk, None] * (X[prand[blk]] - pai) + F[blk, None] * (T[r1[blk]] - X[r2[blk]])
                    U    = np.where(mask[blk], self.respawn(V, rng), pai)

                    rawU      = self.score(U)
                    wfU       = rawU[:, 0] * self.problem.weight
                    self.nfe += len(blk)
                    wfP       = rawT[seat[blk], 0] * self.problem.weight
                    hits      = np.nonzero(wfU >= wfP)[0]   # o empate conta como sucesso, como na referência
                    if not len(hits):
                        continue

                    # o ganho sai antes da escrita: o pai pode ser justamente a casa que o ponteiro vai ocupar
                    gain.append(np.abs(wfU[hits] - wfP[hits]))
                    cr.append(mask[blk[hits]].mean(1))
                    # o sucesso entra no anel na posição do ponteiro, por cima do que estava lá, bom ou não
                    slot = (self.pf + np.arange(len(hits))) % n
                    T[slot], rawT[slot] = U[hits], rawU[hits]
                    self.pf = int((self.pf + len(hits)) % n)
                    X, raw = np.vstack([X, U[hits]]), np.vstack([raw, rawU[hits]])

                self.sr = sum(len(c) for c in cr) / n
                if cr:
                    self.remember(mCR, np.concatenate(cr), np.concatenate(gain))

                top = int(np.argmax(rawT[:, 0] * self.problem.weight))
                if rawT[top, 0] * self.problem.weight > elite[1][0] * self.problem.weight:
                    elite = (T[top].copy(), rawT[top].copy())
                X, raw, T, rawT = self.reduce(X, raw, T, rawT, rng)
                signal = float(elite[1][0] * self.problem.weight)

                if self.bar is not None:
                    self.bar.update(n)
                used, spent = self.nfe - spent, self.nfe
                if self.tick(gen, used, T, rawT, signal, pop=len(T), sr=f'{self.sr:.2f}'):
                    break
            self.stopped = gen
            self.shut()
        finally:
            self.pool.stop()

        return self.finish(elite[0], elite[1][0], gen)

    # O TETO É AVALIAÇÃO, NÃO GERAÇÃO: A POPULAÇÃO ENCOLHE E O NÚMERO DE GERAÇÕES NÃO É FIXO
    def budget(self):
        return self.maxNfe, 'ev'

    def config(self):
        return {'population': self.size, 'generations': self.span, 'patience': self.patience,
                'target': self.target, 'maxNfe': self.maxNfe}

    # ALÉM DA GERAÇÃO, O CHECKPOINT LEVA O ORÇAMENTO, OS DOIS PONTEIROS E A TAXA DE SUCESSO CORRENTE
    def mark(self, gen):
        return {**super().mark(gen), 'nfe': self.nfe, 'k': self.k, 'pf': self.pf, 'sr': float(self.sr)}

    # F_i ~ N(mF, 0,02) REAMOSTRADO ATÉ CAIR EM [0, 1]. A NORMAL ESTREITA É DE PROPÓSITO: NA L-SRTDE QUEM
    # CARREGA A ADAPTAÇÃO É A MÉDIA, QUE VEM DA TAXA DE SUCESSO, E NÃO A CAUDA DA DISTRIBUIÇÃO.
    def scale(self, mean, n, rng):
        F = mean + self.SIGMA_F * rng.standard_normal(n)
        while ((F < 0.0) | (F > 1.0)).any():
            bad = (F < 0.0) | (F > 1.0)
            F[bad] = mean + self.SIGMA_F * rng.standard_normal(int(bad.sum()))
        return F

    # GENE FORA DA CAIXA VOLTA SORTEADO DENTRO DELA — É O TRATAMENTO DO CÓDIGO DE REFERÊNCIA DA L-SRTDE, E
    # NÃO O MEIO-CAMINHO DA L-SHADE: COM O ALVO VINDO DO ANEL, RESSORTEAR REPÕE DIVERSIDADE EM VEZ DE TIRÁ-LA
    def respawn(self, V, rng):
        out = (V < self.problem.low) | (V > self.problem.up)
        return np.where(out, rng.uniform(self.problem.low, self.problem.up, V.shape), V) if out.any() else V

    # A CÉLULA NOVA É A MÉDIA ENTRE A DE LEHMER DOS SUCESSOS E O QUE JÁ ESTAVA LÁ (iL-SHADE). SOMA ZERO
    # DEVOLVE 1,0, NÃO O VALOR TERMINAL DA L-SHADE: AQUI CR = 0 EM TODO SUCESSO SIGNIFICA CRUZAMENTO CHEIO.
    def remember(self, mCR, sCR, gains):
        w   = gains / gains.sum() if gains.sum() > 0 else np.full(len(gains), 1.0 / len(gains))
        low = float(np.sum(w * sCR))
        mCR[self.k] = 0.5 * (mCR[self.k] + (1.0 if abs(low) <= 1e-8 else float(np.sum(w * sCR ** 2)) / low))
        self.k = (self.k + 1) % self.H

    # LPSR NAS DUAS POPULAÇÕES. NO ANEL SAI O PIOR SEM REORDENAR O RESTO, PORQUE A ORDEM É O QUE O PONTEIRO
    # CIRCULAR PERCORRE; NA newest FICAM OS MELHORES ENTRE OS ANTIGOS E OS SUCESSOS DESTA GERAÇÃO.
    def reduce(self, X, raw, T, rawT, rng):
        n = max(self.N_MIN, int((self.N_MIN - self.size) / self.maxNfe * (self.nfe - self.origin) + self.size))
        if n < len(T):
            seat    = np.sort(np.argsort(-(rawT[:, 0] * self.problem.weight))[:n])
            T, rawT = T[seat], rawT[seat]
            self.pf = int(self.pf % n)
        keep = np.argsort(-(raw[:, 0] * self.problem.weight))[:len(T)]
        return X[keep], raw[keep], T, rawT
