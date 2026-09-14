import os
import random
import warnings
import numpy as np
from ..Problem.index import Problem
from ..Recorder.index import Recorder, progress
from ..Memory.index import Memory


# EARLY STOPPING POR PACIÊNCIA E POR ALVO, SOBRE O SINAL PONDERADO EM QUE MAIOR É SEMPRE MELHOR.
class Stopper:
    TOL = 1e-9

    def __init__(self, patience, target=None, weight=1.0):
        self.patience = patience
        self.target   = None if target is None else weight * target   # alvo entra como f cru
        self.best     = -np.inf
        self.bad      = 0

    def check(self, signal):
        if self.target is not None and signal >= self.target:
            return True
        if self.patience is None:
            return False
        if signal > self.best + self.TOL:
            self.best, self.bad = signal, 0
            return False
        self.bad += 1
        return self.bad >= self.patience

    # VAI JUNTO DO state.npz: SEM A CONTAGEM DE ESTAGNAÇÃO A RETOMADA REINICIA A PACIÊNCIA
    def get(self):
        return {'best': float(self.best), 'bad': int(self.bad)}

    def set(self, snap):
        if snap is None:
            return  # campanha gravada antes deste campo existir
        self.best, self.bad = float(snap['best']), int(snap['bad'])


# COMO A POPULAÇÃO É AVALIADA: SERIAL, THREADS OU PROCESSOS.
class Pool:
    THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')

    def __init__(self, workers, backend, seed=None):
        self.workers = (os.cpu_count() or 1) if workers == -1 else int(workers)
        self.backend = backend
        self.seed    = seed
        self.pool    = None
        if backend == 'vector' and self.workers > 1:
            warnings.warn("backend='vector' avalia a população de uma vez; workers é ignorado.", stacklevel=3)
        # o OpenBLAS lê essas variáveis ao inicializar, antes deste ponto: daqui só dá para avisar
        if backend == 'process' and self.workers > 1 and not any(os.environ.get(v) for v in self.THREADS):
            warnings.warn(f"backend='process' com BLAS multithread: exporte {self.THREADS[0]}=1 (e "
                          f'{self.THREADS[1]}=1) antes de abrir o Python, senão os workers disputam núcleos.',
                          stacklevel=3)

    def start(self):
        if self.workers <= 1 or self.backend == 'vector':
            return map
        if self.backend == 'thread':
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
            return self.pool.map
        try:
            from multiprocess import Pool as ProcessPool
        except ImportError:
            from multiprocessing import Pool as ProcessPool
            warnings.warn("Instale 'multiprocess' para backend='process' em notebooks.", stacklevel=3)
        if self.seed is not None:
            warnings.warn("Reprodutibilidade não garantida com backend='process'.", stacklevel=3)
        self.pool = ProcessPool(self.workers)
        return self.pool.map

    def stop(self):
        if self.pool is None:
            return
        if hasattr(self.pool, 'shutdown'):
            self.pool.shutdown(wait=True)
        else:
            self.pool.close()
            self.pool.join()
        self.pool = None


# CLASSE-BASE DOS MODELOS: open() → resume() → begin() → tick() a cada geração → finish(). O MODELO ENTRA
# COM MODEL/DESC/UNIT, self.size/self.span, config() PARA O best.json E self.pack PARA O CHECKPOINT.
class Engine:
    UNIT   = 'gen'
    XLABEL = 'Geração'

    def __init__(self, objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target):
        self.problem    = Problem(objective, variables, maximize, constraints, backend)
        self.patience   = patience
        self.target     = target
        self.seed       = seed
        self.memoryPath = memory
        self.workers    = workers
        self.backend    = backend
        self.verbose    = verbose
        self.shared     = None   # barra emprestada pelo seletor: uma só para as n corridas do algoritmo

    def open(self):
        self.recorder = Recorder(self.problem.genes, self.span, self.verbose, self.DESC, self.UNIT)
        self.memory   = Memory(self.memoryPath, self.MODEL, self.config(), self.problem, self.recorder)
        self.stopper  = Stopper(self.patience, self.target, self.problem.weight)
        self.pool     = Pool(self.workers, self.backend, self.seed)
        self.stopped  = self.span
        self.map      = self.pool.start()
        return self.reset()

    # A objective É CÓDIGO DO USUÁRIO E PODE SORTEAR PELO random OU PELO np.random GLOBAL: OS TRÊS SÃO
    # SEMEADOS, E É ISSO QUE FAZ seed= VALER PARA O RUN INTEIRO E A RETOMADA SAIR BIT-IDÊNTICA.
    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        return self.rng

    def snap(self):
        return {'np': self.rng.bit_generator.state, 'py': list(random.getstate())}

    def restore(self, snap):
        self.rng.bit_generator.state = snap['np']
        py = snap['py']
        random.setstate((py[0], tuple(py[1]), py[2]))

    def score(self, X):
        return self.problem.score(X, self.map)

    # MUDAR A POPULAÇÃO ABRE CAMPANHA NOVA; SEM key NÃO HÁ CHECAGEM, QUE É O CASO DA L-SHADE. E UM state.npz
    # DE OUTRA VERSÃO DO MODELO TEM O MESMO 'model' MAS OUTROS ARRAYS: SEM ISSO A RETOMADA MORRE NO KeyError
    def resume(self, key=None):
        state = self.memory.start()
        if state and not set(self.ARRAYS) <= set(state[0]):
            state = None
        if state and key and state[0][key].shape[0] != self.size:
            state = None
        self.fresh = state is None
        if self.fresh:
            return None
        arrays, self.meta = state
        self.restore(self.meta['rng'])
        return arrays

    # FECHA A PREPARAÇÃO E DEVOLVE O TETO DO CICLO; A RETOMADA NÃO RE-GRAVA A GERAÇÃO ZERO
    def begin(self, done, X, raw, nevals, signal, budget):
        self.origin = self.memory.origin(done)
        if self.fresh:
            self.recorder.record(0, nevals, raw)
            self.recorder.sample(X, raw)
        elif not self.origin:
            self.stopper.set(self.meta.get('stop'))  # extensão reinicia a paciência, retomada não
        self.stopper.check(signal)
        return self.origin + budget

    # PROGRESSO DE 0 A 1 CONTADO DA ORIGEM: É ISSO QUE FAZ AS RAMPAS RECOMEÇAREM NUMA EXTENSÃO
    def phase(self, gen):
        return (gen - self.origin - 1) / max(self.span - 1, 1)

    # QUANTO UMA CORRIDA GASTA E EM QUE UNIDADE — O SELETOR MULTIPLICA ISSO PELAS n CORRIDAS DA BARRA DELE
    def budget(self):
        return self.span, self.UNIT

    # BARRA EM AVALIAÇÕES DOS MODELOS QUE CONTAM AVALIAÇÃO. COM A EMPRESTADA NÃO SE ABRE OUTRA
    def cycle(self, total, initial):
        if self.shared is not None:
            return self.shared
        return progress(total=total, initial=initial, desc=self.DESC, unit='ev') if self.verbose else None

    # A EMPRESTADA SEGUE PARA A PRÓXIMA CORRIDA, ENTÃO SÓ SE FECHA A QUE É DESTA
    def shut(self):
        if self.bar is not None and self.bar is not self.shared:
            self.bar.close()

    def track(self, start):
        if self.shared is None:
            self.bar = self.recorder.bar(start)
            return self.bar
        self.bar = self.shared
        return self.steps(start)

    # COM A BARRA EMPRESTADA O LAÇO ANDA NUM range CRU: QUEM AVANÇA A BARRA, UMA GERAÇÃO POR VEZ, É AQUI
    def steps(self, start):
        for gen in range(start, self.recorder.span + 1):
            yield gen
            self.shared.update(1)

    # HISTÓRICO, NUVEM, CHECKPOINT E A PERGUNTA DA PARADA — True QUANDO É PARA PARAR
    def tick(self, gen, nevals, X, raw, signal, **show):
        self.recorder.record(gen, nevals, raw)
        self.recorder.sample(X, raw)
        if self.memory.path:
            self.memory.save(self.pack(), self.mark(gen))
        if self.verbose:
            # set_postfix() custa 10 us e redesenha a barra toda geração; o texto pronto sai por 0,06 us,
            # e quem redesenha passa a ser o update() da barra, no mininterval do progress()
            extra = ''.join(f', {k}={v}' for k, v in show.items())
            self.bar.set_postfix_str(f'best={signal * self.problem.weight:.6g}{extra}', refresh=False)
        if not self.stopper.check(signal):
            return False
        self.stopped = gen
        if self.verbose and self.bar is not self.shared:
            self.bar.set_description(f'Early stopping @ {self.UNIT} {gen}')
        return True

    def mark(self, gen):
        return {'done': gen, 'rng': self.snap(), 'stop': self.stopper.get()}

    # DECODIFICA O VENCEDOR E DEIXA A MEMÓRIA DIZER SE ELE É O MELHOR DE TODAS AS CAMPANHAS
    def finish(self, genome, score, done):
        self.best      = self.problem.decode(genome)
        self.bestScore = float(score)
        self.best, self.bestScore = self.memory.commit(self.best, self.bestScore, genome, self.stopped,
                                                       self.pack(), self.mark(done))
        return self.best, self.bestScore

    # REPARO DE BORDA DA L-SHADE (Tanabe & Fukunaga), USADO TAMBÉM PELA DE: O GENE FORA DA CAIXA VAI PARA O MEIO
    # ENTRE O PAI E A BORDA. O np.clip EMPILHA A POPULAÇÃO INTEIRA SOBRE A FACE E CONGELA A COORDENADA, PORQUE A
    # DIFERENÇA ENTRE DOIS INDIVÍDUOS GRUDADOS NO MESMO LIMITE É ZERO.
    def repair(self, X, V):
        lo, up = self.problem.low, self.problem.up
        return np.where(V < lo, 0.5 * (lo + X), np.where(V > up, 0.5 * (up + X), V))

    # MÁSCARA DO CROSSOVER BINOMIAL: CADA GENE VEM DO MUTANTE COM PROBABILIDADE CR, MAIS UM GENE SORTEADO
    # QUE VEM SEMPRE DELE. CR POR INDIVÍDUO, QUE É COMO A L-SHADE E A L-SRTDE SORTEIAM.
    def mask(self, shape, CR, rng):
        n, d = shape
        m    = rng.random((n, d)) < CR[:, None]
        m[np.arange(n), rng.integers(d, size=n)] = True
        return m

    # SORTEIO POR REJEIÇÃO: REAMOSTRA QUEM COLIDIR COM QUALQUER DOS ÍNDICES JÁ TIRADOS. O draw RECEBE QUANTOS
    # FALTAM, ENTÃO SERVE TANTO PARA SORTEIO UNIFORME QUANTO PARA SORTEIO COM PRESSÃO DE POSTO.
    def apart(self, draw, n, *avoid):
        r = draw(n)
        while True:
            clash = np.zeros(n, bool)
            for a in avoid:
                clash |= r == a
            if not clash.any():
                return r
            r[clash] = draw(int(clash.sum()))

    # ESCALAR VIRA CONSTANTE, PAR VIRA RAMPA LINEAR
    def pair(self, value):
        return tuple(map(float, value)) if isinstance(value, (list, tuple)) else (float(value), float(value))

    # ÍNDICE UNIFORME SOBRE OS n-1 QUE NÃO SÃO avoid, SEM LAÇO DE REJEIÇÃO
    def pick(self, size, avoid, rng):
        r = rng.integers(self.size - 1, size=size)
        return r + (r >= avoid)

    # IMPORT LOCAL PORQUE O Portrait CARREGA O matplotlib, QUE UMA OTIMIZAÇÃO SEM GRÁFICO NÃO PRECISA PAGAR
    def portrait(self):
        from ..Portrait.index import Portrait

        if not hasattr(self, 'recorder'):   # sem update() nesta sessão: reconstrói só do que a memory guardou
            self.recorder = Recorder(self.problem.genes, self.span, False, self.DESC, self.UNIT)
            self.memory   = Memory(self.memoryPath, self.MODEL, self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore, self.XLABEL)
