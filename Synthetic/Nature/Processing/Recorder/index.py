import sys
import numpy as np
from tqdm import tqdm


# BARRA DE TEXTO NO stdout, A MESMA NO TERMINAL E NO NOTEBOOK. A tqdm.auto TROCA POR UM WIDGET HTML DE FUNDO
# BRANCO DENTRO DO JUPYTER, E A tqdm CRUA ESCREVE NO stderr, QUE O JUPYTER PINTA COM O FUNDO DE ERRO.
def progress(iterable=None, **kw):
    return tqdm(iterable, file=sys.stdout, dynamic_ncols=True, mininterval=0.3, **kw)


# HISTÓRICO DA BUSCA: ESTATÍSTICAS POR GERAÇÃO E NUVEM DE AMOSTRAS, ALIMENTADAS NO tick() DE TODO MODELO.
class Recorder:
    SAMPLES = 200      # gerações retidas na nuvem (subamostra temporal p/ o scatter do Plotter)
    PER_GEN = 40       # teto de indivíduos por geração (limita memória; a cobertura do range é feita no Plotter)
    BUDGET  = 1000000  # teto de floats da nuvem inteira: ela entra no state.npz a cada Memory.EVERY gerações
    KEYS    = ('gen', 'nevals', 'min', 'max')

    def __init__(self, genes, span, verbose, desc, unit='gen'):
        self.genes   = genes
        self.span    = span
        self.verbose = verbose
        self.desc    = desc
        self.unit    = unit
        # sem teto a nuvem virava 97% do checkpoint em D=500, e o Plotter desenha 800 pontos por variável
        self.perGen  = max(1, min(self.PER_GEN, self.BUDGET // (2 * self.SAMPLES * len(genes))))
        self.records = {k: [] for k in self.KEYS}
        self.cloudX  = []
        self.cloudY  = []
        self.cloudG  = []
        self.srng    = np.random.default_rng(0)  # isolado do rng da busca: amostrar não altera a otimização

    def bar(self, start=1):
        span = range(start, self.span + 1)
        return progress(span, desc=self.desc, unit=self.unit) if self.verbose else span

    def record(self, gen, nevals, fits):
        # retomada re-registra a geração onde parou: descarta o trecho refeito para não duplicar ponto
        if self.records['gen'] and gen <= self.records['gen'][-1]:
            keep = int(np.searchsorted(self.records['gen'], gen))
            for k in self.KEYS:
                self.records[k] = self.records[k][:keep]
            if not keep:
                self.cloudX, self.cloudY, self.cloudG = [], [], []
        fits = np.asarray(fits)
        self.records['gen'].append(gen)
        self.records['nevals'].append(nevals)
        self.records['min'].append(np.min(fits, axis=0))
        self.records['max'].append(np.max(fits, axis=0))

    # NUVEM DO QUE FOI EXPLORADO, PARA OS SCATTERS: perGen INDIVÍDUOS POR GERAÇÃO, AFINADA PELA METADE AO
    # PASSAR DE 2·SAMPLES GERAÇÕES. A GERAÇÃO VEM DO record(), QUE TODO MODELO CHAMA ANTES DAQUI.
    def sample(self, genomes, fits):
        f    = np.asarray(fits, float)
        y    = f[:, 0] if f.ndim > 1 else f
        pick = np.arange(len(y)) if len(y) <= self.perGen else self.srng.choice(len(y), self.perGen, replace=False)
        # np.take copia: sem isso a nuvem guardaria uma view do array que o modelo sobrescreve in-place
        self.cloudX.append(np.take(np.asarray(genomes, float), pick, axis=0))
        self.cloudY.append(np.take(y, pick))
        self.cloudG.append(self.records['gen'][-1])
        if len(self.cloudX) > 2 * self.SAMPLES:
            self.cloudX, self.cloudY, self.cloudG = self.cloudX[::2], self.cloudY[::2], self.cloudG[::2]

    def cloud(self):
        if not self.cloudX:
            return None, None, None
        gens = np.repeat(self.cloudG, [len(y) for y in self.cloudY])
        return np.vstack(self.cloudX), np.concatenate(self.cloudY), gens

    # VAI JUNTO DO state.npz, SENÃO A RETOMADA PLOTA SÓ O TRECHO DEPOIS DO RESTART. A NUVEM É RAGGED:
    # SAI ACHATADA COM OS ÍNDICES DE CORTE.
    def get(self):
        state = {'rec' + k.capitalize(): np.array(self.records[k]) for k in self.KEYS}
        state['recCloudX'] = np.vstack(self.cloudX) if self.cloudX else np.zeros((0, len(self.genes)))
        state['recCloudY'] = np.concatenate(self.cloudY) if self.cloudY else np.zeros(0)
        state['recCloudG'] = np.array(self.cloudG, int)
        state['recSplit']  = np.cumsum([len(y) for y in self.cloudY])[:-1].astype(int)
        return state

    def set(self, state):
        if 'recGen' not in state:
            return
        for k in self.KEYS:
            self.records[k] = list(state['rec' + k.capitalize()])
        cuts        = np.asarray(state['recSplit'], int)
        self.cloudX = [a for a in np.split(state['recCloudX'], cuts) if len(a)]
        self.cloudY = [a for a in np.split(state['recCloudY'], cuts) if len(a)]
        self.cloudG = list(state.get('recCloudG', []))
        if len(self.cloudG) != len(self.cloudY):
            self.cloudX, self.cloudY, self.cloudG = [], [], []  # campanha gravada antes da geração entrar na nuvem
