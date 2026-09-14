from .Models.Genetic.index import CMAES
from .Models.PSO.index import PSO
from .Models.DE.index import DifferentialEvolution
from .Models.LSHADE.index import LSHADE
from .Models.LSRTDE.index import LSRTDE
from .Processing.Recorder.index import progress


# FACHADA QUE ESCOLHE O MODELO PELO NOME E DELEGA update()/portrait()/info() — O ÚNICO IMPORT DO NOTEBOOK.
class NatureSelector:
    MODELS = {
        'genetic': CMAES,
        'pso': PSO,
        'de':  DifferentialEvolution,
        'lshade': LSHADE,
        'lsrtde': LSRTDE
    }

    NAMES = tuple(MODELS)

    def __init__(self, name, params, memory=None, n_gaussian=1):
        self.name   = name
        self.params = {**params, 'memory': memory} if memory else params
        self.n      = int(n_gaussian)
        
        if self.n < 1:
            raise ValueError('n_gaussian deve ser >= 1.')

        if self.n > 1 and self.params.get('memory'):
            raise ValueError('n_gaussian > 1 não combina com memory: as corridas estenderiam a mesma campanha.')

        self.best   = None
        self.score  = None
        self.scores = []
        self.optimizer = self.get(self.params)

    def get(self, params):
        return self.MODELS[self.name](**params)

    # CONSECUTIVAS A PARTIR DA SEMENTE DADA, PARA A AMOSTRA INTEIRA SER REPRODUTÍVEL
    def seeds(self):
        seed = self.params.get('seed')
        return [None] * self.n if seed is None else [seed + i for i in range(self.n)]

    def update(self):
        up          = self.optimizer.problem.weight > 0
        self.scores = []
        keep        = None
        total, unit = self.optimizer.budget()
        bar         = progress(total=total * self.n, desc=self.name, unit=unit) if self.params.get('verbose', True) else None

        for seed in self.seeds():
            model        = self.get(self.params if seed is None else {**self.params, 'seed': seed})
            model.shared = bar
            best, score  = model.update()
            self.scores.append(score)

            if keep is None or (score > keep[1]) == up:
                keep = (best, score, model)

        if bar is not None:
            bar.total = bar.n   # quem parou no alvo não gastou a cota: sem isto a barra fecharia pela metade
            bar.close()

        self.best, self.score, self.optimizer = keep
        return self.best, self.score

    # O MODELO NÃO SABE COM QUE NOME FOI ESCOLHIDO, E SEM ISSO NENHUM PAINEL DIZ DE QUEM É
    def portrait(self):
        portrait      = self.optimizer.portrait()
        portrait.name = self.name
        return portrait

    def plotMetrics(self, save=None):
        self.portrait().plotMetrics(save)

    def plotGraph(self, save=None):
        self.portrait().plotGraph(save)

    def plotVariables(self, mode='range', save=None):
        self.portrait().plotVariables(mode, save)

    def plot(self, save=None):
        self.plotMetrics(save if save is None else f'{save}_metrics.png')
        self.plotGraph(save if save is None else f'{save}_graph.png')
        self.plotGraphVariables(save if save is None else f'{save}_variables.png')

    def info(self):
        if self.best is None:
            self.update()

        row  = {k: round(v, 5) if isinstance(v, float) else v for k, v in self.best.items()}
        return {'algorithm': self.name, 'f': self.score, 'stopped': self.optimizer.stopped, **row}

    # PRECISÃO CHEIA E VARIÁVEIS NUM CAMPO SÓ, AO CONTRÁRIO DO info(), QUE É A LINHA ACHATADA DO QUADRO
    def getBest(self):
        if self.best is None:
            self.update()

        return {'algorithm': self.name, 'f': self.score, 'runs': len(self.scores), 'stopped': self.optimizer.stopped, 'variables': self.best}
