import json
import os
import numpy as np
from datetime import datetime


# PERSISTÊNCIA OPCIONAL (memory=<pasta>): state.npz guarda o estado completo e opaco de um run, best.json
# acumula o melhor entre campanhas e history.json registra um objeto por run.
class Memory:
    EVERY = 10

    def __init__(self, path, model, params, problem, recorder=None):
        self.path     = path
        self.model    = model
        self.problem  = problem
        self.recorder = recorder
        self.maximize = problem.weight > 0
        self.params   = params
        self.best     = None
        self.score    = None
        self.genome   = None
        self.runs     = 0
        self.ended    = False
        self.since    = 0
        if path is None:
            return
        os.makedirs(path, exist_ok=True)
        self.read()

    def file(self, name):
        return os.path.join(self.path, name)

    def start(self):
        if self.path is None or not os.path.exists(self.file('state.npz')):
            return None
        d    = np.load(self.file('state.npz'), allow_pickle=False)
        meta = json.loads(str(d['meta']))
        if meta.get('model') != self.model or meta.get('names') != self.problem.names:
            return None   # outro problema/algoritmo na mesma pasta: começa limpo
        self.ended = bool(meta.get('ended'))
        if self.recorder is not None:
            self.recorder.set({k: d[k] for k in d.files if k.startswith('rec')})
        return {k: d[k] for k in d.files if k != 'meta' and not k.startswith('rec')}, meta

    # SEM A MARCA 'ended' (QUE SÓ O commit() GRAVA) A CAMPANHA MORREU NO MEIO E A CHAMADA NOVA SÓ A TERMINA;
    # COM ELA A CHAMADA NOVA É UMA EXTENSÃO, COM CRONOGRAMA PRÓPRIO.
    def origin(self, done):
        return done if self.ended else 0

    def save(self, arrays, meta):
        if self.path is None:
            return
        self.since += 1
        if self.since < self.EVERY:
            return
        self.since = 0
        self.dump(arrays, meta)

    def commit(self, best, score, genome, stopped, arrays, meta):
        if self.path is None:
            return best, score
        self.runs += 1
        improved = self.score is None or (score > self.score if self.maximize else score < self.score)
        if improved:
            self.best, self.score, self.genome = best, score, [float(g) for g in genome]
        stamp  = datetime.now().isoformat(timespec='seconds')
        record = {'model': self.model, 'score': self.score, 'best': self.best, 'genome': self.genome,
                  'names': self.problem.names, 'stopped': stopped, 'runs': self.runs, 'params': self.params, 'updated': stamp}
        self.write('best.json', record)
        self.log({'run': self.runs, 'at': stamp, 'model': self.model, 'score': float(score),
                   'improved': bool(improved), 'stopped': stopped, 'best': best, 'params': self.params})
        self.dump(arrays, {**meta, 'ended': True})   # o estado final persiste, marcado como fechado
        return self.best, self.score

    def read(self):
        if not os.path.exists(self.file('best.json')):
            return
        data = json.load(open(self.file('best.json')))
        if data.get('names') != self.problem.names:
            raise ValueError(f"memory '{self.path}' pertence a outro problema {data.get('names')}; use outra pasta.")
        self.best, self.score, self.genome, self.runs = data['best'], data['score'], data['genome'], data.get('runs', 0)

    def dump(self, arrays, meta):
        meta = {**meta, 'model': self.model, 'names': self.problem.names}
        if self.recorder is not None:
            arrays = {**arrays, **self.recorder.get()}   # histórico junto do estado: o plot sobrevive ao restart
        np.savez(self.file('state.tmp.npz'), meta=json.dumps(meta), **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(self.file('state.tmp.npz'), self.file('state.npz'))   # troca atômica: resiste a crash

    # UM REGISTRO POR RUN: LÊ, ACRESCENTA E REESCREVE INTEIRO, QUE CABE NA MEMÓRIA E VIRA DataFrame DIRETO
    def log(self, row):
        old = json.load(open(self.file('history.json'))) if os.path.exists(self.file('history.json')) else []
        self.write('history.json', old + [row])

    def write(self, name, obj):
        json.dump(obj, open(self.file(name + '.tmp'), 'w'), indent=2, ensure_ascii=False, default=str)
        os.replace(self.file(name + '.tmp'), self.file(name))

    def info(self):
        return {'path': self.path, 'model': self.model, 'runs': self.runs, 'score': self.score, 'best': self.best}
