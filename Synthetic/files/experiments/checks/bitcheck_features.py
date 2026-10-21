# AS ETAPAS DA SESSÃO ANTERIOR LIGADAS (CONFIG DO C0_v1, COLUNA COM HACHURA E TRAÇO MORTO, CORPOS, FALHA SUTIL COM ARRASTO):
# A CÓPIA NOVA TEM DE REPETIR O ARQUIVO DE ANTES DA MUDANÇA (index_before.py) BIT A BIT
import sys, json, importlib.util, time
import numpy as np, torch
torch.set_num_threads(4)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SyntheticGenerator

Old, New = load('index_before.py', 'antes'), load('index.py', 'novo')
specs    = json.load(open('/home/grva-mint/Projects/Falhas/Synthetic/files/experiments/datasets/specs.json'))
configs  = {name: {k: tuple(v) if isinstance(v, list) else v for k, v in specs[name]['regions']['tiles']['params'].items()} for name in ('C0_v1', 'B2_column', 'C1_subtle', 'B1_bodies')}
ok = True
for name, params in configs.items():
    for seed in (7, 2026):
        out = []
        for cls in (Old, New):
            g = cls(seed=seed); g.set({**params, 'device': 'cpu'})
            img, msk = g.get()
            out.append((img, msk, np.random.get_state()[1].copy()))
        (i0, m0, s0), (i1, m1, s1) = out
        same = np.array_equal(m0, m1) and float(np.abs(i0 - i1).max()) <= 1e-8 and np.array_equal(s0, s1)
        ok &= same
        print(f'{name:12s} semente {seed:5d}: máscara igual {np.array_equal(m0, m1)}, |Δimagem| máx {float(np.abs(i0 - i1).max()):.2e}, rng igual {np.array_equal(s0, s1)}', flush=True)
print('TUDO IGUAL' if ok else 'DIFERENÇA ENCONTRADA')
