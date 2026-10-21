# PROVA DE COMPATIBILIDADE: O SyntheticGenerator NOVO CONTRA O DO git HEAD, MESMOS PARÂMETROS E SEMENTE -> MÁSCARA IDÊNTICA,
# IMAGEM COM |Δ| <= 1e-8 E O MESMO ESTADO DO np.random DEPOIS DO TILE (NENHUM SORTEIO A MAIS)
import sys, json, importlib.util, time
import numpy as np, torch
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
ROOT = '/home/grva-mint/Projects/Falhas'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SyntheticGenerator

New = load(f'{ROOT}/Synthetic/index.py', 'novo')
Old = load(sys.argv[1], 'head')
configs = {
    'padrao': {},
    'dataset_74.json': {k: tuple(v) if isinstance(v, list) else v for k, v in json.load(open(f'{ROOT}/Dataset/dataset_74/synthetic.json')).items()},
    'sem_falha': {'faultCount': (0, 1)},
    'estresse_listrica': {'faultCount': (10, 16), 'faultCurveProb': 1.0, 'faultThrow': (5, 40), 'faultCurveMax': 12.0},
    'fast': {'fast': True},
}
ok = True
for name, params in configs.items():
    for seed in (7, 2026):
        out = []
        for cls in (Old, New):
            g = cls(seed=seed)
            g.set(params)
            t0 = time.time()
            img, msk = g.get()
            out.append((img, msk, np.random.get_state()[1].copy(), time.time() - t0))
        (i0, m0, s0, t0), (i1, m1, s1, t1) = out
        same = np.array_equal(m0, m1) and float(np.abs(i0 - i1).max()) <= 1e-8 and np.array_equal(s0, s1)
        ok &= same
        print(f'{name:18s} semente {seed:5d}: máscara igual {np.array_equal(m0, m1)}, |Δimagem| máx {float(np.abs(i0 - i1).max()):.2e}, rng igual {np.array_equal(s0, s1)}, rótulo {m0.mean():.4f}, {t0:.0f}s/{t1:.0f}s', flush=True)
print('TUDO IGUAL' if ok else 'DIFERENÇA ENCONTRADA')
