# INJEÇÃO DE FALHA EM TILE REAL (OBJETIVO SEM RÓTULO CANDIDATO E3): TILES VIVOS DOS PATCHES DE DEV, NA ESCALA DOS .dat, COM UMA
# FALHA SINTÉTICA DE REJEITO CONHECIDO (0,25–1,25 PERÍODO DO PRÓPRIO TILE; METADE COM ARRASTO, COMO AS FALHAS DO MARLIM) APLICADA AO
# DADO REAL. make() GRAVA injection/<n>.npz;
# run() MEDE NUM MODELO O RECALL DA FALHA INJETADA (CONTEXTO REAL) E O QUANTO ELE MARCA O TILE REAL SEM INJEÇÃO (FUNDO).
# NADA DISSO USA A INTERPRETAÇÃO: SÓ O DADO SÍSMICO E A FALHA QUE NÓS MESMOS PUSEMOS.
import os, sys, json, glob
import numpy as np, pandas as pd
import scipy.ndimage as ndi

ROOT   = '/home/grva-mint/Projects/Falhas'
EXP    = f'{ROOT}/Synthetic/files/experiments'
OUT    = f'{EXP}/injection'
N_TILE = 20      # POR PATCH DE DEV
BANDS  = ((300, 450), (450, 700), (700, 950), (950, 1200))
ALIVE  = 0.0552
WATER  = 0.50263


def plane(rng, shape):
    dip, strike = np.radians(rng.uniform(50, 80)), rng.uniform(0, 2 * np.pi)
    normal = np.array([np.sin(dip) * np.cos(strike), np.cos(dip) * rng.choice([-1.0, 1.0]), np.sin(dip) * np.sin(strike)])
    along  = np.array([-normal[2], 0.0, normal[0]]) / np.hypot(normal[0], normal[2])
    down   = np.cross(normal, along)
    down   = down * np.sign(down[1]) / np.linalg.norm(down)
    return rng.uniform(0.35, 0.65, 3) * np.array(shape), normal, along, down


# UMA FALHA NO TILE REAL: O BLOCO DO LADO POSITIVO DA NORMAL DESCE AO LONGO DO MERGULHO, REJEITO GAUSSIANO EM VOLTA DO CENTRO; COM
# ARRASTO c, OS DOIS BLOCOS FLEXIONAM PERTO DO PLANO (A CONTA DO applyDrag DO GERADOR) E O SALTO NO PLANO É (1 - c) DO REJEITO
def inject(tile, period, rng):
    p0, normal, along, down = plane(rng, tile.shape)
    grid  = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in tile.shape], indexing='ij')
    d     = [g - c for g, c in zip(grid, p0)]
    dist  = sum(n * x for n, x in zip(normal, d))
    sa    = sum(a * x for a, x in zip(along, d))
    sd    = sum(a * x for a, x in zip(down, d))
    ratio = rng.uniform(0.25, 1.25)
    throw = np.clip(ratio * period, 2.0, 25.0) * np.exp(-(sa ** 2 + sd ** 2) / (2 * rng.uniform(40, 70) ** 2))
    drag  = rng.uniform(0.0, 0.6) if rng.random() < 0.5 else 0.0
    width = rng.uniform(4, 12)
    shift = throw * np.where(dist > 0, 1 - 0.5 * drag * np.exp(-np.clip(dist, 0, None) / width), 0.5 * drag * np.exp(np.clip(dist, None, 0) / width))
    coords = [g - shift * c for g, c in zip(grid, down)]
    faulted = ndi.map_coordinates(tile, coords, order=1, mode='nearest').astype(np.float32)
    a     = faulted - WATER
    live  = np.sqrt(ndi.gaussian_filter(a * a, 4)) > ALIVE
    label = (np.abs(dist) <= 1.0) & ((1 - drag) * throw > max(1.0, 0.15 * period)) & live
    return faulted, label, ratio, drag


def make():
    sys.path.insert(0, f'{EXP}/profile')
    from ruler import BlockSampler
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261020)
    for pid in ('1200', '1300'):
        sampler = BlockSampler(f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles')
        table   = pd.read_pickle(f'{EXP}/profile/marlim_{pid}.pkl')
        table   = table[(table.live > 0.8) & table.period.notna()].assign(zc=lambda t: t.z0 + 32)
        picks   = pd.concat([g.sample(min(len(g), N_TILE // len(BANDS)), random_state=int(pid)) for _, g in table.groupby(pd.cut(table.zc, [b[0] for b in BANDS] + [BANDS[-1][1]]), observed=True)])
        for _, row in picks.iterrows():
            path = f'{OUT}/{pid}_{row["name"][:-4]}.npz'
            if os.path.exists(path):
                continue
            tile         = sampler.get(row['name'])
            faulted, lab, ratio, drag = inject(tile, float(row.period), rng)
            np.savez_compressed(path, real=tile.astype(np.float16), faulted=faulted.astype(np.float16), label=lab, period=row.period, zc=row.zc, ratio=ratio, drag=drag)
            print(pid, row['name'], f'período {row.period:.1f}, rejeito {ratio:.2f} período, arrasto {drag:.2f}, rótulo {lab.mean():.4f}', flush=True)


# NO MODELO: RECALL POR VOXEL DA FALHA INJETADA, MARCAÇÃO DO TILE REAL (FUNDO VIVO) E O QUE A INJEÇÃO MUDOU LONGE DELA
def run(modelDir, tag, results=f'{EXP}/results.jsonl'):
    import torch
    sys.path.insert(0, f'{EXP}/harness')
    import marlim
    network, info = marlim.loadNetwork(modelDir)
    rows = []

    with torch.no_grad():
        for path in sorted(glob.glob(f'{OUT}/*.npz')):
            data  = np.load(path)
            probs = [torch.sigmoid(network.model(torch.from_numpy(data[k].astype(np.float32))[None, None].to(network.device))).float().cpu().numpy()[0, 0] for k in ('real', 'faulted')]
            real, faulted, label = probs[0] >= 0.5, probs[1] >= 0.5, data['label']
            a    = data['real'].astype(np.float32) - WATER
            live = np.sqrt(ndi.gaussian_filter(a * a, 4)) > ALIVE
            far  = ~ndi.binary_dilation(label, iterations=4) & live
            rows.append({'recall': float(faulted[label].mean()) if label.any() else np.nan, 'fundo': float(real[live].mean()), 'mudanca': float((real != faulted)[far].mean()), 'zc': float(data['zc']), 'ratio': float(data['ratio']), 'drag': float(data['drag'])})

    table = pd.DataFrame(rows)
    out   = {'tag': tag, 'modelo': modelDir, 'metodo': 'injecao', 'inj_recall': round(float(table.recall.mean()), 4), 'inj_fundo': round(float(table.fundo.mean()), 5), 'inj_mudanca': round(float(table.mudanca.mean()), 5), **{f'inj_recall_z{lo}': round(float(table[(table.zc >= lo) & (table.zc < hi)].recall.mean()), 4) for lo, hi in BANDS}, 'inj_recall_sutil': round(float(table[table.ratio < 0.6].recall.mean()), 4), 'inj_recall_arrasto': round(float(table[table.drag > 0].recall.mean()), 4)}
    with open(results, 'a') as file:
        file.write(json.dumps(out) + '\n')
    print(json.dumps(out), flush=True)
    del network
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    make()
