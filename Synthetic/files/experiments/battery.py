# BATERIA DE SÓSIAS: TILES SINTÉTICOS FIXOS, CADA CASO COM UM SÓ TIPO DE ESTRUTURA SEM FALHA (E UM CONTROLE COM FALHAS), JÁ NA
# ESCALA DA REDE (MIN-MAX DO VOLUME, EIXOS (x, z, y)). make() GRAVA battery/<caso>_<semente>.npy; run() MEDE QUANTO UM MODELO
# MARCA EM CADA CASO (FALSO POSITIVO NOS SEM FALHA, RECALL POR VOXEL NO CONTROLE) E ACRESCENTA AO results.jsonl
import os, sys, json, glob, time
import numpy as np

ROOT  = '/home/grva-mint/Projects/Falhas'
EXP   = f'{ROOT}/Synthetic/files/experiments'
CASES = {
    'camadas': {'faultCount': (0, 1)},
    'corpos': {'faultCount': (0, 1), 'bodyCount': (2, 4)},
    'dobras': {'faultCount': (0, 1), 'foldSigma': (6, 20), 'foldAmplitude': (-40, 40)},
    'coluna': {'faultCount': (0, 1), 'seafloorProb': 1.0, 'deadZoneProb': 1.0, 'seafloorDepth': (0.1, 0.3), 'deadZoneDepth': (0.4, 0.7), 'deadZoneHatch': (0.1, 0.2)},
    'ganho': {'faultCount': (0, 1), 'gainField': 0.6},
    'periodo_longo': {'faultCount': (0, 1), 'waveletFreq': (33, 55)},
    'ruido_baixo': {'faultCount': (0, 1), 'noiseLevel': (0.0, 0.15)},
    'lentes': {'faultCount': (0, 1), 'lensSigma': (8, 50), 'lensCut': (0.3, 0.6), 'lensEdge': (0.2, 0.5)},
    'corpos_camadas': {'faultCount': (0, 1), 'bodyCount': (2, 4), 'bodyLayers': 1.0},
    'falhas': {},
}
SEEDS = (41, 42, 43, 44)


def make():
    import torch
    sys.path.insert(0, ROOT)
    from Synthetic.index import SyntheticGenerator
    torch.set_num_threads(int(os.environ.get('THREADS', 8)))
    os.makedirs(f'{EXP}/battery', exist_ok=True)

    for case, params in CASES.items():
        for seed in SEEDS:
            path = f'{EXP}/battery/{case}_{seed}.npy'
            if os.path.exists(path):
                continue
            g = SyntheticGenerator(seed=seed)
            g.set(params)
            img, msk = g.get()
            img = (img - img.min()) / (img.max() - img.min())
            np.save(path, np.stack([np.transpose(img, (0, 2, 1)), np.transpose(msk, (0, 2, 1)).astype(np.float32)]).astype(np.float32))
            print(case, seed, flush=True)


def run(modelDir, tag, results=f'{EXP}/results.jsonl'):
    import torch
    sys.path.insert(0, f'{EXP}/harness')
    import marlim
    network, info = marlim.loadNetwork(modelDir)
    rows = {}

    with torch.no_grad():
        for path in sorted(glob.glob(f'{EXP}/battery/*.npy')):
            case    = os.path.basename(path).rsplit('_', 1)[0]
            img, msk = np.load(path)
            prob    = torch.sigmoid(network.model(torch.from_numpy(img)[None, None].to(network.device))).float().cpu().numpy()[0, 0]
            pred, m = prob >= 0.5, msk > 0
            rows.setdefault(case, []).append((float(pred.mean()), float((pred & m).sum() / max(m.sum(), 1))))

    out = {'tag': tag, 'modelo': modelDir, 'metodo': 'bateria', **{f'bat_{c}': round(float(np.mean([r[0] for r in v])), 5) for c, v in rows.items()}, 'bat_recall_falhas': round(float(np.mean([r[1] for r in rows.get('falhas', [(0, 0)])])), 4)}
    with open(results, 'a') as file:
        file.write(json.dumps(out) + '\n')
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    make()
