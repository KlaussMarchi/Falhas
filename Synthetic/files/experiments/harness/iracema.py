# HARNESS DO IRACEMA: AS CLASSES DO Blocks/Iracema/Analysis.ipynb (EXECUTADAS DAS CÉLULAS) E A PREDIÇÃO DE UMA INLINE SÓ COM OS
# TILES QUE A COBREM. O IRACEMA NÃO TEM INTERPRETAÇÃO: SAEM A SÍSMICA E A PROBABILIDADE DA INLINE, PARA O PERFIL SEM RÓTULO
import os, re, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marlim import ROOT, NB as MARLIM, loadNetwork

FOLDER = f'{ROOT}/Blocks/Iracema/files'
CUT    = (r'^grid = TileGrid', r"^\{'shape': grid")


def loadNotebook():
    nb   = json.load(open(f'{ROOT}/Blocks/Iracema/Analysis.ipynb'))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    ns   = {}
    exec(code[0].replace("sys.path.append('../../Model')", '').replace('from Network.index import ModelNetwork', ''), ns)
    for src in code:
        if 'class TileGrid' in src:
            exec('\n'.join(line for line in src.split('\n') if not any(re.match(p, line) for p in CUT)), ns)
    return ns


NB = loadNotebook()


def predictInline(modelDir, inline):
    network, info = loadNetwork(modelDir)
    grid          = NB['TileGrid'](f'{FOLDER}/patch_metadata.json')
    lo, hi        = grid.ov, [p - o for p, o in zip(grid.size, grid.ov)]
    prob, seismic = np.zeros(grid.shape[1:], np.float32), np.zeros(grid.shape[1:], np.float32)

    with torch.no_grad():
        for name in grid.names(inline):
            (d, h, w), (cd, ch, cw) = grid.core(name)
            tile = NB['getDAT'](f'{FOLDER}/patches/{name}', grid.size)
            p    = torch.sigmoid(network.model(torch.from_numpy(tile[None, None]).to(network.device))).float().cpu().numpy()[0, 0]
            prob[h:h + ch, w:w + cw]    = p[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][inline - d, :ch, :cw]
            seismic[h:h + ch, w:w + cw] = tile[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][inline - d, :ch, :cw]

    return prob, seismic


# SEÇÃO VIVA PELA MESMA ENERGIA DO EXTRATOR QUE O marlim.liveMask USA
def liveMask(seismic):
    return MARLIM['FaultStickExtractor']().energy(seismic) > 0.05


def equalize(prob, live, frac=0.01):
    t = min(max(float(np.quantile(prob[live], 1 - frac)), 1e-6), 1 - 1e-6)
    return np.power(np.clip(prob, 0, 1), np.log(0.5) / np.log(t)).astype(np.float32)


if __name__ == '__main__':
    modelDir, tag, inline = sys.argv[1], sys.argv[2], int(sys.argv[3])
    prob, seismic = predictInline(modelDir, inline)
    out = f'{ROOT}/Synthetic/files/experiments/sections'
    np.save(f'{out}/{tag}_iracema{inline}.npy', prob.astype(np.float16))
    np.save(f'{out}/iracema{inline}_seismic.npy', seismic)
    print(tag, inline, prob.shape, 'marca 0,5:', round(float((prob[liveMask(seismic)] >= 0.5).mean()), 4))
