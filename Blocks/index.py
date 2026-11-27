import os, re, gc, json, gzip, base64, cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from itertools import chain
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Network.index import ModelNetwork


# ESCALAR OU LISTA VIRA TUPLA DE 3 EIXOS, PARA JANELA E TILE ENTRAREM NO MESMO FORMATO
def asTriple(value, default=None):
    value = default if value is None else value
    value = [value] * 3 if np.isscalar(value) else list(value)

    if len(value) != 3:
        raise ValueError(f'esperado 3 eixos, recebido {value}')

    return tuple(int(v) for v in value)


# LÊ O .DAT CRU NO SHAPE DO TILE (NÃO É FIXO EM 128³)
def getDAT(path, shape):
    data = np.fromfile(path, dtype=np.float32)

    if data.size != int(np.prod(shape)):
        raise ValueError(f'{os.path.basename(path)}: {data.size} valores, esperado {tuple(shape)}')

    return data.reshape(shape)


def gauss(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


# SERVE OS TILES UM A UM, NO SHAPE DECLARADO NO PATCH_METADATA
class PredictDataset(Dataset):
    def __init__(self, paths, shape):
        self.paths = paths
        self.shape = asTriple(shape)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.from_numpy(getDAT(self.paths[index], self.shape))


# GRADE DE TILES DE UM PATCH_METADATA: QUAIS TILES EXISTEM E ONDE O NÚCLEO DE CADA UM CAI NO VOLUME
class TileGrid:
    def __init__(self, metaPath):
        meta        = json.load(open(metaPath))
        self.size   = asTriple(meta['patch_size'])
        self.ov     = asTriple(meta['overlap_voxels'])
        self.shape  = tuple(meta['original_shape'])
        self.counts = tuple(meta['grid_shape'])
        self.stride = tuple(p - 2 * o for p, o in zip(self.size, self.ov))
        self.last   = [n // s for n, s in zip(self.shape, self.stride)]

    # NOMES DOS TILES DO GRID; COM inline, SÓ OS QUE TÊM O NÚCLEO SOBRE ELA
    def names(self, inline=None):
        names = [f'patch_d{d:03d}_h{h:03d}_w{w:03d}.dat' for d, h, w in np.ndindex(*self.counts)]
        return names if inline is None else [n for n in names if self.covers(n, inline)]

    # TILES SEM ARQUIVO INTEIRO EM folder (UM .DAT CORTADO NO MEIO CONTA COMO FALTANDO)
    def missing(self, folder, inline=None):
        size = 4 * int(np.prod(self.size))
        return [n for n in self.names(inline) if not (os.path.exists(f'{folder}/{n}') and os.path.getsize(f'{folder}/{n}') == size)]

    # ORIGEM E TAMANHO DO NÚCLEO DE UM TILE NO VOLUME, PELO ÍNDICE NO NOME; O ÚLTIMO DE CADA EIXO ENCOSTA NO FIM
    def core(self, name):
        index  = map(int, re.match(r'patch_d(\d+)_h(\d+)_w(\d+)\.dat', name).groups())
        origin = [max(0, n - s) if i == last else i * s for i, last, n, s in zip(index, self.last, self.shape, self.stride)]
        sizes  = [min(s, n - o) for s, n, o in zip(self.stride, self.shape, origin)]
        return origin, sizes

    def covers(self, name, inline):
        (d, _, _), (cd, _, _) = self.core(name)
        return d <= inline < d + cd

    # COPIA O NÚCLEO DE UM TILE PARA O VOLUME OU, COM inline, PARA A SEÇÃO 2D DELA
    def place(self, target, name, tile, inline=None):
        (d, h, w), (cd, ch, cw) = self.core(name)
        lo   = self.ov
        core = tile[lo[0]:lo[0] + cd, lo[1]:lo[1] + ch, lo[2]:lo[2] + cw]

        if inline is None:
            target[d:d + cd, h:h + ch, w:w + cw] = core
            return

        target[h:h + ch, w:w + cw] = core[inline - d]

    # MONTA O VOLUME (OU SÓ A INLINE) PELOS NÚCLEOS DOS TILES DE folder; UM TILE FALTANDO LEVANTA ERRO EM VEZ DE DEIXAR BURACO ZERADO
    def reconstruct(self, folder, desc='montando', inline=None):
        missing = self.missing(folder, inline)

        if missing:
            raise FileNotFoundError(f'{folder}: faltam {len(missing)} de {len(self.names(inline))} tiles')

        volume = np.zeros(self.shape if inline is None else self.shape[1:], np.float32)

        for name in tqdm(self.names(inline), desc=desc):
            self.place(volume, name, getDAT(f'{folder}/{name}', self.size), inline)

        return volume


# RESUMO DE UM BLOCO: GRADE, TILES FALTANDO EM folder E OS VALORES NUMA AMOSTRA DE n TILES ESPALHADOS PELA GRADE
def getBlockInfo(grid, folder, n=100):
    missing = set(grid.missing(folder))
    present = [name for name in grid.names() if name not in missing]
    sample  = [present[int(i)] for i in np.linspace(0, len(present) - 1, min(n, len(present)))]
    stats   = np.array([[tile.min(), tile.max(), tile.mean(), np.square(tile).mean()] for tile in (getDAT(f'{folder}/{name}', grid.size).astype(np.float64) for name in sample)])
    mean    = stats[:, 2].mean()

    info = {
        'pasta': folder,
        'shape': grid.shape,
        'tile': grid.size,
        'grade': grid.counts,
        'tiles': len(grid.names()),
        'faltando': len(missing),
        'amostra': len(sample),
        'min': stats[:, 0].min(),
        'max': stats[:, 1].max(),
        'media': mean,
        'desvio': np.sqrt(stats[:, 3].mean() - mean ** 2)
    }
    return pd.DataFrame([info])


# VOLUME INTEIRO DO BLOCO PELOS NÚCLEOS, GRAVADO EM cache NA PRIMEIRA VEZ; context > 0 (SÓ NUMA GRADE DE UMA CAMADA DE INLINES, COMO OS PATCHES
# DO MARLIM) GUARDA TAMBÉM context INLINES REAIS DE CADA LADO DO NÚCLEO, A VIZINHANÇA QUE O TILE TEM E O NÚCLEO NÃO
def getVolume(grid, folder, cache, context=0):
    if os.path.exists(cache):
        return np.load(cache, mmap_mode='r')

    if context and (grid.counts[0] != 1 or context > grid.ov[0]):
        raise ValueError(f'contexto de {context} inlines pede uma camada só de tiles e no máximo {grid.ov[0]}')

    missing = grid.missing(folder)
    if missing:
        raise FileNotFoundError(f'{folder}: faltam {len(missing)} de {len(grid.names())} tiles')

    volume = np.zeros((grid.shape[0] + 2 * context, *grid.shape[1:]), np.float32)
    lo     = grid.ov

    for name in tqdm(grid.names(), desc=f'volume {os.path.basename(cache)}'):
        (d, h, w), (cd, ch, cw) = grid.core(name)
        tile = getDAT(f'{folder}/{name}', grid.size)
        volume[d:d + cd + 2 * context, h:h + ch, w:w + cw] = tile[lo[0] - context:lo[0] + cd + context, lo[1]:lo[1] + ch, lo[2]:lo[2] + cw]

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, volume)
    return np.load(cache, mmap_mode='r')


# CORTA UM VOLUME EM TILES NO FORMATO DOS BLOCOS: NÚCLEO DE size - 2 * overlap NO MEIO DO TILE, O ÚLTIMO DE CADA EIXO ENCOSTADO NO FIM E BORDA
# ESPELHADA (np.pad 'reflect', COMO O GERADOR DOS TILES DO F3); GRAVA OS .dat E O patch_metadata.json QUE O TileGrid LÊ
def writeTiles(volume, folder, meta, metaPath=None, size=128, overlap=32):
    stride = size - 2 * overlap
    shape  = tuple(int(n) for n in volume.shape)
    counts = [int(np.ceil(n / stride)) for n in shape]
    names  = [f'patch_d{d:03d}_h{h:03d}_w{w:03d}.dat' for d, h, w in np.ndindex(*counts)]
    meta   = {'original_shape': list(shape), 'patch_size': size, 'overlap_percent': 100.0 * overlap / size, 'overlap_voxels': overlap, 'do_overlap': True, 'stride': stride, 'grid_shape': counts, 'patch_files': names, 'dtype': 'float32', 'total_patches': len(names), **meta}

    metaPath = metaPath or f'{folder}/patch_metadata.json'
    os.makedirs(folder, exist_ok=True)
    json.dump(meta, open(metaPath, 'w'), indent=2)
    grid   = TileGrid(metaPath)
    padded = np.pad(np.asarray(volume, np.float32), overlap, mode='reflect')

    for name in tqdm(names, desc=f'tiles {os.path.basename(folder)}'):
        (d, h, w), _ = grid.core(name)
        padded[d:d + size, h:h + size, w:w + size].tofile(f'{folder}/{name}')

    return grid


# SÍSMICA DE UMA INLINE, MONTADA SÓ PELOS TILES QUE A COBREM NA PRIMEIRA VEZ E LIDA DO CACHE DEPOIS
def getSeismic(grid, folder, cache, inline):
    if not 0 <= inline < grid.shape[0]:
        raise ValueError(f'inline {inline} fora do bloco (0 a {grid.shape[0] - 1})')

    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, grid.reconstruct(folder, f'sísmica da inline {inline}', inline))

    return np.load(cache)


# JANELA DESLIZANTE SOBRE UM TILE, COM PESO DE HANNING NA EMENDA DAS JANELAS
class SlidingWindow:
    BUDGET    = 128 ** 3   # VOXELS POR LOTE NA GPU
    MAX_BATCH = 256

    def __init__(self, window, shape, overlap):
        self.tile    = asTriple(shape)
        self.window  = asTriple(window)
        self.overlap = float(overlap)

        self.shape  = tuple(max(t, w) for t, w in zip(self.tile, self.window))   # TILE MENOR QUE A JANELA -> REFLECT
        self.pad    = tuple(s - t for s, t in zip(self.shape, self.tile))
        self.starts = [self.getStarts(s, w) for s, w in zip(self.shape, self.window)]

        self.positions = [(d, h, w) for d in self.starts[0] for h in self.starts[1] for w in self.starts[2]]
        self.weight    = self.getWeight(self.window)
        self.batch     = int(np.clip(self.BUDGET // int(np.prod(self.window)), 1, self.MAX_BATCH))

    # INÍCIO DE CADA JANELA NUM EIXO; A ÚLTIMA ENCOSTA NO FIM DO TILE
    def getStarts(self, size, window):
        step = max(int(round(window * (1 - self.overlap))), 1)
        return sorted(set(list(range(0, size - window + 1, step)) + [size - window]))

    # PESO 3D SEPARÁVEL, ~1 NO CENTRO DA JANELA E ~0 NA BORDA, PARA A EMENDA NÃO VIRAR DEGRAU
    @staticmethod
    def getWeight(window):
        d, h, w = [np.hanning(n + 2)[1:-1].astype(np.float32) for n in window]
        return np.maximum(d[:, None, None] * h[None, :, None] * w[None, None, :], 1e-3)

    # PROBABILIDADE DO TILE INTEIRO, SOMANDO AS JANELAS PONDERADAS PELO PESO
    def process(self, network, tile):
        volume = tile.numpy() if torch.is_tensor(tile) else np.asarray(tile)
        volume = np.pad(volume, [(0, p) for p in self.pad], mode='reflect') if any(self.pad) else volume
        volume = torch.from_numpy(np.ascontiguousarray(volume, np.float32)).to(network.device)

        weight = torch.from_numpy(self.weight).to(network.device)
        total  = torch.zeros((network.classes, *self.shape), dtype=torch.float32, device=network.device)
        norm   = torch.zeros(self.shape, dtype=torch.float32, device=network.device)
        wd, wh, ww = self.window

        with torch.no_grad():
            for start in range(0, len(self.positions), self.batch):
                chunk  = self.positions[start:start + self.batch]
                batch  = torch.stack([volume[d:d + wd, h:h + wh, w:w + ww] for d, h, w in chunk]).unsqueeze(1)
                logits = network.model(batch)
                probs  = torch.softmax(logits, dim=1) if network.multiclass else torch.sigmoid(logits)

                for i, (d, h, w) in enumerate(chunk):
                    total[:, d:d + wd, h:h + wh, w:w + ww] += probs[i].float() * weight
                    norm[d:d + wd, h:h + wh, w:w + ww]     += weight

        probs = (total / norm).cpu().numpy().astype(np.float32)
        probs = probs[0] if network.classes == 1 else probs
        return probs[..., :self.tile[0], :self.tile[1], :self.tile[2]]

    def __repr__(self):
        return f'SlidingWindow(janela={self.window}, tile={self.tile}, overlap={self.overlap}, janelas/tile={len(self.positions)}, batch={self.batch})'


# UM MODELO SALVO APLICADO AOS TILES DE UM BLOCO: A REDE SÓ É CARREGADA QUANDO HÁ TILE A PREDIZER
class BlockPredictor:
    def __init__(self, modelPath, overlap, window=None, skipDone=True, saveTiles=False, savePred=False):
        self.modelPath = modelPath
        self.modelName = os.path.basename(os.path.normpath(modelPath))
        self.overlap   = overlap
        self.window    = window
        self.skipDone  = skipDone
        self.saveTiles = saveTiles
        self.savePred  = savePred
        self.network   = None

    # PREDIZ O BLOCO (OU SÓ OS TILES QUE COBREM inline) E DEVOLVE A INLINE PREDITA; None QUANDO O MODELO NÃO SERVE PARA O BLOCO
    def predict(self, grid, tilesDir, outputDir, inline, full=True):
        if self.savePred and not full:
            raise ValueError('SAVE_PRED pede o bloco inteiro: FULL_PREDICT = True')

        region  = None if full else inline
        names   = grid.names(region)
        pending = self.pending(grid, outputDir, region)
        npyPath = f'{outputDir}/predicted.npy'

        if not pending:
            print(f'[{self.modelName}] {os.path.basename(outputDir)} já predito — pulando')

        if not pending and self.savePred and os.path.exists(npyPath):
            return np.array(np.load(npyPath, mmap_mode='r')[inline])

        if pending and not self.load():
            return None

        os.makedirs(f'{outputDir}/masks' if self.saveTiles else outputDir, exist_ok=True)
        waiting = set(pending)
        reuse   = [name for name in names if name not in waiting and (self.savePred or grid.covers(name, inline))]
        sliding = SlidingWindow(self.window, grid.size, self.overlap) if pending else None
        volume  = np.lib.format.open_memmap(f'{outputDir}/predicted.tmp.npy', mode='w+', dtype=np.float32, shape=grid.shape) if self.savePred else None
        section = np.zeros(grid.shape[1:], np.float32)

        for name, probs in chain(self.read(grid, outputDir, reuse), self.process(sliding, tilesDir, outputDir, pending)):
            if volume is not None:
                grid.place(volume, name, probs)

            if grid.covers(name, inline):
                grid.place(section, name, probs, inline)

        if volume is not None:
            volume.flush()
            del volume
            os.replace(f'{outputDir}/predicted.tmp.npy', npyPath)    # SÓ O ARQUIVO COMPLETO GANHA O NOME FINAL

        if pending:
            info = {'model': self.modelName, 'block': os.path.basename(outputDir), 'inline': region, 'tiles': len(names), 'tile_shape': list(sliding.tile), 'img_size': list(sliding.window), 'overlap': sliding.overlap, 'windows_per_tile': len(sliding.positions)}
            json.dump(info, open(f'{outputDir}/predict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        return section

    # PASSA OS TILES names PELA JANELA DESLIZANTE; COM saveTiles CADA MÁSCARA TAMBÉM VAI PARA masks/, NO SHAPE DA ENTRADA
    def process(self, sliding, tilesDir, outputDir, names):
        if not names:
            return

        tag    = os.path.basename(outputDir)
        loader = DataLoader(PredictDataset([f'{tilesDir}/{name}' for name in names], sliding.tile), batch_size=1, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
        print(f'[{self.modelName}] {tag}: {sliding} — {len(names)} tiles pendentes')

        if len(sliding.positions) == 1:
            print(f'[{self.modelName}] janela == tile: uma passada por tile — OVERLAP={self.overlap} e o peso Hann não têm efeito aqui')

        for name, tile in zip(names, tqdm(loader, desc=f'{self.modelName} - {tag}')):
            probs = sliding.process(self.network, tile[0])

            if self.saveTiles:
                probs.tofile(f'{outputDir}/masks/{name}')

            yield name, probs

    # TILES DO BLOCO (region=None) OU DA INLINE region QUE FALTAM PREDIZER; COM skipDone, UM predicted.npy PRONTO OU A MÁSCARA INTEIRA EM masks/ DISPENSA O TILE
    def pending(self, grid, outputDir, region=None):
        if not self.skipDone:
            return grid.names(region)

        if self.savePred and os.path.exists(f'{outputDir}/predicted.npy'):
            return []

        return grid.missing(f'{outputDir}/masks', region)

    # CARREGA OS PESOS UMA VEZ, COM A JANELA EM QUE A REDE FOI TREINADA; False QUANDO O MODELO NÃO SERVE PARA O BLOCO
    def load(self):
        if self.network is not None:
            return True

        if not os.path.exists(f'{self.modelPath}/model.pth'):
            print(f'[{self.modelName}] sem model.pth — pulando')
            return False

        info    = json.load(open(f'{self.modelPath}/info.json', encoding='utf-8'))
        options = info.get('model', {})
        scaling = info.get('processing', {}).get('scaling', 'normalize')

        if scaling not in ('normalize', 'percentile'):    # O BLOCO ESTÁ EM [0, 1]
            print(f'[{self.modelName}] treinado com scaling={scaling!r}, fora de [0, 1] — pulando')
            return False

        print(f'[{self.modelName}] opções do modelo:', json.dumps(options, indent=4))
        self.network = ModelNetwork(**options)
        self.network.model.load_state_dict(torch.load(f'{self.modelPath}/model.pth', map_location=self.network.device)['model'])
        self.network.model.eval()
        self.window = asTriple(self.window or options.get('img_size'), default=128)
        return True

    def read(self, grid, outputDir, names):
        for name in tqdm(names, desc=f'{self.modelName} - masks/', disable=not names):
            yield name, getDAT(f'{outputDir}/masks/{name}', grid.size)

    # SOLTA A REDE E A MEMÓRIA DA GPU ANTES DO PRÓXIMO MODELO
    def reset(self):
        self.network = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# FALHAS DE UMA SEÇÃO 2D COMO RETAS: VOTAÇÃO TIPO HOUGH, TRAÇADO COM LACUNAS, PCA E MATCHING PURSUIT
class FaultStickExtractor:
    def __init__(self, tolerance=0.80, minLen=140, maxGap=59):
        t    = float(np.clip(tolerance, 0.0, 1.0))
        lerp = lambda a, b: a + (b - a) * t

        self.tolerance = t
        self.minLen    = float(minLen)
        self.maxGap    = int(maxGap)
        self.dipMin    = 43.0
        self.dipMax    = 76.0

        self.evSigma  = 1.0
        self.evBudget = 0.020    # FRAÇÃO DA SEÇÃO QUE SE ESPERA MARCADA; ACIMA DISSO O PISO SOBE
        self.evLo     = 0.20
        self.evHi     = 0.50

        self.scanMin   = 34.0    # VARREDURA UM POUCO MAIS LARGA QUE O MERGULHO ACEITO, PARA O PCA TER DE ONDE GIRAR
        self.angleStep = 3.0
        self.seedLen   = 140
        self.seedWidth = 5
        self.seedThr   = lerp(0.198, 0.048)
        self.maxSeeds  = 1200
        self.nmsOffset = 9
        self.nmsAlong  = 51
        self.nmsAngle  = 9.9

        self.corrWidth    = 5
        self.supThr       = lerp(0.381, 0.211)
        self.traceMax     = 560
        self.stretchGap   = 86
        self.stretchFill  = 0.385
        self.stretchWidth = 8
        self.consumeWidth = 4

        self.minFill     = lerp(0.615, 0.415)
        self.energyMin   = 0.167
        self.energyFrac  = 0.213
        self.energySigma = 25
        self.layerMin    = lerp(35.8, 25.8)
        self.layerSigma  = 12.0

        self.mergeAngle  = 8.65
        self.mergeOffset = 6.0
        self.mergeGap    = 157.0
        self.dupCover    = 0.722
        self.dupOffset   = 17.7

        self.layer = self.coh = self.energyMap = None

    # STICKS DE UM MAPA DE PROBABILIDADE; A SÍSMICA LIGA OS FILTROS DE ENERGIA E DE CORTE DAS CAMADAS
    def extract(self, prob, seismic=None):
        self.energyMap = self.energy(seismic) if seismic is not None else None
        self.layer, self.coh = self.layering(seismic) if seismic is not None else (None, None)

        q = self.evidence(prob)
        return self.build(q, self.votes(q))

    # DA EVIDÊNCIA AOS STICKS FINAIS: PERSEGUIR, FUNDIR, REMEDIR O PREENCHIMENTO E TIRAR OS REPETIDOS
    def build(self, q, votes):
        sticks = self.merge(self.pursue(q, q.copy(), self.seeds(votes)))

        for s in sticks:
            s['fill'] = self.fracAbove(q, s, self.supThr)

        sticks = [s for s in sticks if s['len'] >= self.minLen and s['fill'] >= self.minFill * 0.9]
        return sorted(self.dedup(sticks), key=lambda s: -s['len'])

    # MATCHING PURSUIT: ACEITA A MELHOR RETA, CONSOME O CORREDOR DELA, SEGUE PARA A PRÓXIMA
    def pursue(self, q, res, seeds):
        sticks = []

        for _, x, y, phi in seeds:
            s = self.trace(res, x, y, phi)

            if s is None:
                continue

            s = self.refine(q, s)
            s['fill'] = self.fracAbove(res, s, self.supThr)

            if not self.accept(s):
                continue

            s = self.refine(q, self.stretch(res, s))
            self.consume(res, s)
            sticks.append(s)

        return sticks

    # PROBABILIDADE -> EVIDÊNCIA EM [0,1]; O PISO SOBE PELO PERCENTIL PARA A PREDIÇÃO DENSA NÃO EXPLODIR O TRAÇADO
    def evidence(self, prob):
        p  = gauss(prob.astype(np.float32), self.evSigma)
        hi = max(self.evHi, float(np.percentile(p, 100 * (1 - self.evBudget))))
        lo = max(self.evLo, float(np.percentile(p, 100 * (1 - 2 * self.evBudget))))
        return np.clip((p - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)

    # AMPLITUDE LOCAL DA SÍSMICA: ~0 NA ÁGUA E NA ZONA MORTA, ~1 NOS REFLETORES
    def energy(self, seismic):
        sec = seismic.astype(np.float32)
        mu  = gauss(sec, self.energySigma)
        var = gauss(sec * sec, self.energySigma) - mu * mu
        E   = np.sqrt(np.clip(var, 0, None))
        return np.clip(E / (np.percentile(E, 95) + 1e-9), 0, 1).astype(np.float32)

    # TENSOR DE ESTRUTURA DA SÍSMICA -> (ÂNGULO DAS CAMADAS EM GRAUS, COERÊNCIA)
    def layering(self, seismic):
        sec = seismic.astype(np.float32)
        Ix  = cv2.Sobel(sec, cv2.CV_32F, 1, 0, ksize=3)
        Iy  = cv2.Sobel(sec, cv2.CV_32F, 0, 1, ksize=3)
        Jxx, Jyy, Jxy = gauss(Ix * Ix, self.layerSigma), gauss(Iy * Iy, self.layerSigma), gauss(Ix * Iy, self.layerSigma)

        angle = (np.degrees(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180.0
        coh   = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
        return angle.astype(np.float32), coh.astype(np.float32)

    # VOTAÇÃO TIPO HOUGH: EVIDÊNCIA MÉDIA NUMA JANELA RETA, PARA CADA ORIENTAÇÃO PLAUSÍVEL
    def votes(self, q):
        canvas = int(np.ceil(np.hypot(*q.shape)))
        ones   = np.ones_like(q)
        kernel = (self.seedWidth, self.seedLen)
        area   = kernel[0] * kernel[1]
        out    = []

        for phi in np.arange(self.scanMin, 180.0 - self.scanMin + 1e-6, self.angleStep):
            M = self.rotation(q.shape, phi - 90.0, canvas)
            R = cv2.warpAffine(q, M, (canvas, canvas), flags=cv2.INTER_LINEAR, borderValue=0)
            V = cv2.warpAffine(ones, M, (canvas, canvas), flags=cv2.INTER_NEAREST, borderValue=0)

            num = cv2.boxFilter(R, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
            den = cv2.boxFilter(V, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
            acc = np.where(den > 0.6 * area, num / np.maximum(den, 1e-6), 0)

            peak   = acc >= cv2.dilate(acc, np.ones((self.seedLen // 2, 9), np.uint8))
            ys, xs = np.nonzero((acc >= self.seedThr) & peak)

            if not len(ys):
                continue

            inv = cv2.invertAffineTransform(M)
            pts = inv[:, :2] @ np.stack([xs, ys]).astype(np.float32) + inv[:, 2:3]
            out += [(float(s), float(x), float(y), float(phi)) for (x, y), s in zip(pts.T, acc[ys, xs])]

        out.sort(key=lambda r: -r[0])
        return out

    # ROTAÇÃO QUE DEIXA UMA RETA DE ÂNGULO phi VERTICAL NUMA TELA QUADRADA
    @staticmethod
    def rotation(shape, angle, canvas):
        h, w = shape
        M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        M[0, 2] += (canvas - 1) / 2 - (w - 1) / 2
        M[1, 2] += (canvas - 1) / 2 - (h - 1) / 2
        return M

    # DESCARTA VOTOS REDUNDANTES: A DISTÂNCIA QUE IMPORTA É A PERPENDICULAR, PORQUE A FAMÍLIA DOMINÓ É PARALELA
    def seeds(self, votes):
        cell, kept, grid = self.nmsAlong, [], {}

        for seed in votes:
            _, x, y, _ = seed
            gx, gy = int(x // cell), int(y // cell)
            near   = [kept[k] for i in (gx - 1, gx, gx + 1) for j in (gy - 1, gy, gy + 1) for k in grid.get((i, j), ())]

            if any(self.redundant(seed, other) for other in near):
                continue

            grid.setdefault((gx, gy), []).append(len(kept))
            kept.append(seed)

            if len(kept) >= self.maxSeeds:
                break

        return kept

    def redundant(self, seed, other):
        _, x, y, phi    = seed
        _, x2, y2, phi2 = other
        da = abs(phi - phi2) % 180

        if min(da, 180 - da) > self.nmsAngle:
            return False

        d = np.array([np.cos(np.radians(phi2)), np.sin(np.radians(phi2))])
        v = np.array([x - x2, y - y2])
        return abs(d[0] * v[1] - d[1] * v[0]) <= self.nmsOffset and abs(d @ v) <= self.nmsAlong

    # ESTENDE A SEMENTE ENQUANTO HOUVER SUPORTE, TOLERANDO LACUNAS: É ISSO QUE COSTURA OS PEDAÇOS
    def trace(self, q, x, y, phi):
        d  = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
        c  = np.array([x, y], float)
        ts = np.arange(-self.traceMax, self.traceMax + 1.0)

        prof, inside = self.profile(q, c, d, ts)
        sup  = (prof >= self.supThr) & inside
        i0   = len(ts) // 2
        half = self.seedLen // 2

        if not sup[max(i0 - half, 0):i0 + half + 1].any():
            return None

        a, b = self.extent(sup, i0, self.maxGap)

        if b - a < self.minLen * 0.6:
            return None

        return self.stick(c + ts[a] * d, c + ts[b] * d, sup[a:b + 1].mean())

    # PROCURA A CONTINUAÇÃO ALÉM DAS PONTAS: DEPOIS DE VALIDADA A RETA VALE ATRAVESSAR UMA LACUNA MAIOR
    def stretch(self, q, s):
        c, d, length = self.unit(s)
        ts = np.arange(-self.traceMax, self.traceMax + 1.0)
        prof, inside = self.profile(q, c, d, ts, self.stretchWidth)
        sup = (prof >= self.supThr) & inside

        i0   = len(ts) // 2
        a0   = max(int(i0 - length / 2), 0)
        b0   = min(int(i0 + length / 2), len(ts) - 1)
        a, b = self.extent(sup, i0, int(self.stretchGap))

        if a < a0 and sup[a:a0].mean() < self.stretchFill:
            a = a0

        if b > b0 and sup[b0:b].mean() < self.stretchFill:
            b = b0

        return self.stick(c + ts[a] * d, c + ts[b] * d, sup[a:b + 1].mean())

    # REAJUSTA A RETA AO CENTRO DE MASSA DA EVIDÊNCIA DENTRO DO CORREDOR
    def refine(self, q, s, iters=2):
        for _ in range(iters):
            c, d, length = self.unit(s)

            if length < 4:
                return s

            n    = np.array([-d[1], d[0]])
            ts   = np.arange(-length / 2, length / 2 + 1.0)
            offs = np.arange(-self.corrWidth, self.corrWidth + 1)
            X    = c[0] + ts[:, None] * d[0] + offs[None, :] * n[0]
            Y    = c[1] + ts[:, None] * d[1] + offs[None, :] * n[1]

            xi, yi = X.round().astype(np.int32), Y.round().astype(np.int32)
            ok     = (yi >= 0) & (yi < q.shape[0]) & (xi >= 0) & (xi < q.shape[1])
            w      = np.zeros(X.shape, np.float32)
            w[ok]  = q[yi[ok], xi[ok]]

            if w.sum() < 1e-3:
                return s

            wf = w.ravel()
            P  = np.stack([X.ravel(), Y.ravel()])
            cm = (P * wf).sum(1) / wf.sum()
            P0 = P - cm[:, None]
            V  = np.linalg.eigh((P0 * wf) @ P0.T / wf.sum())[1]
            dn = V[:, -1] * (1 if V[1, -1] >= 0 else -1)

            if abs(float(dn @ d)) < 0.985:    # REJEITA GIROS BRUSCOS
                return s

            t = dn @ P0
            s = self.stick(cm + dn * t.min(), cm + dn * t.max(), s['fill'])

        return s

    # DESCARTA O QUE NÃO PARECE FALHA: FORA DO MERGULHO, CURTA, MAL PREENCHIDA, SEM REFLETORES OU PARALELA ÀS CAMADAS
    def accept(self, s):
        if not (self.dipMin <= s['dip'] <= self.dipMax) or s['len'] < self.minLen or s['fill'] < self.minFill:
            return False

        if self.energyMap is not None:
            s['energy'] = round(self.fracAbove(self.energyMap, s, self.energyMin), 3)

            if s['energy'] < self.energyFrac:
                return False

        if self.layer is not None:
            s['cut'] = round(self.cutAngle(s), 1)

            if s['cut'] < self.layerMin:
                return False

        return True

    # ZERA O CORREDOR DA RETA ACEITA, PARA A PRÓXIMA SEMENTE NÃO REDESENHAR A MESMA FALHA
    def consume(self, res, s):
        c, d, length = self.unit(s)
        n    = np.array([-d[1], d[0]])
        ts   = np.arange(-length / 2, length / 2 + 1.0)
        offs = np.arange(-self.consumeWidth, self.consumeWidth + 1)
        xs   = (c[0] + ts[:, None] * d[0] + offs[None, :] * n[0]).round().astype(np.int32)
        ys   = (c[1] + ts[:, None] * d[1] + offs[None, :] * n[1]).round().astype(np.int32)
        ok   = (ys >= 0) & (ys < res.shape[0]) & (xs >= 0) & (xs < res.shape[1])
        res[ys[ok], xs[ok]] = 0.0

    # FUNDE OS GRUPOS DE RETAS COLINEARES NUMA SÓ, PELO PCA DAS PONTAS
    def merge(self, sticks):
        parent = list(range(len(sticks)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(sticks)):
            for j in range(i + 1, len(sticks)):
                if self.near(sticks[i], sticks[j]):
                    parent[find(i)] = find(j)

        groups = {}

        for i in range(len(sticks)):
            groups.setdefault(find(i), []).append(i)

        out = []

        for index in groups.values():
            if len(index) == 1:
                out.append(sticks[index[0]])
                continue

            pts = np.array([sticks[i]['p0'] for i in index] + [sticks[i]['p1'] for i in index], float)
            c   = pts.mean(0)
            P0  = (pts - c).T
            V   = np.linalg.eigh(P0 @ P0.T / len(pts))[1]
            d   = V[:, -1] * (1 if V[1, -1] >= 0 else -1)
            t   = d @ P0
            out.append(self.stick(c + d * t.min(), c + d * t.max(), np.mean([sticks[i]['fill'] for i in index])))

        return out

    # DUAS RETAS SÃO O MESMO TRAÇO: MESMA DIREÇÃO, PONTAS PRÓXIMAS E SEM DEGRAU LATERAL
    def near(self, a, b):
        da, db = self.unit(a)[1], self.unit(b)[1]

        if np.degrees(np.arccos(np.clip(abs(float(da @ db)), 0, 1))) > self.mergeAngle:
            return False

        ea = [np.array(a['p0']), np.array(a['p1'])]
        eb = [np.array(b['p0']), np.array(b['p1'])]

        if min(np.linalg.norm(x - y) for x in ea for y in eb) > self.mergeGap:
            return False

        return max(abs(da[0] * (p - ea[0])[1] - da[1] * (p - ea[0])[0]) for p in eb) <= self.mergeOffset

    # TIRA A RETA QUE JÁ CORRE POR CIMA DE OUTRA MAIOR: É O QUE POLUI A FIGURA
    def dedup(self, sticks):
        out = []

        for s in sorted(sticks, key=lambda s: -s['len']):
            if not any(self.overlap(s, o) >= self.dupCover for o in out):
                out.append(s)

        return out

    # FRAÇÃO DE a QUE CORRE COLADA EM b
    def overlap(self, a, b):
        p0, p1 = np.array(a['p0'], float), np.array(a['p1'], float)
        P      = p0 + (p1 - p0) * np.linspace(0, 1, max(int(a['len'] / 4) + 1, 2))[:, None]
        q0, q1 = np.array(b['p0'], float), np.array(b['p1'], float)
        v      = q1 - q0
        u      = np.clip(((P - q0) @ v) / (v @ v + 1e-9), 0, 1)
        return float((np.linalg.norm(P - (q0 + u[:, None] * v), axis=1) <= self.dupOffset).mean())

    # VALOR DE UM MAPA AO LONGO DA RETA: PARA CADA t, O MÁXIMO NA LARGURA DO CORREDOR
    def profile(self, m, c, d, ts, width=None):
        n    = np.array([-d[1], d[0]])
        w    = self.corrWidth if width is None else width
        offs = np.arange(-w, w + 1)
        xs   = (c[0] + ts[:, None] * d[0] + offs[None, :] * n[0]).round().astype(np.int32)
        ys   = (c[1] + ts[:, None] * d[1] + offs[None, :] * n[1]).round().astype(np.int32)
        ok   = (ys >= 0) & (ys < m.shape[0]) & (xs >= 0) & (xs < m.shape[1])

        values     = np.zeros(xs.shape, np.float32)
        values[ok] = m[ys[ok], xs[ok]]
        return values.max(1), ok.any(1)

    # MAIOR INTERVALO EM TORNO DE i0 CUJAS LACUNAS SEM SUPORTE CABEM EM maxGap
    @staticmethod
    def extent(sup, i0, maxGap):
        n, ends = len(sup), []

        for step in (1, -1):
            i, last, gap = i0, i0, 0

            while 0 <= i + step < n:
                i += step

                if sup[i]:
                    last, gap = i, 0
                    continue

                gap += 1

                if gap > maxGap:
                    break

            ends.append(last)

        return ends[1], ends[0]

    # FRAÇÃO DO COMPRIMENTO DA RETA EM QUE O MAPA PASSA DO LIMIAR
    def fracAbove(self, m, s, thr):
        c, d, length = self.unit(s)
        prof, _ = self.profile(m, c, d, np.arange(-length / 2, length / 2 + 1.0))
        return float((prof >= thr).mean())

    # ÂNGULO ENTRE A RETA E AS CAMADAS, MEDIANA PONDERADA PELA COERÊNCIA
    def cutAngle(self, s):
        c, d, length = self.unit(s)
        ts = np.arange(-length / 2, length / 2 + 1.0)
        xi = np.clip((c[0] + ts * d[0]).round().astype(int), 0, self.layer.shape[1] - 1)
        yi = np.clip((c[1] + ts * d[1]).round().astype(int), 0, self.layer.shape[0] - 1)

        diff = np.abs(self.layer[yi, xi] - np.degrees(np.arctan2(d[1], d[0])) % 180) % 180
        diff = np.minimum(diff, 180 - diff)
        w    = self.coh[yi, xi]

        if w.sum() < 1e-6:
            return 90.0

        order = np.argsort(diff)
        return float(diff[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.5)])

    # CENTRO, DIREÇÃO UNITÁRIA E COMPRIMENTO DE UM STICK
    @staticmethod
    def unit(s):
        p0, p1 = np.array(s['p0'], float), np.array(s['p1'], float)
        d      = p1 - p0
        length = float(np.hypot(*d)) + 1e-9
        return (p0 + p1) / 2, d / length, length

    # UM STICK: AS DUAS PONTAS, O MERGULHO ABSOLUTO, O COMPRIMENTO E O PREENCHIMENTO
    @staticmethod
    def stick(p0, p1, fill=0.0):
        d = np.asarray(p1, float) - np.asarray(p0, float)
        return {
            'p0': (float(p0[0]), float(p0[1])),
            'p1': (float(p1[0]), float(p1[1])),
            'dip': float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-12))),
            'len': float(np.hypot(*d)),
            'fill': round(float(fill), 3)
        }

    # ANOTAÇÃO DO ESPECIALISTA -> STICKS: ESQUELETIZA, CORTA NAS JUNÇÕES E ENDIREITA OS PEDAÇOS
    @classmethod
    def fromRaster(cls, mask, maxDev=7.0, minLen=30):
        sk  = skeletonize(np.asarray(mask) > 0.5)
        nb  = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - sk
        lbl = ndi.label(sk & (nb <= 2), structure=np.ones((3, 3)))[0]
        out = []

        for i, sl in enumerate(ndi.find_objects(lbl), 1):
            ys, xs = np.nonzero(lbl[sl] == i)

            if len(ys) >= minLen:
                out += cls.splitCurve(ys + sl[0].start, xs + sl[1].start, maxDev, minLen)

        tool = cls()
        tool.mergeGap, tool.mergeAngle, tool.mergeOffset = 70.0, 10.0, 10.0
        return tool.merge(out)

    # QUEBRA UM TRAÇO TORTO NO PONTO DE MAIOR DESVIO ATÉ CADA PEDAÇO CABER EM maxDev
    @classmethod
    def splitCurve(cls, ys, xs, maxDev, minLen, depth=0):
        P   = np.stack([xs, ys]).astype(float)
        c   = P.mean(1)
        P0  = P - c[:, None]
        V   = np.linalg.eigh(P0 @ P0.T / P.shape[1])[1]
        d   = V[:, -1] * (1 if V[1, -1] >= 0 else -1)
        t   = d @ P0
        dev = np.abs(-d[1] * P0[0] + d[0] * P0[1])

        if dev.max() > maxDev and depth < 6 and len(ys) >= 2 * minLen:
            m = t <= t[int(np.argmax(dev))]

            if m.sum() >= minLen and (~m).sum() >= minLen:
                return cls.splitCurve(ys[m], xs[m], maxDev, minLen, depth + 1) + cls.splitCurve(ys[~m], xs[~m], maxDev, minLen, depth + 1)

        if t.max() - t.min() < minLen:
            return []

        return [cls.stick(c + d * t.min(), c + d * t.max(), 1.0)]


# INLINE DE UM BLOCO COM AS CAMADAS PREDITAS POR CIMA: EXTRAI OS STICKS E DESENHA OS PAINÉIS; A CAMADA expert É A ANOTAÇÃO DO ESPECIALISTA
class BlockSection:
    PANELS = {
        'seismic': [('sísmica', False, False, True)],
        'mask': [('máscara', True, False, True)],
        'sticks': [('sticks', False, True, True)],
        'both': [('sísmica', False, False, False), ('máscara', True, False, True)]
    }

    def __init__(self, seismic, extractor, threshold):
        self.seismic   = seismic
        self.extractor = extractor
        self.threshold = threshold
        self.shape     = seismic.shape
        self.layers    = {}
        self.lines     = {}

    def setSection(self, label, values, color, expert=False):
        self.layers[label] = {'values': np.asarray(values, np.float32), 'color': color, 'expert': expert}

    # STICKS DE UMA CAMADA SOBRE A SÍSMICA; A ANOTAÇÃO ENTRA PELO fromRaster, NA MESMA REPRESENTAÇÃO
    def sticks(self, label):
        if label not in self.lines:
            layer = self.layers[label]
            self.lines[label] = FaultStickExtractor.fromRaster(layer['values']) if layer['expert'] else self.extractor.extract(layer['values'], seismic=self.seismic)

        return self.lines[label]

    # RESUMO DA INLINE: QUANTO DELA PASSA DO LIMIAR E O QUE O EXTRATOR TIROU DALI
    def metrics(self, label):
        lengths = np.array([s['len'] for s in self.sticks(label)])
        return {
            'fracao_mascara': round(float((self.layers[label]['values'] >= self.threshold).mean()), 4),
            'sticks': len(lengths),
            'len_p50': round(float(np.median(lengths)), 1) if len(lengths) else 0.0,
            'len_total': int(lengths.sum()),
            'dip_p50': round(float(np.median([s['dip'] for s in self.sticks(label)])), 1) if len(lengths) else 0.0
        }

    # UMA FIGURA DO PAINEL PEDIDO; save GRAVA EM DISCO EM VEZ DE MOSTRAR
    def show(self, mode='seismic', title=None, save=None):
        panels    = self.PANELS[mode]
        height    = 15 * self.shape[0] / self.shape[1]
        fig, axes = plt.subplots(1, len(panels), figsize=(15 * len(panels), height), squeeze=False)

        for ax, (name, dark, asSticks, draw) in zip(axes[0], panels):
            detail = f'tolerance {self.extractor.tolerance}' if asSticks else f'threshold {self.threshold}'
            header = name if not draw else f'{name} ({detail})'
            ax.imshow(np.zeros(self.shape) if dark else self.seismic, cmap='gray', aspect='auto', zorder=0)
            ax.set_title(f'{title} — {header}' if title else header)
            ax.set_xlabel('crossline'); ax.set_ylabel('z')

            if not draw:
                continue

            self.plotLayers(ax, asSticks)
            ax.set_xlim(-0.5, self.shape[1] - 0.5)
            ax.set_ylim(self.shape[0] - 0.5, -0.5)

            if self.layers:
                ax.legend(handles=[Patch(facecolor=to_rgb(layer['color']), label=label) for label, layer in self.layers.items()], loc='upper right', framealpha=0.9)

        plt.tight_layout()

        if save:
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save, dpi=200, bbox_inches='tight')
            plt.close(fig)
            return print(f'figura salva em {save}')

        plt.show()

    # AS CAMADAS SOBRE O PAINEL: MÁSCARA OU STICKS PARA AS REDES, PONTOS PARA A ANOTAÇÃO
    def plotLayers(self, ax, asSticks):
        for z, (label, layer) in enumerate(self.layers.items(), 2):
            color = to_rgb(layer['color'])

            if asSticks:
                for s in self.sticks(label):
                    ax.plot([s['p0'][0], s['p1'][0]], [s['p0'][1], s['p1'][1]], color=color, lw=2.2 if layer['expert'] else 1.6, zorder=z)
                continue

            if layer['expert']:
                gy, gx = np.where(layer['values'] > 0.5)
                ax.scatter(gx, gy, s=0.05, c=[color], alpha=0.8, zorder=z)
                continue

            rgba = np.zeros((*self.shape, 4), np.float32)
            rgba[..., :3] = color
            rgba[..., 3]  = layer['values'] >= self.threshold
            ax.imshow(rgba, aspect='auto', zorder=z)


# VISUALIZADOR 3D EM HTML (FATIAS INLINE, CROSSLINE E TEMPO COM A PREDIÇÃO POR CIMA, COMO A FIGURA 3D DO WU), EM RESOLUÇÃO CHEIA: NADA É REDUZIDO.
# A PROBABILIDADE VAI EM uint8 (PASSO DE 1/255) COM O PISO floor (ABAIXO DELE O VOXEL NUNCA APARECE: É O MENOR LIMIAR DO HTML), EM gzip + base64,
# EMBUTIDA NO HTML OU, COM external, NUM .js AO LADO (CARREGADO SÓ QUANDO A REDE É ESCOLHIDA). A SÍSMICA DO BLOCO É GRAVADA UMA VEZ EM cache
# (PARTES .js DE ATÉ PART VOXELS, CARREGADAS PELO HTML POR CAMINHO RELATIVO) E SERVE A TODOS OS MODELOS; A FRAÇÃO ACIMA DO LIMIAR SAI DO HISTOGRAMA
class Viewer3D:
    TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer3d.html')
    PART     = 32_000_000

    def __init__(self, seismic, cache, origin=(0, 0, 0), floor=0.05):
        self.shape  = tuple(int(n) for n in seismic.shape)
        self.origin = [int(o) for o in origin]
        self.floor  = floor
        self.cache  = cache
        self.parts  = self.writeSeismic(seismic)
        self.layers = []
        self.expert = None

    # PARTES DA SÍSMICA EM cache, AO LONGO DO EIXO DAS INLINES; REGRAVA SE O shape OU A AMOSTRA DE CONFERÊNCIA MUDOU
    def writeSeismic(self, seismic):
        step  = max(1, self.PART // (self.shape[1] * self.shape[2]))
        names = [f'seismic_{k:03d}.js' for k in range(int(np.ceil(self.shape[0] / step)))]
        check = {'shape': list(self.shape), 'amostra': float(np.round(np.asarray(seismic[::17, ::13, ::11], np.float64).sum(), 3)), 'partes': names}
        meta  = f'{self.cache}/seismic.json'

        if os.path.exists(meta) and json.load(open(meta)) == check and all(os.path.exists(f'{self.cache}/{n}') for n in names):
            return names

        os.makedirs(self.cache, exist_ok=True)
        for k, name in enumerate(names):
            block = np.round(np.clip(np.asarray(seismic[k * step:(k + 1) * step], np.float32), 0, 1) * 255).astype(np.uint8)
            open(f'{self.cache}/{name}', 'w').write(f"(window.V3D_SEISMIC = window.V3D_SEISMIC || {{}})[{k}] = '{self.encode(block)}';\n")
        json.dump(check, open(meta, 'w'))
        return names

    def addLayer(self, label, probs, info=None, external=None):
        if tuple(probs.shape) != self.shape:
            raise ValueError(f'{label}: probabilidade {tuple(probs.shape)} fora do shape da sísmica {self.shape}')

        quantized = np.round(np.clip(np.asarray(probs, np.float32), 0, 1) * 255).astype(np.uint8)
        hist      = np.bincount(quantized.ravel(), minlength=256).tolist()
        quantized[quantized < round(self.floor * 255)] = 0
        layer     = {'label': label, 'info': info or {}, 'hist': hist, 'prob': None, 'src': None, 'key': f'layer{len(self.layers)}'}

        if external:
            os.makedirs(external, exist_ok=True)
            layer['src'] = f'{external}/{layer["key"]}.js'
            open(layer['src'], 'w').write(f"(window.V3D_LAYERS = window.V3D_LAYERS || {{}})['{layer['key']}'] = '{self.encode(quantized)}';\n")
        else:
            layer['prob'] = self.encode(quantized)
        self.layers.append(layer)

    # ANOTAÇÃO 2D (z, xline) NA FATIA inline, PARA APARECER NA PAREDE INLINE QUANDO ELA PASSA POR ALI
    def setExpert(self, inline, raster):
        self.expert = {'inline': int(inline), 'data': self.encode((np.asarray(raster) > 0).astype(np.uint8))}

    @staticmethod
    def encode(array):
        return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), 6)).decode('ascii')

    def save(self, path, title, subtitle='', meta=None, start=None, threshold=0.5):
        folder  = os.path.dirname(os.path.abspath(path))
        rel     = lambda target: os.path.relpath(os.path.abspath(target), folder).replace(os.sep, '/')
        layers  = [{**layer, 'src': rel(layer['src']) if layer['src'] else None} for layer in self.layers]
        payload = {'title': title, 'subtitle': subtitle, 'shape': list(self.shape), 'origin': self.origin, 'meta': meta or {}, 'threshold': threshold, 'floor': self.floor, 'start': list(start) if start else None, 'seismic': [rel(f'{self.cache}/{n}') for n in self.parts], 'expert': self.expert, 'layers': layers}
        html    = open(self.TEMPLATE, encoding='utf-8').read().replace('__TITLE__', title[:60]).replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/'))
        os.makedirs(folder, exist_ok=True)
        open(path, 'w', encoding='utf-8').write(html)
        return f'{path}: {os.path.getsize(path) / 1e6:.1f} MB, volume {list(self.shape)} em resolução cheia, sísmica em {len(self.parts)} parte(s) de {rel(self.cache)}'


# RESUMO DO MODELO PELO INFO.JSON, COM A JANELA DE TREINO NO RÓTULO
def modelSummary(base, name):
    path = f'{base}/{name}/info.json'
    info = json.load(open(path)) if os.path.exists(path) else {}

    size    = info.get('model', {}).get('img_size')
    size    = [] if size is None else ([size] * 3 if np.isscalar(size) else list(size))
    network = info.get('model', {}).get('network', '?')
    window  = '×'.join(str(int(s)) for s in size) if size else '?'
    iou     = info.get('trainer', {}).get('test_iou')

    return {
        'nome': name,
        'rede': network,
        'janela': window,
        'dataset': info.get('processing', {}).get('dataset', '?'),
        'iou_teste': round(float(iou), 4) if iou is not None else None,
        'rotulo': f'{name} · {network} {window}' + (f' (iou={iou:.3f})' if iou is not None else '')
    }


# FICHA DO MODELO PARA O PAINEL DO Viewer3D: REDE, DADO E RECEITA DO info.json E COMO O BLOCO FOI PREDITO (predict.json DA PASTA DO BLOCO)
def viewerInfo(base, name, blockDir):
    info    = json.load(open(f'{base}/{name}/info.json')) if os.path.exists(f'{base}/{name}/info.json') else {}
    run     = json.load(open(f'{blockDir}/predict.json')) if os.path.exists(f'{blockDir}/predict.json') else {}
    process = info.get('processing', {})
    model   = info.get('model', {})
    trainer = info.get('trainer', {})
    fields  = {
        'modelo': name,
        'rede': model.get('network'),
        'janela de treino': '×'.join(str(s) for s in model.get('img_size', [])) or None,
        'filtros': model.get('num_filters'),
        'parâmetros': info.get('parametros'),
        'dataset': process.get('dataset'),
        'escalonamento': process.get('scaling'),
        'loss': process.get('loss') or trainer.get('loss'),
        'lr': process.get('lr'),
        'lote': process.get('batch_size'),
        'agenda': process.get('scheduler'),
        'épocas': trainer.get('epochs'),
        'IoU de teste': round(float(trainer['test_iou']), 4) if trainer.get('test_iou') is not None else None,
        'referência': info.get('referencia'),
        'origem': info.get('origem'),
        'receita': info.get('receita') if isinstance(info.get('receita'), str) else None,
        'predição': run.get('metodo') or (f"janela {run['img_size']}, sobreposição {run.get('overlap')}, {run.get('windows_per_tile')} janelas por tile (BlockPredictor)" if run.get('img_size') else None)
    }
    return {key: value for key, value in fields.items() if value not in (None, '', [])}


# GRAVA O view3d.html DE UM MODELO NA PASTA DO BLOCO, A PARTIR DO predicted.npy; expert = (inline, raster) PÕE A ANOTAÇÃO NA FATIA INLINE
def saveViewer(base, name, blockDir, seismic, block, cache, threshold=0.5, origin=(0, 0, 0), expert=None, start=None):
    if not os.path.exists(f'{blockDir}/predicted.npy'):
        return f'[{name}] {blockDir}: sem predicted.npy (SAVE_PRED = True) — sem 3D'

    summary = modelSummary(base, name)
    viewer  = Viewer3D(seismic, cache, origin)
    viewer.addLayer(summary['rotulo'], np.load(f'{blockDir}/predicted.npy', mmap_mode='r'), viewerInfo(base, name, blockDir))

    if expert is not None:
        viewer.setExpert(*expert)

    title = f"{block} · {name} ({summary['rede']})"
    return viewer.save(f'{blockDir}/view3d.html', title, f"dataset {summary['dataset']} · volume {list(seismic.shape)} (inline, z, xline)", {'bloco': block}, start, threshold)


# JUNTA AS LINHAS NOVAS AO CSV ANTERIOR: UM PAR (modelo, key) RECALCULADO SUBSTITUI O ANTIGO, E REDE, JANELA E IOU SÃO RELIDOS DO info.json
def saveReport(rows, path, base, key, sort):
    report   = pd.DataFrame(rows)
    done     = set(zip(report.modelo, report[key]))
    previous = pd.read_csv(path, dtype={key: report[key].dtype}) if os.path.exists(path) else pd.DataFrame(columns=report.columns)
    previous = previous.loc[[pair not in done for pair in zip(previous.modelo, previous[key])]].copy()
    summary  = [modelSummary(base, name) for name in previous.modelo]

    for column in ('rede', 'janela', 'iou_teste'):
        previous[column] = [model[column] for model in summary]

    report = pd.concat([frame for frame in (report, previous) if len(frame)], ignore_index=True).sort_values([key, sort], ascending=[True, False]).reset_index(drop=True)
    report.to_csv(path, index=False)
    return report
