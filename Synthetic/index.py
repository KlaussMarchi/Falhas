import numpy as np
import scipy.ndimage as ndimage
import os, json, copy, shutil
import torch
import torch.nn.functional as F
from functools import lru_cache
from tqdm import tqdm


# VOLUME SÍSMICO SINTÉTICO COM A MÁSCARA DAS FALHAS, MONTADO NUM CUBO COM MARGEM E CORTADO NO FIM; O CUBO É CALCULADO NA PLACA, EM float64
class SyntheticGenerator:
    PAD = 12    # PREENCHIMENTO QUE O map_coordinates FAZ NO MODO 'nearest'

    def __init__(self, shape=(128, 128, 128), seed=None):
        self.margin     = 64                  # BORDA QUE ABSORVE DOBRA E REJEITO ANTES DO CORTE
        self.finalShape = shape

        self.layerRange     = (26, 233)       # CAMADAS NA COLUNA
        self.layerThickness = (1, 4)          # ESPESSURA DA CAMADA EM VOXELS

        self.foldCount     = (15, 48)         # GAUSSIANAS DE DOBRA
        self.foldSigma     = (17, 57)         # LARGURA DA DOBRA
        self.foldAspect    = 1.0              # ALONGAMENTO DA DOBRA EM x: ABAIXO DE 1 A DOBRA SE ESTENDE MAIS NO EIXO x
        self.foldAmplitude = (-35, 5)         # ALTURA DA DOBRA
        self.foldDamping   = 1.35             # QUANTO A DOBRA CRESCE COM A PROFUNDIDADE
        self.foldBaseShift = (-0.75, 4.45)    # DESLOCAMENTO VERTICAL DO BLOCO INTEIRO
        self.shearOffset   = (-8.68, 3.3)     # DESLOCAMENTO VERTICAL CONSTANTE
        self.shearGradient = (-0.1, 0.02)     # MERGULHO REGIONAL DAS CAMADAS

        self.faultCount      = (5, 10)        # FALHAS POR TILE, TETO EXCLUSIVO
        self.faultThrow      = (15, 32)       # REJEITO MÁXIMO EM VOXELS
        self.faultDipAngle   = (55, 81)       # MERGULHO DO PLANO EM GRAUS
        self.faultRoughness  = 3.54           # AMPLITUDE DA RUGOSIDADE DO PLANO
        self.faultRoughSigma = 8.55           # COMPRIMENTO DE ONDA DA RUGOSIDADE
        self.faultDecaySigma = (49, 59)       # ALCANCE DO REJEITO GAUSSIANO
        self.faultZoneWidth  = 0.99           # MEIA-ESPESSURA DO RÓTULO
        self.faultThreshold  = 0.77           # REJEITO MÍNIMO PARA ROTULAR
        self.faultCurveProb  = 0.07           # CHANCE DE A FALHA SER LÍSTRICA
        self.faultCurveMax   = 8.44           # CURVATURA MÁXIMA DA LÍSTRICA

        self.waveletFreq     = (72, 99)       # FREQUÊNCIA DO RICKER
        self.waveletDuration = 0.1            # MEIA-DURAÇÃO DO KERNEL EM SEGUNDOS
        self.waveletDt       = 0.0012         # AMOSTRAGEM DO KERNEL

        self.noiseLevel = (0.015, 0.618)      # RUÍDO EM FRAÇÃO DO DESVIO DO SINAL
        self.noiseSigma = (1.0, 1.0, 0.5)     # GRÃO DO RUÍDO EM (x, y, z)

        self.gain       = None                # GANHO DO TILE Z-SCORADO; None GRAVA O TILE COMO SAI DO get()
        self.gainJitter = 0.0                 # DESVIO DO LOG-GANHO ENTRE TILES DO MESMO LOTE
        self.clip       = None                # SATURAÇÃO SIMÉTRICA DEPOIS DO GANHO

        self.fast   = False                   # True SORTEIA OS CAMPOS DE RUÍDO NA PLACA: MESMA DISTRIBUIÇÃO E MAIS RÁPIDO, MAS A SEMENTE DÁ OUTRO TILE
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.nx    = self.finalShape[0] + 2 * self.margin
        self.ny    = self.finalShape[1] + 2 * self.margin
        self.nz    = self.finalShape[2] + 2 * self.margin
        self.shape = (self.nx, self.ny, self.nz)

        if seed is not None:
            np.random.seed(seed)

    def get(self):
        model, mask = self.applyFaulting(self.applyShearing(self.applyFolding(self.tensor(self.genReflectivity()))))
        image = self.crop(self.applyNoise(self.applyWavelet(model)))
        image = (image - image.mean()) / (image.std(correction=0) + 1e-8)
        return image.float().cpu().numpy(), self.crop(mask).cpu().numpy()

    def set(self, options):
        for key, value in options.items():
            setattr(self, key, value)

    def tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float64, device=self.device)

    # CADA ETAPA DEVOLVE NO TIPO EM QUE RECEBEU: numpy PARA QUEM CHAMA A ETAPA SOLTA, E NO get() O CUBO FICA NA PLACA DE UMA ETAPA À OUTRA
    def output(self, volume, source):
        return volume if torch.is_tensor(source) else volume.cpu().numpy()

    # CAMPO NORMAL PADRÃO: O float64 DO np.random OU, NO MODO fast, UM float32 SORTEADO NA PLACA COM A SEMENTE TIRADA DO np.random
    def getField(self, shape):
        if not self.fast:
            return self.tensor(np.random.normal(0, 1, shape))

        return torch.randn(shape, generator=torch.Generator(self.device).manual_seed(int(np.random.randint(2 ** 31))), device=self.device)

    # A MATRIZ DO gaussian_filter1d DO scipy, COM A BORDA DELE: O FILTRO APLICADO ÀS COLUNAS DA IDENTIDADE
    @staticmethod
    @lru_cache(maxsize=32)
    def getGaussian(sigma, size, dtype, device):
        return torch.as_tensor(ndimage.gaussian_filter1d(np.eye(size), sigma, axis=0), dtype=dtype, device=device)

    # A MATRIZ DO PRÉ-FILTRO DO SPLINE CÚBICO NO MODO 'nearest'
    @staticmethod
    @lru_cache(maxsize=4)
    def getSpline(size, device):
        return torch.as_tensor(ndimage.spline_filter1d(np.eye(size), 3, axis=0, mode='nearest'), dtype=torch.float64, device=device)

    # UM FILTRO 1D AO LONGO DE UM EIXO DO VOLUME É O PRODUTO PELA MATRIZ DELE; AS TRÊS FORMAS DEVOLVEM O VOLUME CONTÍGUO
    def applyOperator(self, volume, operator, axis):
        if axis == 0:
            return (operator @ volume.reshape(volume.shape[0], -1)).reshape(-1, *volume.shape[1:])

        return operator @ volume if axis == 1 else volume @ operator.T

    # O gaussian_filter DO scipy: UM OPERADOR POR EIXO, NA ORDEM DOS EIXOS, PULANDO O EIXO DE SIGMA ZERO
    def gaussian(self, volume, sigma):
        for axis, value in enumerate(np.broadcast_to(sigma, volume.ndim)):
            if value > 1e-15:
                volume = self.applyOperator(volume, self.getGaussian(float(value), volume.shape[axis], volume.dtype, self.device), axis)

        return volume

    def genReflectivity(self):
        reflectivity = np.zeros(self.nz, dtype=np.float64)

        for _ in range(np.random.randint(*self.layerRange)):
            pos       = np.random.randint(0, self.nz)
            thickness = np.random.randint(*self.layerThickness)
            reflectivity[pos:pos + thickness] = np.random.uniform(-1, 1)

        return reflectivity

    def applyFolding(self, reflectivity):
        xx, yy = torch.meshgrid(self.tensor(np.arange(self.nx)), self.tensor(np.arange(self.ny)), indexing='ij')
        a0     = np.random.uniform(*self.foldBaseShift)
        shift  = torch.zeros_like(xx)

        for _ in range(np.random.randint(*self.foldCount)):
            x0     = np.random.uniform(-self.nx * 0.3, self.nx * 1.3)
            y0     = np.random.uniform(-self.ny * 0.3, self.ny * 1.3)
            sigmaX = np.random.uniform(*self.foldSigma)
            sigmaY = np.random.uniform(*self.foldSigma)
            theta  = np.random.uniform(0, np.pi)
            amp    = np.random.uniform(*self.foldAmplitude)
            dx, dy = (xx - x0) * self.foldAspect, yy - y0
            u      = np.cos(theta) * dx + np.sin(theta) * dy
            v      = -np.sin(theta) * dx + np.cos(theta) * dy
            shift += amp * torch.exp(-(u ** 2 / (2 * sigmaX ** 2) + v ** 2 / (2 * sigmaY ** 2)))

        z       = self.tensor(np.arange(self.nz))
        zTarget = z + (a0 + shift[:, :, None] * (self.foldDamping * z / (self.nz - 1)))
        floor   = zTarget.floor()
        return self.output(self.evaluate(self.getCoefficients(self.tensor(reflectivity)).expand(self.nx, self.ny, -1), floor.long(), zTarget - floor), reflectivity)

    def applyShearing(self, reflectivity):
        e0    = np.random.uniform(*self.shearOffset)
        f     = np.random.uniform(*self.shearGradient)
        g     = np.random.uniform(*self.shearGradient)
        shift = e0 + f * np.arange(self.nx, dtype=np.float64)[:, None] + g * np.arange(self.ny, dtype=np.float64)[None, :]
        return self.output(self.interpolate(self.tensor(reflectivity), shift), reflectivity)

    # O map_coordinates CÚBICO EM (x, y, z + shift): COM x E y INTEIROS O SPLINE 3D SE REDUZ AO SPLINE 1D EM z
    def interpolate(self, volume, shift):
        floor = np.floor(shift)
        start = self.tensor(floor).long()[:, :, None] + torch.arange(volume.shape[2], device=self.device)
        return self.evaluate(self.getCoefficients(volume), start, self.tensor(shift - floor)[:, :, None])

    # COEFICIENTES DO SPLINE CÚBICO EM z, COM O PREENCHIMENTO DE BORDA QUE O map_coordinates FAZ NO MODO 'nearest'
    def getCoefficients(self, volume):
        lead   = volume.shape[:-1]
        padded = torch.cat([volume[..., :1].expand(*lead, self.PAD), volume, volume[..., -1:].expand(*lead, self.PAD)], dim=-1)
        return padded @ self.getSpline(padded.shape[-1], self.device).T

    # OS 4 TAPS DO SPLINE EM z NA COORDENADA floor + t, COM O ÍNDICE DO ESTÊNCIL PRESO NA BORDA
    def evaluate(self, coef, floor, t):
        weights = ((1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6)
        return sum(weight * coef.gather(-1, (floor + self.PAD - 1 + k).clamp(0, coef.shape[-1] - 1)) for k, weight in enumerate(weights))

    def applyFaulting(self, reflectivity):
        model = self.tensor(reflectivity)
        masks = torch.zeros(self.shape, dtype=torch.uint8, device=self.device)

        for _ in range(np.random.randint(*self.faultCount)):
            model, masks = self.applyFault(model, masks)

        return self.output(model, reflectivity), self.output(masks, reflectivity)

    def applyFault(self, model, masks):
        p0        = np.random.uniform(0.15, 0.85, 3) * np.array(self.shape)
        dipRad    = np.deg2rad(np.random.uniform(*self.faultDipAngle))
        strikeRad = np.random.uniform(0, 2 * np.pi)
        normal    = np.array([np.sin(dipRad) * np.cos(strikeRad), np.sin(dipRad) * np.sin(strikeRad), np.cos(dipRad) * np.random.choice([-1.0, 1.0])])
        strike    = np.array([-normal[1], normal[0], 0.0])
        strike    = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(strike) < 1e-6 else strike / np.linalg.norm(strike)
        dip       = np.cross(normal, strike)
        dip      /= np.linalg.norm(dip)
        grid      = [self.tensor(np.arange(size)).reshape(shape) for size, shape in zip(self.shape, ((-1, 1, 1), (1, -1, 1), (1, 1, -1)))]

        normal, strike, dip = normal.tolist(), strike.tolist(), dip.tolist()
        dx, dy, dz          = [points - origin for points, origin in zip(grid, p0)]

        distDip   = dip[0] * dx + dip[1] * dy + dip[2] * dz
        distPlane = normal[0] * dx + normal[1] * dy + normal[2] * dz
        bend      = self.getBend(distDip)
        distPlane = distPlane + self.gaussian(self.getField(self.shape), self.faultRoughSigma).double() * self.faultRoughness - bend
        throw     = self.getThrow(strike[0] * dx + strike[1] * dy + strike[2] * dz, distDip, np.random.uniform(*self.faultThrow))

        hanging = distPlane > 0
        coords  = [(points + throw * step).float() for points, step in zip(grid, dip)]
        model   = torch.where(hanging, self.resample(model, coords), model)
        masks   = torch.where(hanging, self.pick(masks, coords), masks) if masks.any() else masks
        return model, masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > self.faultThreshold))

    # FALHA LÍSTRICA: O PLANO SE DESLOCA COM O QUADRADO DA DISTÂNCIA AO LONGO DO MERGULHO
    def getBend(self, distDip):
        if np.random.random() >= self.faultCurveProb:
            return 0.0

        intensity = np.random.uniform(self.faultCurveMax * 0.5, self.faultCurveMax) * np.random.choice([-1.0, 1.0])
        return intensity * ((distDip / (max(self.shape) / 1.5)) ** 2)

    # REJEITO DE UMA FALHA: GAUSSIANO EM TORNO DO CENTRO OU RAMPA AO LONGO DO MERGULHO, METADE DAS VEZES CADA
    def getThrow(self, distStrike, distDip, maxDisp):
        if np.random.random() < 0.5:
            return torch.exp((distStrike ** 2 + distDip ** 2) / -(2.0 * np.random.uniform(*self.faultDecaySigma) ** 2)) * maxDisp

        return (distDip / np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2) * np.random.choice([-1, 1]) + 0.5).clamp(0, 1) * maxDisp

    # O map_coordinates DE ORDEM 1 NO MODO 'nearest': TRILINEAR NA COORDENADA float32, PRESA NA BORDA
    def resample(self, volume, coords):
        grid = torch.stack([2 * coord.double() / (size - 1) - 1 for coord, size in zip(coords[::-1], volume.shape[::-1])], dim=-1)
        return F.grid_sample(volume[None, None], grid[None], mode='bilinear', padding_mode='border', align_corners=True)[0, 0]

    # O map_coordinates DE ORDEM 0 NO MODO 'constant': O VOXEL MAIS PRÓXIMO, E ZERO PARA A COORDENADA FORA DO VOLUME
    def pick(self, masks, coords):
        index  = [(coord.double() + 0.5).floor().long().clamp(0, size - 1) for coord, size in zip(coords, masks.shape)]
        inside = [(coord >= 0) & (coord <= size - 1) for coord, size in zip(coords, masks.shape)]
        return masks[index[0], index[1], index[2]] * (inside[0] & inside[1] & inside[2])

    # O convolve1d DO RICKER EM z COMO MATRIZ: O FILTRO DO scipy APLICADO ÀS LINHAS DA IDENTIDADE
    def applyWavelet(self, model):
        f       = np.random.uniform(*self.waveletFreq)
        t       = np.arange(-self.waveletDuration, self.waveletDuration, self.waveletDt)
        wavelet = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-((np.pi * f * t) ** 2))
        return self.output(self.tensor(model) @ self.tensor(ndimage.convolve1d(np.eye(self.nz), wavelet, axis=1)), model)

    def applyNoise(self, image):
        volume = self.tensor(image)
        scale  = np.random.uniform(*self.noiseLevel) * volume.std(correction=0)
        noise  = self.gaussian(self.getField(volume.shape), self.noiseSigma).double()
        return self.output(self.gaussian(volume + noise * (scale / (noise.std(correction=0) + 1e-8)), (0.5, 0.5, 0)), image)

    def crop(self, volume):
        return volume[self.margin:self.nx - self.margin, self.margin:self.ny - self.margin, self.margin:self.nz - self.margin]

    # CONTRASTE DE CADA TILE DE UM LOTE: LOG-NORMAL COM MEDIANA 1, ENTÃO O GANHO NÃO DEPENDE DO TAMANHO DO LOTE
    def getJitter(self, n, seed):
        draw = np.exp(self.gainJitter * np.clip(np.random.RandomState(seed).normal(0, 1, n), -2, 2))
        return draw / np.median(draw)

    def info(self):
        return {'shape': self.shape, **{key: value for key, value in vars(self).items() if key not in ('finalShape', 'nx', 'ny', 'nz', 'shape')}}

    def print(self):
        print(json.dumps(self.info(), indent=4))

    def applyGain(self, image, jitter=1.0):
        if self.gain is None:
            return image

        image = image * (self.gain * jitter)
        return (image if self.clip is None else np.clip(image, -self.clip, self.clip)).astype(np.float32)

    # options = {'directory', 'regions': {nome: {'n_images', 'output', 'params'}}, 'seed'}; SEM seed A BASE É SORTEADA
    def dataset(self, options):
        unknown = set(options) - {'directory', 'regions', 'seed'}

        if unknown:
            raise ValueError(f'chaves desconhecidas em options: {sorted(unknown)}')

        regions = options['regions']
        folders = {name: os.path.join(options.get('directory', 'output'), cfg.get('output', name)) for name, cfg in regions.items()}

        if len(set(folders.values())) < len(folders):
            raise ValueError(f'duas regiões gravam na mesma pasta: {folders}')

        for name, cfg in regions.items():
            extra   = set(cfg) - {'n_images', 'output', 'params'}
            unknown = set(cfg.get('params', {})) - set(vars(self))

            if extra or unknown:
                raise ValueError(f"'{name}': chaves desconhecidas {sorted(extra)}, parâmetros desconhecidos {sorted(unknown)}")

        seed   = options['seed'] if 'seed' in options else np.random.randint(0, 1000000)
        offset = 0

        for name, cfg in regions.items():
            n         = cfg.get('n_images', 200)
            generator = copy.deepcopy(self)
            generator.set(cfg.get('params', {}))
            images    = os.path.join(folders[name], 'images')
            masks     = os.path.join(folders[name], 'masks')

            shutil.rmtree(folders[name], ignore_errors=True)
            os.makedirs(images)
            os.makedirs(masks)

            tasks = [(offset + i, seed + offset + i, jitter, images, masks) for i, jitter in enumerate(generator.getJitter(n, seed + offset))]
            list(tqdm(map(generator.saveTile, tasks), total=n, desc=name))
            offset += n

    # UM TILE DO dataset(): SEMENTE PRÓPRIA, GANHO DA REGIÃO E EIXOS (x, z, y) DOS DATASETS DO PROJETO
    def saveTile(self, task):
        index, seed, jitter, images, masks = task
        np.random.seed(seed)
        image, mask = self.get()
        np.save(os.path.join(images, f'img_{index:04d}.npy'), np.transpose(self.applyGain(image, jitter), (0, 2, 1)))
        np.save(os.path.join(masks, f'img_{index:04d}.npy'), np.transpose(mask, (0, 2, 1)))
