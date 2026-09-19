import numpy as np
import scipy.ndimage as ndimage
import os, json, copy, shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# VOLUME SÍSMICO SINTÉTICO COM A MÁSCARA DAS FALHAS, MONTADO NUM CUBO COM MARGEM E CORTADO NO FIM
class SyntheticGenerator:
    PAD  = 12       # PREENCHIMENTO QUE O map_coordinates FAZ NO MODO 'nearest'
    TAIL = 1e-30    # TAP DA WAVELET ABAIXO DESTA FRAÇÃO DO PICO NÃO MUDA A SOMA EM float64
    NICE = 10       # PRIORIDADE DOS PROCESSOS DO dataset(), PARA A MÁQUINA SEGUIR USÁVEL

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

        self.nx    = self.finalShape[0] + 2 * self.margin
        self.ny    = self.finalShape[1] + 2 * self.margin
        self.nz    = self.finalShape[2] + 2 * self.margin
        self.shape = (self.nx, self.ny, self.nz)

        if seed is not None:
            np.random.seed(seed)

    def get(self):
        model, mask = self.applyFaulting(self.applyShearing(self.applyFolding(self.genReflectivity())))
        image       = self.crop(self.applyNoise(self.applyWavelet(model)))
        image       = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image.astype(np.float32), self.crop(mask).astype(np.uint8)

    def set(self, options):
        for key, value in options.items():
            setattr(self, key, value)

    def genReflectivity(self):
        reflectivity = np.zeros(self.nz, dtype=np.float64)

        for _ in range(np.random.randint(*self.layerRange)):
            pos       = np.random.randint(0, self.nz)
            thickness = np.random.randint(*self.layerThickness)
            reflectivity[pos:pos + thickness] = np.random.uniform(-1, 1)

        return reflectivity

    def applyFolding(self, reflectivity):
        xx, yy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        a0     = np.random.uniform(*self.foldBaseShift)
        shift  = np.zeros((self.nx, self.ny), dtype=np.float64)

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
            shift += amp * np.exp(-(u ** 2 / (2 * sigmaX ** 2) + v ** 2 / (2 * sigmaY ** 2)))

        z       = np.arange(self.nz, dtype=np.float64)
        zTarget = z[None, None, :] + (a0 + shift[:, :, None] * (self.foldDamping * z / (self.nz - 1))[None, None, :])
        return ndimage.map_coordinates(reflectivity, [zTarget], order=3, mode='nearest')

    def applyShearing(self, reflectivity):
        e0    = np.random.uniform(*self.shearOffset)
        f     = np.random.uniform(*self.shearGradient)
        g     = np.random.uniform(*self.shearGradient)
        shift = e0 + f * np.arange(self.nx, dtype=np.float64)[:, None] + g * np.arange(self.ny, dtype=np.float64)[None, :]
        return self.interpolate(reflectivity, shift)

    # O map_coordinates CÚBICO EM (x, y, z + shift): COM x E y INTEIROS O SPLINE 3D SE REDUZ AO SPLINE 1D EM z
    def interpolate(self, volume, shift):
        coef    = ndimage.spline_filter1d(np.pad(volume, ((0, 0), (0, 0), (self.PAD, self.PAD)), mode='edge'), 3, axis=2, mode='nearest')
        floor   = np.floor(shift)
        t       = shift - floor
        start   = floor.astype(np.int64)[:, :, None] + np.arange(self.PAD - 1, self.PAD - 1 + volume.shape[2])[None, None, :]
        weights = ((1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6)
        output  = np.zeros(volume.shape)

        for k, weight in enumerate(weights):
            output += weight[:, :, None] * np.take_along_axis(coef, np.clip(start + k, 0, coef.shape[2] - 1), axis=2)

        return output

    def applyFaulting(self, reflectivity):
        model = reflectivity
        masks = np.zeros(self.shape, dtype=np.uint8)

        for _ in range(np.random.randint(*self.faultCount)):
            model, masks = self.applyFault(model, masks)

        return model, masks

    def applyFault(self, model, masks):
        p0        = np.random.uniform(0.15, 0.85, 3) * np.array(self.shape)
        dipRad    = np.deg2rad(np.random.uniform(*self.faultDipAngle))
        strikeRad = np.random.uniform(0, 2 * np.pi)
        normal    = np.array([np.sin(dipRad) * np.cos(strikeRad), np.sin(dipRad) * np.sin(strikeRad), np.cos(dipRad) * np.random.choice([-1.0, 1.0])])
        strike    = np.array([-normal[1], normal[0], 0.0])
        strike    = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(strike) < 1e-6 else strike / np.linalg.norm(strike)
        dip       = np.cross(normal, strike)
        dip      /= np.linalg.norm(dip)

        dx = (np.arange(self.nx) - p0[0])[:, None, None]
        dy = (np.arange(self.ny) - p0[1])[None, :, None]
        dz = (np.arange(self.nz) - p0[2])[None, None, :]

        distDip    = dip[0] * dx + dip[1] * dy + dip[2] * dz
        distPlane  = normal[0] * dx + normal[1] * dy + normal[2] * dz
        bend       = self.getBend(distDip)
        noise      = ndimage.gaussian_filter(np.random.normal(0, 1, self.shape), sigma=self.faultRoughSigma)
        noise     *= self.faultRoughness
        distPlane += noise
        distPlane -= bend
        throw      = self.getThrow(strike[0] * dx + strike[1] * dy + strike[2] * dz, distDip, np.random.uniform(*self.faultThrow))

        hanging = distPlane > 0
        moved   = throw[hanging]
        coords  = np.array([(points + moved * step).astype(np.float32) for points, step in zip(np.nonzero(hanging), dip)])
        model   = self.moveHangingWall(model, hanging, coords, order=1, mode='nearest')
        masks   = self.moveHangingWall(masks, hanging, coords, order=0, mode='constant', cval=0) if masks.any() else masks

        masks[(np.abs(distPlane) <= self.faultZoneWidth) & (np.abs(throw) > self.faultThreshold)] = 1
        return model, masks

    # FALHA LÍSTRICA: O PLANO SE DESLOCA COM O QUADRADO DA DISTÂNCIA AO LONGO DO MERGULHO
    def getBend(self, distDip):
        if np.random.random() >= self.faultCurveProb:
            return 0.0

        intensity = np.random.uniform(self.faultCurveMax * 0.5, self.faultCurveMax) * np.random.choice([-1.0, 1.0])
        return intensity * ((distDip / (max(self.shape) / 1.5)) ** 2)

    # REJEITO DE UMA FALHA: GAUSSIANO EM TORNO DO CENTRO OU RAMPA AO LONGO DO MERGULHO, METADE DAS VEZES CADA
    def getThrow(self, distStrike, distDip, maxDisp):
        if np.random.random() < 0.5:
            throw  = distStrike ** 2
            throw += distDip ** 2
            throw /= -(2.0 * np.random.uniform(*self.faultDecaySigma) ** 2)
            np.exp(throw, out=throw)
            throw *= maxDisp
            return throw

        throw  = distDip / np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2)
        throw *= np.random.choice([-1, 1])
        throw += 0.5
        np.clip(throw, 0, 1, out=throw)
        throw *= maxDisp
        return throw

    # SÓ O BLOCO ALTO É REAMOSTRADO: NO BAIXO A COORDENADA É INTEIRA E O map_coordinates DEVOLVERIA O PRÓPRIO VOXEL
    def moveHangingWall(self, volume, hanging, coords, **options):
        moved          = volume.copy()
        moved[hanging] = ndimage.map_coordinates(volume, coords, **options)
        return moved

    # O convolve1d DO RICKER SEM AS CAUDAS ABAIXO DE TAIL, NA MESMA ORIGEM: A SOMA NÃO MUDA E O KERNEL ENCURTA ATÉ 4X
    def applyWavelet(self, model):
        f       = np.random.uniform(*self.waveletFreq)
        t       = np.arange(-self.waveletDuration, self.waveletDuration, self.waveletDt)
        wavelet = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-((np.pi * f * t) ** 2))
        keep    = np.flatnonzero(np.abs(wavelet) > self.TAIL * np.abs(wavelet).max())
        size    = len(wavelet)
        start   = size - 1 - keep[-1]
        kernel  = wavelet[::-1][start:size - keep[0]]
        return ndimage.correlate1d(model, kernel, axis=2, origin=size // 2 - (1 - size % 2) - start - len(kernel) // 2)

    def applyNoise(self, image):
        scale  = np.random.uniform(*self.noiseLevel) * np.std(image)
        noise  = ndimage.gaussian_filter(np.random.normal(0.0, 1.0, image.shape), sigma=self.noiseSigma)
        noise *= scale / (np.std(noise) + 1e-8)
        image += noise
        return ndimage.gaussian_filter(image, sigma=(0.5, 0.5, 0))

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
    def dataset(self, options, n_jobs=None):
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

        with ProcessPoolExecutor(max_workers=n_jobs or max(1, os.cpu_count() // 2), initializer=os.nice, initargs=(self.NICE,)) as executor:
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
                list(tqdm(executor.map(generator.saveTile, tasks), total=n, desc=name))
                offset += n

    # UM TILE DO dataset(): SEMENTE PRÓPRIA, GANHO DA REGIÃO E EIXOS (x, z, y) DOS DATASETS DO PROJETO
    def saveTile(self, task):
        index, seed, jitter, images, masks = task
        np.random.seed(seed)
        image, mask = self.get()
        np.save(os.path.join(images, f'img_{index:04d}.npy'), np.transpose(self.applyGain(image, jitter), (0, 2, 1)))
        np.save(os.path.join(masks, f'img_{index:04d}.npy'), np.transpose(mask, (0, 2, 1)))
