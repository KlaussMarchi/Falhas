import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import hilbert
from scipy.special import ndtri
import os, json, copy, shutil
import torch
import torch.nn.functional as F
from functools import lru_cache
from tqdm import tqdm


# VOLUME SÍSMICO SINTÉTICO COM A MÁSCARA DAS FALHAS, MONTADO NUM CUBO COM MARGEM E CORTADO NO FIM; O CUBO É CALCULADO NA PLACA, EM float64
class SyntheticGenerator:
    PAD   = 12      # PREENCHIMENTO QUE O map_coordinates FAZ NO MODO 'nearest'
    WATER = -2.0    # REFLETIVIDADE DA ÁGUA: O CONTRASTE COM A ROCHA É O REFLETOR FORTE DO FUNDO DO MAR

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

        self.faultThrowPeriod = (0.0, 0.0)    # REJEITO MÁXIMO EM PERÍODOS DA WAVELET DO TILE NO LUGAR DO faultThrow; (0, 0) DESLIGA
        self.faultVisible     = 0.0           # SÓ ROTULA ONDE O REJEITO PASSA DESTA FRAÇÃO DO PERÍODO; 0 DESLIGA
        self.faultDrag        = (0.0, 0.0)    # FRAÇÃO DO REJEITO QUE VIRA FLEXURA (ARRASTO) NOS DOIS BLOCOS; (0, 0) NÃO SORTEIA
        self.faultDragWidth   = (4, 16)       # LARGURA DA FLEXURA EM VOXELS
        self.faultLabelMin    = 0.0           # REJEITO MÁXIMO (VOXELS) ABAIXO DO QUAL A FALHA DESLOCA SEM RÓTULO: SÓ AS PRINCIPAIS, COMO O INTÉRPRETE
        self.faultLabelPeriod = 0.0           # O MESMO EM PERÍODOS DA WAVELET DO TILE (O LIMIAR ACOMPANHA A ESCALA); 0 DESLIGA
        self.faultLabelEnergy = 0.0           # SÓ ROTULA ONDE A ENERGIA LOCAL DO REFLETOR SEM RUÍDO PASSA DESTA FRAÇÃO DA MEDIANA DO TILE; 0 DESLIGA

        self.waveletFreq     = (72, 99)       # FREQUÊNCIA DO RICKER
        self.waveletDuration = 0.1            # MEIA-DURAÇÃO DO KERNEL EM SEGUNDOS
        self.waveletDt       = 0.0012         # AMOSTRAGEM DO KERNEL
        self.waveletLog      = False          # True SORTEIA A FREQUÊNCIA EM ESCALA LOG: CADA OITAVA DE PERÍODO COM O MESMO PESO
        self.waveletPhase    = (0.0, 0.0)     # ROTAÇÃO DE FASE DO RICKER EM GRAUS; (0, 0) NÃO SORTEIA

        self.noiseLevel = (0.015, 0.618)      # RUÍDO EM FRAÇÃO DO DESVIO DO SINAL
        self.noiseSigma = (1.0, 1.0, 0.5)     # GRÃO DO RUÍDO EM (x, y, z)

        self.gainField = 0.0                  # DESVIO DO LOG-GANHO SUAVE DA REFLETIVIDADE: O CONTRASTE MUDA DENTRO DO TILE; 0 NÃO SORTEIA
        self.gainSigma = (40, 40, 30)         # CORRELAÇÃO DO CAMPO DE GANHO EM (x, y, z)

        self.bodyCount   = (0, 1)             # CORPOS SEM RÓTULO (MONTE, CANAL, LENTE) POR TILE, TETO EXCLUSIVO; (0, 1) NÃO SORTEIA
        self.bodyHeight  = (15, 120)          # ESPESSURA MÁXIMA DO CORPO EM VOXELS
        self.bodyWidth   = (8, 50)            # LARGURA (σ) DO CORPO: ESPESSURA/LARGURA DÁ O MERGULHO DO FLANCO
        self.bodyStretch = (1, 4)             # ALONGAMENTO EM PLANTA: 1 É UM MONTE REDONDO, 4 JÁ É UM CANAL
        self.bodyLevel   = (0.0, 0.3)         # RECHEIO CAÓTICO EM FRAÇÃO DO DESVIO DAS CAMADAS: 0 É TRANSPARENTE
        self.bodyLayers  = 0.0                # CHANCE DE O CORPO VIR ENTULHADO DE CAMADAS PLANAS NOVAS (CANAL COM ONLAP) NO LUGAR DO RECHEIO CAÓTICO

        self.lensSigma = (0.0, 0.0)           # COMPRIMENTO (σ LATERAL) DOS TRECHOS DE CADA REFLETOR: O BRILHO VARIA E O REFLETOR SOME EM PARTES; (0, 0) NÃO SORTEIA
        self.lensCut   = (0.0, 0.5)           # FRAÇÃO DA CAMADA EM QUE O REFLETOR SOME (ACUNHAMENTO, TERMINAÇÃO SEM REJEITO)
        self.lensEdge  = (0.3, 2.0)           # LARGURA DA BORDA DO TRECHO EM DESVIOS DO CAMPO: PEQUENA É TERMINAÇÃO SECA, GRANDE É BRILHO QUE VARIA E ACUNHA
        self.lensThick = 3.0                  # σ VERTICAL DO CAMPO: CADA CAMADA DE 1-4 VOXELS GANHA O SEU PRÓPRIO PADRÃO LATERAL

        self.seafloorProb   = 0.0             # CHANCE DE ÁGUA NO TOPO DO TILE: AMPLITUDE ZERO ACIMA DO FUNDO DO MAR, SEM RÓTULO
        self.seafloorDepth  = (0.0, 0.6)      # FUNDO DO MAR EM FRAÇÃO DA ALTURA DO TILE
        self.deadZoneProb   = 0.0             # CHANCE DE ZONA MORTA NA BASE DO TILE: REFLETIVIDADE FRACA, SEM RÓTULO
        self.deadZoneDepth  = (0.2, 1.0)      # TOPO DA ZONA MORTA EM FRAÇÃO DA ALTURA DO TILE
        self.deadZoneLevel  = (0.02, 0.15)    # REFLETIVIDADE DA ZONA MORTA EM FRAÇÃO DA VIVA
        self.deadZoneHatch  = (0.0, 0.0)      # HACHURA DE MIGRAÇÃO NA ZONA MORTA EM FRAÇÃO DO DESVIO DA REFLETIVIDADE; (0, 0) NÃO SORTEIA
        self.deadZoneLabel  = 0.5             # A ZONA MORTA SÓ TIRA O RÓTULO ONDE A REFLETIVIDADE CAI ABAIXO DESTA FRAÇÃO DA VIVA; 0 DEIXA A FALHA ATRAVESSAR, COMO O ESPECIALISTA
        self.hatchDip       = (30, 60)        # MERGULHO DA HACHURA EM GRAUS (VOXEL), NOS DOIS SENTIDOS
        self.hatchPeriod    = (8, 20)         # COMPRIMENTO DE ONDA DA HACHURA EM VOXELS
        self.surfaceDip     = 0.3             # INCLINAÇÃO MÁXIMA (TANGENTE) DO FUNDO DO MAR E DO TOPO DA ZONA MORTA
        self.deadTraceProb  = 0.0             # CHANCE DE TRAÇOS MORTOS DE BORDA, COM O DADO ESPELHADO EM VOLTA COMO NOS TILES DE BORDA DO MARLIM
        self.deadTraceWidth = (1, 8)          # TRAÇOS MORTOS DE CADA LADO DO EIXO DO ESPELHO, TETO EXCLUSIVO

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

    # A FREQUÊNCIA DA WAVELET SÓ É SORTEADA ANTES DAS FALHAS QUANDO O REJEITO OU O RÓTULO DEPENDEM DO PERÍODO; SENÃO FICA ONDE SEMPRE ESTEVE
    def get(self):
        f                 = self.getFrequency() if any(self.faultThrowPeriod) or self.faultVisible > 0 or self.faultLabelPeriod > 0 else None
        model, mask       = self.applyFaulting(self.applyBodies(self.applyGainField(self.applyShearing(self.applyFolding(self.applyLenses(self.tensor(self.genReflectivity())))))), f)
        model, mask, rock = self.applyColumn(model, mask)
        wave              = self.applyWavelet(model, f)
        image, mask       = self.applyDeadTraces(self.crop(self.applyNoise(wave) * rock), self.crop(self.applyLabelEnergy(wave, mask)))
        image = (image - image.mean()) / (image.std(correction=0) + 1e-8)
        return image.float().cpu().numpy(), mask.cpu().numpy()

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

    # REFLETORES LENTICULARES: A REFLETIVIDADE 1D VIRA UM CUBO MODULADO POR UM CAMPO SUAVE, LONGO NA LATERAL E CURTO EM z, ANTES DA
    # DOBRA (O PADRÃO ACOMPANHA A CAMADA): CADA REFLETOR VARIA DE BRILHO E SOME EM TRECHOS, TERMINAÇÕES ESPALHADAS SEM REJEITO E SEM RÓTULO
    def applyLenses(self, reflectivity):
        if not any(self.lensSigma):
            return reflectivity

        sigma = np.random.uniform(*self.lensSigma)
        cut   = np.random.uniform(*self.lensCut)
        edge  = np.random.uniform(*self.lensEdge)
        field = self.gaussian(self.getField(self.shape), (sigma, sigma, self.lensThick)).double()
        field = field / (field.std(correction=0) + 1e-8)
        gate  = torch.clamp((field - ndtri(cut)) / edge, 0, 1)
        return self.tensor(reflectivity) * gate

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

    # CONTRASTE QUE MUDA DENTRO DO TILE: LOG-GANHO SUAVE NA REFLETIVIDADE ANTES DO FALHAMENTO, ENTÃO O BRILHO ACOMPANHA A CAMADA
    def applyGainField(self, model):
        if self.gainField <= 0:
            return model

        field = self.gaussian(self.getField(self.shape), self.gainSigma).double()
        return self.output(self.tensor(model) * torch.exp(self.gainField * field / (field.std(correction=0) + 1e-8)), model)

    def applyBodies(self, model):
        if self.bodyCount[1] <= 1:
            return model

        volume = self.tensor(model)

        for _ in range(np.random.randint(*self.bodyCount)):
            volume = self.applyBody(volume)

        return self.output(volume, model)

    # UM CORPO ENTRE O TOPO E A BASE DE UMA GAUSSIANA ELÍPTICA (upper 1 = MONTE DE BASE PLANA, 0 = CANAL DE TOPO PLANO, NO MEIO
    # UMA LENTE): O RECHEIO É CAÓTICO E FRACO, ENTÃO AS CAMADAS TERMINAM NO LIMITE DELE SEM REJEITO NENHUM
    def applyBody(self, volume):
        x0, y0  = np.random.uniform(0.25, 0.75, 2) * np.array([self.nx, self.ny])
        zc      = np.random.uniform(0.3, 0.7) * self.nz
        height  = np.random.uniform(*self.bodyHeight)
        sigma   = np.random.uniform(*self.bodyWidth)
        stretch = np.random.uniform(*self.bodyStretch)
        theta   = np.random.uniform(0, np.pi)
        upper   = np.random.uniform(0, 1)
        level   = np.random.uniform(*self.bodyLevel)
        layered = self.bodyLayers > 0 and np.random.random() < self.bodyLayers

        xx, yy = torch.meshgrid(self.tensor(np.arange(self.nx) - x0), self.tensor(np.arange(self.ny) - y0), indexing='ij')
        u      = np.cos(theta) * xx + np.sin(theta) * yy
        v      = -np.sin(theta) * xx + np.cos(theta) * yy
        relief = height * torch.exp(-(u ** 2 / (2 * sigma ** 2) + v ** 2 / (2 * (sigma * stretch) ** 2)))
        z      = self.tensor(np.arange(self.nz))
        inside = torch.clamp(torch.minimum(z - (zc - upper * relief)[:, :, None], (zc + (1 - upper) * relief)[:, :, None] - z), 0, 1)
        return volume * (1 - inside) + self.getBodyFill(volume, level, layered) * inside

    # RECHEIO DO CORPO: CAÓTICO E FRACO, OU CAMADAS PLANAS NOVAS (CANAL ENTULHADO) QUE TERMINAM NA MARGEM SEM CONTINUAR DO OUTRO LADO
    def getBodyFill(self, volume, level, layered):
        if layered:
            layers = self.tensor(self.genReflectivity())
            return layers * (np.random.uniform(0.5, 1.0) * volume.std(correction=0) / (layers.std(correction=0) + 1e-8))

        fill = self.gaussian(self.getField(self.shape), (2.0, 2.0, 1.0)).double()
        return fill * (level * volume.std(correction=0) / (fill.std(correction=0) + 1e-8))

    def applyFaulting(self, reflectivity, frequency=None):
        model  = self.tensor(reflectivity)
        masks  = torch.zeros(self.shape, dtype=torch.uint8, device=self.device)
        period = None if frequency is None else 1 / (frequency * self.waveletDt)

        for _ in range(np.random.randint(*self.faultCount)):
            model, masks = self.applyFault(model, masks, period)

        return self.output(model, reflectivity), self.output(masks, reflectivity)

    def applyFault(self, model, masks, period=None):
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
        limit     = np.random.uniform(*self.faultThrowPeriod) * period if any(self.faultThrowPeriod) else np.random.uniform(*self.faultThrow)
        throw     = self.getThrow(strike[0] * dx + strike[1] * dy + strike[2] * dz, distDip, limit)
        visible   = max(self.faultThreshold, self.faultVisible * period) if self.faultVisible > 0 else self.faultThreshold
        visible   = np.inf if limit < self.faultLabelMin or (self.faultLabelPeriod > 0 and limit < self.faultLabelPeriod * period) else visible

        if any(self.faultDrag):
            return self.applyDrag(model, masks, grid, dip, distPlane, throw, visible)

        hanging = distPlane > 0
        coords  = [(points + throw * step).float() for points, step in zip(grid, dip)]
        model   = torch.where(hanging, self.resample(model, coords), model)
        masks   = torch.where(hanging, self.pick(masks, coords), masks) if masks.any() else masks
        return model, masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > visible))

    # ARRASTO: A FRAÇÃO c DO REJEITO VIRA FLEXURA DE LARGURA w NOS DOIS BLOCOS E O RESTO É O SALTO NO PLANO; O RÓTULO VÊ O REJEITO TODO
    def applyDrag(self, model, masks, grid, dip, distPlane, throw, visible):
        c, w   = np.random.uniform(*self.faultDrag), np.random.uniform(*self.faultDragWidth)
        shift  = throw * torch.where(distPlane > 0, 1 - 0.5 * c * torch.exp(-distPlane.clamp(min=0) / w), 0.5 * c * torch.exp(distPlane.clamp(max=0) / w))
        coords = [(points + shift * step).float() for points, step in zip(grid, dip)]
        masks  = self.pick(masks, coords) if masks.any() else masks
        return self.resample(model, coords), masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > visible))

    # COLUNA DE UM BLOCO REAL: ÁGUA ACIMA DO FUNDO DO MAR (rock ZERA A IMAGEM DEPOIS DO RUÍDO) E ZONA MORTA FRACA ABAIXO DE UMA
    # SUPERFÍCIE; NENHUMA DAS DUAS TEM RÓTULO. DESLIGADA, NÃO SORTEIA NADA E rock É 1
    def applyColumn(self, model, mask):
        rock  = 1.0
        z     = self.tensor(np.arange(self.nz))
        scale = model.std(correction=0) if self.deadZoneProb > 0 else 1.0

        if self.seafloorProb > 0 and np.random.random() < self.seafloorProb:
            floor = self.getSurface(self.seafloorDepth)[:, :, None]
            model = torch.where(z < floor, self.WATER, model)
            mask  = mask * (z >= floor + 4)
            rock  = (z >= floor - 4).double()

        if self.deadZoneProb > 0 and np.random.random() < self.deadZoneProb:
            top   = self.getSurface(self.deadZoneDepth)[:, :, None]
            dead  = torch.sigmoid((z - top) / 6)
            fade  = 1 - (1 - np.random.uniform(*self.deadZoneLevel)) * dead
            hatch = self.getHatch() * (np.random.uniform(*self.deadZoneHatch) * scale) if any(self.deadZoneHatch) else 0.0
            model = model * fade + hatch * dead
            mask  = mask * (fade > self.deadZoneLabel)

        return model, mask, rock

    # HACHURA DE MIGRAÇÃO: RUÍDO FILTRADO NO ESPECTRO PERTO DAS NORMAIS DE DOIS PLANOS QUE MERGULHAM ±θ NUM AZIMUTE SORTEADO,
    # NUMA BANDA EM TORNO DO COMPRIMENTO DE ONDA SORTEADO; DESVIO 1
    def getHatch(self):
        theta  = np.radians(np.random.uniform(*self.hatchDip))
        phi    = np.random.uniform(0, np.pi)
        period = np.random.uniform(*self.hatchPeriod)
        k      = [self.tensor(f).reshape(shape) for f, shape in zip((np.fft.fftfreq(self.nx), np.fft.fftfreq(self.ny), np.fft.rfftfreq(self.nz)), ((-1, 1, 1), (1, -1, 1), (1, 1, -1)))]
        radius = torch.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2) + 1e-12
        cone   = sum(torch.exp(-(1 - ((sign * np.sin(theta) * (np.cos(phi) * k[0] + np.sin(phi) * k[1]) + np.cos(theta) * k[2]) / radius) ** 2) / (2 * 0.15 ** 2)) for sign in (1, -1))
        band   = torch.exp(-((radius - 1 / period) ** 2) / (2 * (0.3 / period) ** 2))
        hatch  = torch.fft.irfftn(torch.fft.rfftn(self.getField(self.shape).double()) * cone * band, s=self.shape)
        return hatch / (hatch.std(correction=0) + 1e-8)

    # BORDA DO LEVANTAMENTO COMO NOS TILES DE BORDA DO MARLIM: O LADO DE FORA DE UMA POSIÇÃO LATERAL É O ESPELHO DO LADO DE DENTRO
    # (O PATCH É COMPLETADO POR REFLEXÃO), COM UMA FAIXA DE AMPLITUDE ZERO NO EIXO E UMA RAMPA ATÉ O DADO VIVO; FAIXA E RAMPA SEM RÓTULO
    def applyDeadTraces(self, image, mask):
        if self.deadTraceProb <= 0 or np.random.random() >= self.deadTraceProb:
            return image, mask

        axis   = np.random.randint(2)
        center = np.random.randint(8, image.shape[axis] // 2)
        width  = np.random.randint(*self.deadTraceWidth)
        ramp   = np.random.randint(0, 4)
        pos    = np.arange(image.shape[axis])
        index  = torch.as_tensor(np.where(pos >= center, pos, 2 * center - pos), device=self.device)
        gain   = self.tensor(np.clip((np.abs(pos - center) - width + 1) / (ramp + 1), 0, 1)).reshape([-1 if a == axis else 1 for a in range(3)])
        return image.index_select(axis, index) * gain, mask.index_select(axis, index) * (gain >= 1)

    # SUPERFÍCIE z(x, y): PLANO INCLINADO NA PROFUNDIDADE SORTEADA (FRAÇÃO DA ALTURA DO TILE), COM ONDULAÇÃO SUAVE DE ATÉ 8 VOXELS
    def getSurface(self, depth):
        z0     = self.margin + np.random.uniform(*depth) * self.finalShape[2]
        gx, gy = np.random.uniform(-self.surfaceDip, self.surfaceDip, 2)
        wave   = self.gaussian(self.getField((self.nx, self.ny, 1)), 25).double()[:, :, 0]
        wave   = wave * (np.random.uniform(0, 8) / (wave.std(correction=0) + 1e-8))
        x, y   = self.tensor(np.arange(self.nx) - self.nx / 2), self.tensor(np.arange(self.ny) - self.ny / 2)
        return z0 + gx * x[:, None] + gy * y[None, :] + wave

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
    def getFrequency(self):
        return np.exp(np.random.uniform(*np.log(self.waveletFreq))) if self.waveletLog else np.random.uniform(*self.waveletFreq)

    def applyWavelet(self, model, frequency=None):
        f       = self.getFrequency() if frequency is None else frequency
        t       = np.arange(-self.waveletDuration, self.waveletDuration, self.waveletDt)
        wavelet = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-((np.pi * f * t) ** 2))
        wavelet = self.rotate(wavelet) if any(self.waveletPhase) else wavelet
        return self.output(self.tensor(model) @ self.tensor(ndimage.convolve1d(np.eye(self.nz), wavelet, axis=1)), model)

    # FASE DO RICKER GIRADA PELA TRANSFORMADA DE HILBERT: 90° É A QUADRATURA, 180° INVERTE A POLARIDADE
    def rotate(self, wavelet):
        phase = np.radians(np.random.uniform(*self.waveletPhase))
        return np.cos(phase) * wavelet + np.sin(phase) * np.imag(hilbert(wavelet))

    # RÓTULO SÓ ONDE HÁ REFLETOR PARA DESLOCAR: ONDE A ENERGIA LOCAL (RMS DE σ 4) DO TRAÇO SEM RUÍDO FICA ABAIXO DE faultLabelEnergy × A
    # MEDIANA DO TILE (LENTE APAGADA, GANHO FRACO), A FALHA NÃO TEM O QUE MOSTRAR E O RÓTULO SAI; NÃO SORTEIA NADA
    def applyLabelEnergy(self, wave, mask):
        if self.faultLabelEnergy <= 0 or not mask.any():
            return mask

        energy = torch.sqrt(self.gaussian(wave ** 2, 4.0))
        live   = energy[energy > 0.05 * energy.max()]
        return mask * (energy > self.faultLabelEnergy * live.median())

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
