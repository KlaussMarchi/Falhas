import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from scipy.signal import hilbert
from scipy.special import ndtri
import os, json, copy, shutil
import cv2
import torch
import torch.nn.functional as F
from functools import lru_cache
from tqdm import tqdm


# VOLUME SÍSMICO SINTÉTICO COM A MÁSCARA DAS FALHAS, MONTADO NUM CUBO COM MARGEM E CORTADO NO FIM; O CUBO É CALCULADO NA PLACA, EM float64
class SyntheticGenerator:
    PAD   = 12                                     # PREENCHIMENTO QUE O map_coordinates FAZ NO MODO 'nearest'
    WATER = -2.0                                   # REFLETIVIDADE DA ÁGUA: O CONTRASTE COM A ROCHA É O REFLETOR FORTE DO FUNDO DO MAR
    AXES  = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))   # FORMA DE CADA EIXO PARA O BROADCAST NO CUBO

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
        self.faultStrike     = (0.0, 2 * np.pi) # AZIMUTE DA NORMAL NO PLANO (x, y) EM RADIANOS; π/2 = FALHA AO LONGO DE x, QUE A SEÇÃO DE x CONSTANTE CORTA NO MERGULHO VERDADEIRO
        self.faultRoughness  = 3.54           # AMPLITUDE DA RUGOSIDADE DO PLANO
        self.faultRoughSigma = 8.55           # COMPRIMENTO DE ONDA DA RUGOSIDADE
        self.faultDecaySigma = (49, 59)       # ALCANCE DO REJEITO GAUSSIANO
        self.faultZoneWidth  = 0.99           # MEIA-ESPESSURA DO RÓTULO
        self.faultThreshold  = 0.77           # REJEITO MÍNIMO PARA ROTULAR
        self.faultCurveProb  = 0.07           # CHANCE DE A FALHA SER LÍSTRICA
        self.faultCurveMax   = 8.44           # CURVATURA MÁXIMA DA LÍSTRICA

        self.faultThrowPeriod = (0.0, 0.0)    # REJEITO MÁXIMO EM PERÍODOS DA WAVELET DO TILE NO LUGAR DO faultThrow; (0, 0) DESLIGA
        self.faultVisible     = 0.0           # SÓ ROTULA ONDE O REJEITO PASSA DESTA FRAÇÃO DO PERÍODO; 0 DESLIGA
        self.faultNormal      = 0.5           # CHANCE DE A FALHA SER NORMAL (O BLOCO DE CIMA DESCE); 0,5 É O SORTEIO DE SEMPRE, 1 VIRA TODA REVERSA EM NORMAL SEM SORTEIO A MAIS
        self.faultDrag        = (0.0, 0.0)    # FRAÇÃO DO REJEITO QUE VIRA FLEXURA (ARRASTO) NOS DOIS BLOCOS; (0, 0) NÃO SORTEIA
        self.faultDragWidth   = (4, 16)       # LARGURA DA FLEXURA EM VOXELS
        self.faultLabelMin    = 0.0           # REJEITO MÁXIMO (VOXELS) ABAIXO DO QUAL A FALHA DESLOCA SEM RÓTULO: SÓ AS PRINCIPAIS, COMO O INTÉRPRETE
        self.faultLabelPeriod = 0.0           # O MESMO EM PERÍODOS DA WAVELET DO TILE (O LIMIAR ACOMPANHA A ESCALA); 0 DESLIGA
        self.faultLabelEnergy = 0.0           # SÓ ROTULA ONDE A ENERGIA LOCAL DO REFLETOR SEM RUÍDO PASSA DESTA FRAÇÃO DA MEDIANA DO TILE; 0 DESLIGA

        self.minorCount = (0, 0)              # FALHAS MENORES SEM RÓTULO POR TILE (CURTAS E DE REJEITO PEQUENO, AS QUE O INTÉRPRETE NÃO DESENHA), TETO EXCLUSIVO; (0, 0) NÃO SORTEIA
        self.minorThrow = (0.15, 0.5)         # REJEITO MÁXIMO DA FALHA MENOR EM PERÍODOS DA WAVELET DO TILE
        self.minorSigma = (10, 30)            # ALCANCE (σ) DO REJEITO GAUSSIANO DA FALHA MENOR EM VOXELS
        self.minorLabel = False               # True ROTULA AS FALHAS MENORES COMO AS PRINCIPAIS (MESMA MEIA-ESPESSURA, MESMO REJEITO VISÍVEL); A IMAGEM NÃO MUDA

        self.waveletFreq     = (72, 99)       # FREQUÊNCIA DO RICKER
        self.waveletDuration = 0.1            # MEIA-DURAÇÃO DO KERNEL EM SEGUNDOS
        self.waveletDt       = 0.0012         # AMOSTRAGEM DO KERNEL
        self.waveletLog      = False          # True SORTEIA A FREQUÊNCIA EM ESCALA LOG: CADA OITAVA DE PERÍODO COM O MESMO PESO
        self.waveletPhase    = (0.0, 0.0)     # ROTAÇÃO DE FASE DO RICKER EM GRAUS; (0, 0) NÃO SORTEIA
        self.lateralSigma    = (0.0, 0.0)     # σ (x, y) DO BORRÃO LATERAL DO SINAL ANTES DO RUÍDO: A RESOLUÇÃO LATERAL DA MIGRAÇÃO; (0, 0) NÃO SORTEIA

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
        self.deadTraceBoth  = False           # True PÕE A BORDA NO LADO DE ÍNDICE ALTO METADE DAS VEZES (O TILE ESPELHADO NO EIXO); False É O SORTEIO DE SEMPRE, SÓ NO LADO BAIXO

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
        f                 = self.getFrequency() if any(self.faultThrowPeriod) or self.faultVisible > 0 or self.faultLabelPeriod > 0 or any(self.minorCount) else None
        model, mask       = self.applyFaulting(self.applyBodies(self.applyGainField(self.applyShearing(self.applyFolding(self.applyLenses(self.tensor(self.genReflectivity())))))), f)
        model, mask, rock = self.applyColumn(model, mask)
        wave              = self.applyLateral(self.applyWavelet(model, f))
        image, mask       = self.applyDeadTraces(self.crop(self.applyNoise(wave) * rock), self.crop(self.applyLabelEnergy(wave, mask)))
        image             = (image - image.mean()) / (image.std(correction=0) + 1e-8)
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

    # REFLETORES LENTICULARES: A REFLETIVIDADE 1D VIRA UM CUBO MODULADO POR UM CAMPO SUAVE, LONGO NA LATERAL E CURTO EM z, ANTES DA DOBRA
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

    # UM CORPO ENTRE O TOPO E A BASE DE UMA GAUSSIANA ELÍPTICA (upper 1 = MONTE DE BASE PLANA, 0 = CANAL DE TOPO PLANO, NO MEIO UMA LENTE)
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
        period       = None if frequency is None else 1 / (frequency * self.waveletDt)
        masks        = torch.zeros(self.shape, dtype=torch.uint8, device=self.device)
        model, masks = self.applyMinorFaults(self.tensor(reflectivity), masks, period)

        for _ in range(np.random.randint(*self.faultCount)):
            model, masks = self.applyFault(model, masks, period)

        return self.output(model, reflectivity), self.output(masks, reflectivity)

    # FALHAS MENORES ANTES DAS PRINCIPAIS (QUE AS CORTAM), NUM FLUXO PRÓPRIO DO np.random DEVOLVIDO NO FIM: O RESTO DO TILE É O DA MESMA SEMENTE SEM ELAS
    def applyMinorFaults(self, model, masks, period):
        if not any(self.minorCount):
            return model, masks

        visible = self.getVisible(period)
        grid    = self.getGrid()
        state   = np.random.get_state()
        np.random.seed((int(state[1][state[2] % 624]) + int(state[2])) % 2 ** 32)

        for _ in range(np.random.randint(*self.minorCount)):
            sense, dip, distStrike, distDip, distPlane = self.getPlane(grid)
            sigma  = np.random.uniform(*self.minorSigma)
            throw  = torch.exp((distStrike ** 2 + distDip ** 2) / -(2.0 * sigma ** 2)) * (np.random.uniform(*self.minorThrow) * period)
            throw  = -throw if sense < 0 and self.isNormal() else throw
            coords = [(points + throw * step).float() for points, step in zip(grid, dip)]
            model  = torch.where(distPlane > 0, self.resample(model, coords), model)

            if self.minorLabel:
                masks = torch.where(distPlane > 0, self.pick(masks, coords), masks) if masks.any() else masks
                masks = masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > visible))

        np.random.set_state(state)
        return model, masks

    def getGrid(self):
        return [self.tensor(np.arange(size)).reshape(axis) for size, axis in zip(self.shape, self.AXES)]

    # PLANO DE UMA FALHA: CENTRO, MERGULHO, AZIMUTE E SENTIDO SORTEADOS; DEVOLVE O SENTIDO, O VETOR DE MERGULHO E A DISTÂNCIA DE CADA VOXEL
    def getPlane(self, grid):
        p0        = np.random.uniform(0.15, 0.85, 3) * np.array(self.shape)
        dipRad    = np.deg2rad(np.random.uniform(*self.faultDipAngle))
        strikeRad = np.random.uniform(*self.faultStrike)
        sense     = np.random.choice([-1.0, 1.0])
        normal    = np.array([np.sin(dipRad) * np.cos(strikeRad), np.sin(dipRad) * np.sin(strikeRad), np.cos(dipRad) * sense])
        strike    = np.array([-normal[1], normal[0], 0.0])
        strike    = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(strike) < 1e-6 else strike / np.linalg.norm(strike)
        dip       = np.cross(normal, strike)
        dip      /= np.linalg.norm(dip)

        normal, strike, dip = normal.tolist(), strike.tolist(), dip.tolist()
        dx, dy, dz          = [points - origin for points, origin in zip(grid, p0)]
        return sense, dip, strike[0] * dx + strike[1] * dy + strike[2] * dz, dip[0] * dx + dip[1] * dy + dip[2] * dz, normal[0] * dx + normal[1] * dy + normal[2] * dz

    # REJEITO ACIMA DO QUAL A FALHA TEM RÓTULO: O LIMIAR FIXO OU A FRAÇÃO faultVisible DO PERÍODO, O QUE FOR MAIOR
    def getVisible(self, period):
        return max(self.faultThreshold, self.faultVisible * period) if self.faultVisible > 0 else self.faultThreshold

    def applyFault(self, model, masks, period=None):
        grid                                       = self.getGrid()
        sense, dip, distStrike, distDip, distPlane = self.getPlane(grid)

        bend      = self.getBend(distDip)
        distPlane = distPlane + self.gaussian(self.getField(self.shape), self.faultRoughSigma).double() * self.faultRoughness - bend
        limit     = np.random.uniform(*self.faultThrowPeriod) * period if any(self.faultThrowPeriod) else np.random.uniform(*self.faultThrow)
        throw     = self.getThrow(distStrike, distDip, limit)
        throw     = -throw if sense < 0 and self.isNormal() else throw
        visible   = self.getVisible(period)
        visible   = np.inf if limit < self.faultLabelMin or (self.faultLabelPeriod > 0 and limit < self.faultLabelPeriod * period) else visible

        if any(self.faultDrag):
            return self.applyDrag(model, masks, grid, dip, distPlane, throw, visible)

        hanging = distPlane > 0
        coords  = [(points + throw * step).float() for points, step in zip(grid, dip)]
        model   = torch.where(hanging, self.resample(model, coords), model)
        masks   = torch.where(hanging, self.pick(masks, coords), masks) if masks.any() else masks
        return model, masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > visible))

    # A FALHA COM sense −1 É REVERSA (A CAPA SOBE); O REJEITO NEGATIVO A FAZ NORMAL COM O MESMO PLANO E O MESMO RÓTULO. ENTRE 0,5 E 1 UM SORTEIO DECIDE
    def isNormal(self):
        if self.faultNormal <= 0.5 or self.faultNormal >= 1:
            return self.faultNormal >= 1

        return np.random.random() < 2 * self.faultNormal - 1

    # ARRASTO: A FRAÇÃO c DO REJEITO VIRA FLEXURA DE LARGURA w NOS DOIS BLOCOS E O RESTO É O SALTO NO PLANO; O RÓTULO VÊ O REJEITO TODO
    def applyDrag(self, model, masks, grid, dip, distPlane, throw, visible):
        c, w   = np.random.uniform(*self.faultDrag), np.random.uniform(*self.faultDragWidth)
        shift  = throw * torch.where(distPlane > 0, 1 - 0.5 * c * torch.exp(-distPlane.clamp(min=0) / w), 0.5 * c * torch.exp(distPlane.clamp(max=0) / w))
        coords = [(points + shift * step).float() for points, step in zip(grid, dip)]
        masks  = self.pick(masks, coords) if masks.any() else masks
        return self.resample(model, coords), masks | ((distPlane.abs() <= self.faultZoneWidth) & (throw.abs() > visible))

    # COLUNA DE UM BLOCO REAL: ÁGUA ACIMA DO FUNDO DO MAR (rock ZERA A IMAGEM DEPOIS DO RUÍDO) E ZONA MORTA FRACA ABAIXO DE UMA SUPERFÍCIE
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

    # HACHURA DE MIGRAÇÃO: RUÍDO FILTRADO NO ESPECTRO PERTO DAS NORMAIS DE DOIS PLANOS QUE MERGULHAM ±θ NUM AZIMUTE SORTEADO
    def getHatch(self):
        theta  = np.radians(np.random.uniform(*self.hatchDip))
        phi    = np.random.uniform(0, np.pi)
        period = np.random.uniform(*self.hatchPeriod)
        k      = [self.tensor(f).reshape(shape) for f, shape in zip((np.fft.fftfreq(self.nx), np.fft.fftfreq(self.ny), np.fft.rfftfreq(self.nz)), self.AXES)]
        radius = torch.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2) + 1e-12
        cone   = sum(torch.exp(-(1 - ((sign * np.sin(theta) * (np.cos(phi) * k[0] + np.sin(phi) * k[1]) + np.cos(theta) * k[2]) / radius) ** 2) / (2 * 0.15 ** 2)) for sign in (1, -1))
        band   = torch.exp(-((radius - 1 / period) ** 2) / (2 * (0.3 / period) ** 2))
        hatch  = torch.fft.irfftn(torch.fft.rfftn(self.getField(self.shape).double()) * cone * band, s=self.shape)
        return hatch / (hatch.std(correction=0) + 1e-8)

    # BORDA DO LEVANTAMENTO COMO NOS TILES DE BORDA DO MARLIM: O LADO DE FORA DE UMA POSIÇÃO LATERAL É O ESPELHO DO LADO DE DENTRO
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
        image, mask = image.index_select(axis, index) * gain, mask.index_select(axis, index) * (gain >= 1)

        if self.deadTraceBoth and np.random.random() < 0.5:
            image, mask = torch.flip(image, (axis,)), torch.flip(mask, (axis,))

        return image, mask

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

    def getFrequency(self):
        return np.exp(np.random.uniform(*np.log(self.waveletFreq))) if self.waveletLog else np.random.uniform(*self.waveletFreq)

    # O convolve1d DO RICKER EM z COMO MATRIZ: O FILTRO DO scipy APLICADO ÀS LINHAS DA IDENTIDADE
    def applyWavelet(self, model, frequency=None):
        f       = self.getFrequency() if frequency is None else frequency
        t       = np.arange(-self.waveletDuration, self.waveletDuration, self.waveletDt)
        wavelet = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-((np.pi * f * t) ** 2))
        wavelet = self.rotate(wavelet) if any(self.waveletPhase) else wavelet
        return self.output(self.tensor(model) @ self.tensor(ndimage.convolve1d(np.eye(self.nz), wavelet, axis=1)), model)

    # O DADO MIGRADO NÃO RESOLVE A FALHA NA LATERAL COMO O MODELO: O CORTE E OS REFLETORES SAEM BORRADOS EM (x, y); O RUÍDO ENTRA DEPOIS, FINO
    def applyLateral(self, wave):
        if not any(self.lateralSigma):
            return wave

        sigma = np.random.uniform(*self.lateralSigma)
        return self.output(self.gaussian(self.tensor(wave), (sigma, sigma, 0)), wave)

    # FASE DO RICKER GIRADA PELA TRANSFORMADA DE HILBERT: 90° É A QUADRATURA, 180° INVERTE A POLARIDADE
    def rotate(self, wavelet):
        phase = np.radians(np.random.uniform(*self.waveletPhase))
        return np.cos(phase) * wavelet + np.sin(phase) * np.imag(hilbert(wavelet))

    # RÓTULO SÓ ONDE HÁ REFLETOR PARA DESLOCAR: ONDE A ENERGIA LOCAL (RMS DE σ 4) DO TRAÇO SEM RUÍDO FICA ABAIXO DE faultLabelEnergy × A
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


# DESCRITORES DE UM TILE PELAS SEÇÕES (z, lateral): COMPOSIÇÃO, CONTRASTE, PERÍODO, MERGULHO, CONTINUIDADE E RUÍDO FINO; O MESMO CÓDIGO NO TILE REAL E NO SINTÉTICO
class TileRuler:
    ALIVE     = 0.0552                  # ENVELOPE RMS DO REFLETOR VIVO NA ESCALA [0, 1] (OTSU DO MARLIM)
    WATER_TOL = 0.002                   # |x - ÁGUA| QUE AINDA É ÁGUA
    DEAD_TOL  = 0.02                    # TRAÇO QUE NUNCA SAI DISSO É TRAÇO MORTO
    SLICES    = (32, 64, 96)            # SEÇÕES INLINE (EIXO 0) E CROSSLINE (EIXO 2) MEDIDAS
    STEEP     = (40, 50, 60, 70, 80)    # MERGULHOS DE LINEAMENTO TIPO FALHA, EM GRAUS
    FLAT      = (0, 10, 20)             # MERGULHOS DE LINEAMENTO AO LONGO DA CAMADA
    LINE      = 31                      # COMPRIMENTO DO NÚCLEO DE LINEAMENTO EM PX

    def __init__(self, water=None):
        self.water   = water
        self.kernels = {angle: self.getKernel(angle) for angle in self.STEEP + self.FLAT}

    # DESFOQUE GAUSSIANO 2D COM BORDA REFLETIDA, O MESMO EM TODAS AS MEDIDAS
    @staticmethod
    def gauss(img, sigma):
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

    def get(self, tile):
        tile  = np.asarray(tile, np.float32)
        table = pd.DataFrame([self.getSection(tile[i]) for i in self.SLICES] + [self.getSection(tile[:, :, j].T) for j in self.SLICES])
        return {**table.mean(numeric_only=True).to_dict(), **self.getVolume(tile)}

    # O QUE SÓ FAZ SENTIDO NO VOLUME: CONTRASTE, SATURAÇÃO, TRAÇOS MORTOS E O RUÍDO FINO DA SUB-BANDA HHH DE HAAR
    def getVolume(self, tile):
        center = self.water if self.water is not None else float(np.median(tile))
        a      = tile - center
        t      = tile[:tile.shape[0] // 2 * 2, :tile.shape[1] // 2 * 2, :tile.shape[2] // 2 * 2]
        hhh    = sum((-1) ** (i + j + k) * t[i::2, j::2, k::2] for i, j, k in np.ndindex(2, 2, 2)) / np.sqrt(8)
        rock   = np.abs(a[::2, ::2, ::2]) > self.WATER_TOL
        live   = np.abs(a) > self.WATER_TOL
        sd     = float(a[live].std()) if live.any() else 1e-6
        return {
            'sigma': float(tile.std()),
            'sat': float(((tile <= 0.002) | (tile >= 0.998)).mean()),
            'deadTraces': float((np.abs(a) < self.DEAD_TOL).all(axis=1).mean()),
            'fine': float(np.median(np.abs(hhh[rock])) / 0.6745 / (sd + 1e-9)) if rock.any() else np.nan
        }

    def getSection(self, section):
        a     = section.astype(np.float32) - (self.water if self.water is not None else float(np.median(section)))
        live  = np.sqrt(self.gauss(a * a, 12)) > self.ALIVE
        out   = {'water': float((np.abs(a) < self.WATER_TOL).mean()), 'live': float(live.mean())}

        if live.mean() < 0.05:
            return out

        zs      = a / (float(a[live].std()) + 1e-8)
        columns = live.mean(0) > 0.5
        return {**out, **self.getSpectrum(zs, columns), **self.getStructure(zs, live), **self.getContinuity(zs, live, columns), **self.getContrast(a, live)}

    # PERÍODO VERTICAL PELO CENTRÓIDE DO ESPECTRO EM z NAS COLUNAS VIVAS, E A LARGURA RELATIVA DA BANDA
    def getSpectrum(self, zs, columns):
        if columns.sum() < 8:
            return {}

        vert  = zs[:, columns] - zs[:, columns].mean(0)
        power = (np.abs(np.fft.rfft(vert, axis=0)) ** 2).mean(1)[1:]
        freqs = np.fft.rfftfreq(vert.shape[0])[1:]
        mean  = float((freqs * power).sum() / (power.sum() + 1e-12))
        width = float(np.sqrt(((freqs - mean) ** 2 * power).sum() / (power.sum() + 1e-12)))
        return {'period': 1.0 / (mean + 1e-12), 'bandwidth': width / (mean + 1e-12)}

    def getStructure(self, zs, live):
        Jzz, Jxx, Jzx = self.getTensor(zs)
        trace = (Jzz + Jxx) * live
        coh   = np.sqrt(np.maximum((Jzz - Jxx) ** 2 + 4 * Jzx ** 2, 0)) / (Jzz + Jxx + 1e-12)
        dip   = np.abs(np.degrees(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx))).ravel()
        order = np.argsort(dip)
        cum   = np.cumsum(trace.ravel()[order]) / (trace.sum() + 1e-12)
        pick  = lambda q: float(dip[order][min(np.searchsorted(cum, q), cum.size - 1)])
        return {'coherence': float((coh * trace).sum() / (trace.sum() + 1e-12)), 'dipP50': pick(0.5), 'dipP90': pick(0.9)}

    def getContinuity(self, zs, live, columns):
        s5, s17 = self.getSemblance(zs, 2), self.getSemblance(zs, 8)
        disc    = np.clip(1 - s5, 0, 1).astype(np.float32)
        steep   = np.max([cv2.filter2D(disc, -1, self.kernels[a], borderType=cv2.BORDER_REFLECT) for a in self.STEEP], axis=0)
        flat    = np.max([cv2.filter2D(disc, -1, self.kernels[a], borderType=cv2.BORDER_REFLECT) for a in self.FLAT], axis=0)
        sd      = zs.std(0)
        columns = columns & (sd > 1e-3)    # TRAÇO MORTO CONSTANTE NO MEIO DE TRAÇOS VIVOS (O ENVELOPE σ 12 O MARCA VIVO) DIVIDIA POR ~0
        col     = zs[:, columns] / sd[None, columns] if columns.sum() > 20 else None
        return {
            'disc5': float((s5 < 0.8)[live].mean()),
            'disc17': float((s17 < 0.8)[live].mean()),
            'steep90': float(np.percentile(steep[live], 90)),
            'flat90': float(np.percentile(flat[live], 90)),
            'lag4': float((col[:, :-4] * col[:, 4:]).mean()) if col is not None else np.nan,
            'lag16': float((col[:, :-16] * col[:, 16:]).mean()) if col is not None else np.nan
        }

    def getContrast(self, a, live):
        vals  = a[live]
        zs    = (vals - vals.mean()) / (vals.std() + 1e-8)
        third = a.shape[0] // 3
        top   = a[:third][live[:third]]
        base  = a[-third:][live[-third:]]
        trend = float(np.log((top.std() + 1e-6) / (base.std() + 1e-6))) if top.size > 100 and base.size > 100 else np.nan
        return {'kurt': float((zs ** 4).mean()), 'trend': trend, 'sigmaLive': float(vals.std())}

    def getTensor(self, zs):
        smooth = self.gauss(zs, 1.0)
        gz, gx = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        return self.gauss(gz * gz, 4.0), self.gauss(gx * gx, 4.0), self.gauss(gz * gx, 4.0)

    # NÚCLEO DE LINHA DE LINE px NO MERGULHO DADO (0 = HORIZONTAL), NORMALIZADO PARA MÉDIA
    def getKernel(self, angle):
        size   = self.LINE + (self.LINE + 1) % 2
        kernel = np.zeros((size, size), np.float32)
        c, r   = size // 2, self.LINE / 2
        dz, dx = np.sin(np.radians(angle)), np.cos(np.radians(angle))
        cv2.line(kernel, (int(round(c - r * dx)), int(round(c - r * dz))), (int(round(c + r * dx)), int(round(c + r * dz))), 1.0, 1)
        return kernel / kernel.sum()

    # SEMBLANCE DE 2·half+1 TRAÇOS SEGUINDO O MERGULHO LOCAL, NUMA JANELA DE 9 AMOSTRAS (A MESMA DO Regions/Marlim)
    def getSemblance(self, section, half, win=4):
        Jzz, Jxx, Jzx = self.getTensor(section)
        dip    = np.clip(-np.tan(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)), -3, 3)
        zz, xx = np.mgrid[:section.shape[0], :section.shape[1]].astype(np.float32)
        traces = [cv2.remap(section, xx + k, zz + dip * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for k in range(-half, half + 1)]
        window = np.ones((2 * win + 1, 1), np.float32)
        num    = cv2.filter2D(sum(traces) ** 2, -1, window, borderType=cv2.BORDER_REFLECT)
        den    = cv2.filter2D(sum(t * t for t in traces) * len(traces), -1, window, borderType=cv2.BORDER_REFLECT)
        return num / (den + 1e-12)
