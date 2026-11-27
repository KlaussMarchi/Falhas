import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize


# clDice COM TOLERÂNCIA (SHIT ET AL., CVPR 2021) QUE SOMA OS VOXELS DE TODAS AS CHAMADAS ANTES DE DIVIDIR, COMO A ConfusionMatrix
class ClDice:
    def __init__(self, tol=3.0):
        self.tol = float(tol)
        self.hit = self.skel = self.cover = self.ref = 0

    # refSkeleton E refNear SÃO O ESQUELETO E A VIZINHANÇA DE target JÁ CALCULADOS, PARA UMA REFERÊNCIA FIXA EM CACHE
    def update(self, pred, target=None, refSkeleton=None, refNear=None):
        refSkeleton = self.skeleton(target) if refSkeleton is None else refSkeleton
        refNear     = self.near(target) if refNear is None else refNear

        if pred.shape != refSkeleton.shape:
            raise ValueError(f'predição {tuple(pred.shape)} e referência {tuple(refSkeleton.shape)} com shapes diferentes')

        skelPred    = self.skeleton(pred)
        self.hit   += int(refNear[skelPred].sum())
        self.skel  += int(skelPred.sum())
        self.cover += int(self.near(pred)[refSkeleton].sum())
        self.ref   += int(refSkeleton.sum())
        return self

    def metrics(self):
        hit   = self.hit / max(self.skel, 1)
        cover = self.cover / max(self.ref, 1)
        return {'cldice': 2 * cover * hit / max(cover + hit, 1e-9), 'acerto': hit, 'cobertura': cover}

    # TUDO O QUE FICA A ATÉ tol AMOSTRAS DA MÁSCARA; VAZIA QUANDO ELA É VAZIA (A DISTÂNCIA NÃO TEM PARA ONDE MEDIR)
    def near(self, mask):
        return ndi.distance_transform_edt(~mask) <= self.tol if mask.any() else np.zeros(mask.shape, bool)

    # NO 3D, A UNIÃO DOS ESQUELETOS 2D DAS SEÇÕES DO PRIMEIRO E DO ÚLTIMO EIXO: O skeletonize 3D REDUZ UMA SUPERFÍCIE A LINHAS
    @staticmethod
    def skeleton(mask):
        if mask.ndim == 2:
            return skeletonize(mask)

        out = np.zeros(mask.shape, bool)

        for i in np.flatnonzero(mask.any(axis=(1, 2))):
            out[i] = skeletonize(mask[i])

        for k in np.flatnonzero(mask.any(axis=(0, 1))):
            out[:, :, k] |= skeletonize(mask[:, :, k])

        return out

    # PROBABILIDADE EM PASSOS DE 1/255, A RESOLUÇÃO DO ORÇAMENTO IGUAL
    @staticmethod
    def quantize(probs):
        return np.round(np.clip(probs, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def histogram(probs):
        levels = probs if probs.dtype == np.uint8 else ClDice.quantize(probs)
        return np.bincount(levels.ravel(), minlength=256)

    # ORÇAMENTO IGUAL: O LIMIAR QUE FAZ A PREDIÇÃO MARCAR A FRAÇÃO PEDIDA, PELO HISTOGRAMA DE UM VOLUME OU SOMADO DE VÁRIOS
    @staticmethod
    def threshold(hist, fraction):
        above = np.cumsum(hist[::-1])[::-1] / hist.sum()
        return float(np.argmin(np.abs(above - fraction)) / 255)
