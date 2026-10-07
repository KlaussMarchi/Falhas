import numpy as np
from scipy import ndimage
from scipy.special import erf
from monai.inferers import sliding_window_inference


# AUMENTAÇÃO DINÂMICA DO TREINO NA RECEITA DA RESACEUNET (ZU ET AL. 2024, data/build_data.py DE github.com/39c5bb-miku/ResACEUnet) MAIS O ZOOM DA ESCALA DO MARLIM
class Transforms:
    OPTIONS = ('zoom', 'contrast', 'rotate90', 'flip', 'rotate', 'smooth', 'noise', 'crop')

    def __init__(self, options=None, seed=42, batch=1):
        self.options = dict(options or {})
        self.seed    = seed
        self.batch   = batch

        # n_aug: QUANTAS VARIAÇÕES DE CADA TILE ENTRAM POR ÉPOCA (1 NOS AUTORES, O num_samples DO RandCropByPosNegLabeld)
        self.copies = int(self.options.pop('n_aug', 1))

        unknown = set(self.options) - set(self.OPTIONS)
        if unknown:
            raise ValueError(f'augmentations desconhecidas: {sorted(unknown)}; as opções são {self.OPTIONS}')

        crop = self.options.get('crop') or {}
        self.window  = tuple(crop['size']) if crop else None
        self.overlap = crop.get('overlap', 0.5)
        self.mode    = crop.get('mode', 'gaussian')

    # APLICA AS AUMENTAÇÕES NA ORDEM DO JSON; A SEMENTE (seed, epoch, index) DÁ O MESMO SORTEIO EM QUALQUER WORKER OU ORDEM
    def apply(self, img, mask, epoch=0, index=0):
        rng = np.random.default_rng([self.seed, epoch, index])

        for name, params in self.options.items():
            if rng.random() < params.get('prob', 1.0):
                img, mask = getattr(self, name)(img, mask, rng, params)

        return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(mask, np.float32)

    # ZOOM ISOTRÓPICO: UM CUBO DE LADO tile/f EM POSIÇÃO SORTEADA VOLTA AO TAMANHO DO TILE, COM f LOG-UNIFORME EM factor. PERÍODO,
    # REJEITO E ESPAÇAMENTO DAS FALHAS CRESCEM JUNTOS E O MERGULHO FICA; f < 1 EXIGIRIA INVENTAR BORDA, POR ISSO É PROIBIDO
    def zoom(self, img, mask, rng, params):
        if min(params['factor']) < 1:
            raise ValueError(f'zoom com fator {params["factor"]}: abaixo de 1 o tile encolheria e a borda teria de ser inventada')

        factor = float(np.exp(rng.uniform(*np.log(params['factor']))))
        shape  = np.array(img.shape)
        start  = rng.uniform(0, shape - shape / factor)
        offset = start + 0.5 / factor - 0.5

        img  = ndimage.affine_transform(img, np.full(img.ndim, 1 / factor), offset, order=1, mode='nearest')
        mask = ndimage.affine_transform(mask, np.full(mask.ndim, 1 / factor), offset, order=1, mode='nearest') >= 0.5
        return img, mask.astype(np.float32)

    # GAMMA NA FAIXA DO PRÓPRIO TILE (AdjustContrast): O MÍNIMO E O MÁXIMO FICAM, O MEIO SOBE OU DESCE
    def contrast(self, img, mask, rng, params):
        gamma = float(rng.uniform(*params['gamma']))
        low, span = img.min(), img.max() - img.min()
        return ((img - low) / (span + 1e-7)) ** gamma * span + low, mask

    # GIRO DE 90, 180 OU 270 GRAUS NO PLANO DOS EIXOS (RandRotate90d); É UMA VIEW, NÃO COPIA O TILE
    def rotate90(self, img, mask, rng, params):
        k, axes = int(rng.integers(1, 4)), tuple(params['axes'])
        return np.rot90(img, k, axes), np.rot90(mask, k, axes)

    # ESPELHA TODOS OS EIXOS DA LISTA NUM SORTEIO SÓ (RandFlipd COM spatial_axis=[0, 1]); TAMBÉM É VIEW
    def flip(self, img, mask, rng, params):
        axes = tuple(params['axes'])
        return np.flip(img, axes), np.flip(mask, axes)

    # ROTAÇÃO LIVRE EM TORNO DO CENTRO, UM ÂNGULO EM [-angle, angle] POR PLANO, BILINEAR E BORDA ZERO (RandRotated)
    # A MÁSCARA PASSA PELA MESMA BILINEAR DOS AUTORES E VOLTA A BINÁRIA EM 0.5, PORQUE AQUI O RÓTULO É INTEIRO
    def rotate(self, img, mask, rng, params):
        matrix = np.eye(3)

        for first, second in params['planes']:
            angle = rng.uniform(-params['angle'], params['angle'])
            plane = np.eye(3)
            plane[[first, first, second, second], [first, second, first, second]] = [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]
            matrix = matrix @ plane

        center = (np.array(img.shape) - 1) / 2
        offset = center - matrix @ center
        img  = ndimage.affine_transform(img, matrix, offset, order=1, mode='grid-constant', cval=0.0)
        mask = ndimage.affine_transform(mask, matrix, offset, order=1, mode='grid-constant', cval=0.0) >= 0.5
        return img, mask.astype(np.float32)

    # SUAVIZAÇÃO GAUSSIANA COM UM SIGMA SORTEADO POR EIXO, KERNEL INTEGRADO POR ERF E BORDA ZERO (RandGaussianSmoothd)
    def smooth(self, img, mask, rng, params):
        for axis, sigma in enumerate(rng.uniform(*params['sigma'], size=img.ndim)):
            tail   = int(max(sigma * 4.0, 0.5) + 0.5)
            x      = np.arange(-tail, tail + 1)
            kernel = 0.5 * (erf((x + 0.5) / (sigma * np.sqrt(2))) - erf((x - 0.5) / (sigma * np.sqrt(2))))
            img    = ndimage.correlate1d(img, kernel, axis=axis, mode='constant', cval=0.0)

        return img, mask

    # RUÍDO GAUSSIANO ADITIVO COM O DESVIO SORTEADO EM [0, std] (RandGaussianNoised COM sample_std)
    def noise(self, img, mask, rng, params):
        std = float(rng.uniform(0, params['std']))
        return img + rng.standard_normal(img.shape, dtype=np.float32) * np.float32(std), mask

    # JANELA CENTRADA NUM VOXEL DE FALHA (pos) OU DE FUNDO COM SINAL (neg), EMPURRADA PARA DENTRO DO TILE (RandCropByPosNegLabeld)
    def crop(self, img, mask, rng, params):
        size, shape = np.array(params['size']), np.array(img.shape)
        if np.any(size > shape):
            raise ValueError(f'recorte {tuple(size)} maior que o tile {tuple(shape)}')

        fault    = np.flatnonzero(mask > 0)
        ground   = np.flatnonzero((mask == 0) & (img > 0))
        positive = rng.random() < params['pos'] / (params['pos'] + params['neg'])
        indices  = fault if (positive and len(fault)) or not len(ground) else ground

        center = np.unravel_index(indices[rng.integers(len(indices))], img.shape)
        start  = np.clip(np.array(center) - size // 2, 0, shape - size)
        window = tuple(slice(s, s + w) for s, w in zip(start, size))
        return img[window], mask[window]

    # PREDIÇÃO NA JANELA DO TREINO: TILE DO TAMANHO DA JANELA VAI DIRETO, MAIOR VAI POR JANELA DESLIZANTE (predict_3d.py DOS AUTORES)
    def infer(self, model, imgs):
        if self.window is None or tuple(imgs.shape[2:]) == self.window:
            return model(imgs)

        return sliding_window_inference(imgs, self.window, self.batch, model, overlap=self.overlap, mode=self.mode)
