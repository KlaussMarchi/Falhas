import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob, os, shutil, itertools


def getFiles(path, limit=None, shuffle=False):
    files = sorted(os.path.abspath(file) for file in glob.glob(os.path.join(path, '*')))

    if shuffle:
        np.random.shuffle(files)

    return files[:limit]


def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


# AS TRÊS SEÇÕES CENTRAIS DE UM VOLUME, COM A MÁSCARA POR CIMA QUANDO ELA VEM
def showTile(img=None, mask=None, save=None):
    overlay = img is not None
    x, y, z = (size // 2 for size in (img if overlay else mask).shape[:3])
    cuts    = {f'Slice X={x}': lambda vol: vol[x], f'Slice Y={y}': lambda vol: np.rot90(vol[:, :, z], -1), f'Slice Z={z}': lambda vol: vol[:, y]}
    masking = {'cmap': ListedColormap([(0, 0, 0, 0) if overlay else 'black', 'red', 'green', 'blue']), 'vmin': 0, 'vmax': 3, 'alpha': 0.6 if overlay else None}
    layers  = [(np.asarray(vol), options) for vol, options in ((img, {'cmap': 'gray'}), (mask, masking)) if vol is not None]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, cut) in zip(axes, cuts.items()):
        for vol, options in layers:
            ax.imshow(cut(vol), **options)

        ax.set_title(title)

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        return plt.close(fig)

    plt.show()


# A ESCALA DE UM CONJUNTO DE VOLUMES DO MESMO TAMANHO: EXTENSÃO E dtype DO ARQUIVO LIDO, SHAPE E MÍNIMO, MÁXIMO, MÉDIA E DESVIO DE TODOS OS VOXELS JUNTOS
def getDataInfo(paths, load, name):
    first = load(paths[0])
    stats = np.array([[img.min(), img.max(), img.mean(), np.square(img).mean()] for img in (load(path).astype(np.float64) for path in paths)])
    mean  = stats[:, 2].mean()

    info = {
        'arquivos': len(paths),
        'formato': f'{os.path.splitext(paths[0])[1]} {first.dtype}',
        'shape': first.shape,
        'min': stats[:, 0].min(),
        'max': stats[:, 1].max(),
        'media': mean,
        'desvio': np.sqrt(stats[:, 3].mean() - mean ** 2)
    }
    return pd.DataFrame([info], index=[name])


# ESCALONAMENTO DOS VOLUMES NO FORMAT, ESCOLHIDO PELO scaling DO info.json E GRAVADO NA COLUNA scaling DO DataBase.csv
class Normalization:
    OPTIONS = ('normalize', 'percentile', 'standardize', None)
    DEFAULT = 'normalize'    # O MIN-MAX DE CADA VOLUME, DECISÃO DE 06/10/2026

    def __init__(self, scaling=DEFAULT):
        if scaling not in self.OPTIONS:
            raise ValueError(f'scaling={scaling!r} desconhecido, use um de {self.OPTIONS}')

        self.scaling = scaling
        self.low     = None    # p01 DO CONJUNTO, SÓ NO 'percentile'
        self.high    = None    # p99 DO CONJUNTO, SÓ NO 'percentile'

    # O PERCENTIL É DO CONJUNTO INTEIRO: LÊ TODOS OS VOLUMES UMA VEZ, ANTES DO PRIMEIRO; OS OUTROS MÉTODOS NÃO LEEM NADA
    def fit(self, paths, load):
        if self.scaling == 'percentile':
            self.low, self.high = np.percentile([load(path) for path in paths], [1, 99]).tolist()

        return self

    def __call__(self, img):
        if self.scaling == 'normalize':
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.scaling == 'percentile':
            return (np.clip(img, self.low, self.high) - self.low) / (self.high - self.low)

        if self.scaling == 'standardize':
            return (img - np.mean(img)) / np.std(img)

        return img

    def __repr__(self):
        names = {
            'normalize': 'min-max de cada volume para [0, 1]',
            'percentile': f'corte no p01/p99 do conjunto ({self.low}, {self.high}) para [0, 1]',
            'standardize': 'padronização de cada volume (média 0, desvio 1)',
            None: 'volume cru, sem escalonamento'
        }
        return f'Normalization({self.scaling!r}): {names[self.scaling]}'

    # ESCALONAMENTO DE UM DataBase.csv: SEM A COLUNA É O DEFAULT, CÉLULA VAZIA É None
    @classmethod
    def read(cls, database):
        df = pd.read_csv(database, nrows=1)

        if 'scaling' not in df:
            return cls.DEFAULT

        value = df['scaling'].iloc[0]
        return value if isinstance(value, str) else None


# CORTA OS VOLUMES DE images/ E masks/ EM BLOCOS DE target_size SEM SOBREPOSIÇÃO, COMPLETANDO A BORDA (REFLEXÃO NA IMAGEM, ZERO NA MÁSCARA)
class TilesBuilder:
    def __init__(self, target_size=(8, 64, 128), main_dir='tiles'):
        if len(target_size) != 3 or any(int(size) <= 0 for size in target_size):
            raise ValueError(f'img_size precisa de 3 valores positivos (z, y, x), recebido: {target_size}')

        self.target_size = tuple(int(size) for size in target_size)
        self.main_dir    = main_dir
        setFolder(main_dir)

    def update(self, path, folder, is_mask=False):
        output = os.path.join(self.main_dir, folder)
        os.makedirs(output, exist_ok=True)

        name   = os.path.splitext(os.path.basename(path))[0]
        data   = np.load(path)
        pad    = [(0, (size - n % size) % size) for n, size in zip(data.shape, self.target_size)]
        data   = np.pad(data, pad, mode='constant' if is_mask else 'reflect') if any(after for _, after in pad) else data
        starts = itertools.product(*(range(0, n, size) for n, size in zip(data.shape, self.target_size)))
        zt, yt, xt = self.target_size

        for index, (z, y, x) in enumerate(starts):
            np.save(os.path.join(output, f'{name}_tile_{index:04d}.npy'), data[z:z + zt, y:y + yt, x:x + xt])
