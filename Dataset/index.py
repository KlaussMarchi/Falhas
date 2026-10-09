import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob, os, shutil


# ARQUIVOS DE UMA PASTA (OU PADRÃO GLOB), EM CAMINHO ABSOLUTO E ORDEM ALFABÉTICA
def getFiles(path, limit=None, shuffle=False):
    target = sorted([os.path.abspath(f) for f in glob.glob(os.path.join(path, '*'))])
    if shuffle:
        np.random.shuffle(target)
    return target[:limit]


# APAGA E RECRIA A PASTA
def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# AS TRÊS SEÇÕES CENTRAIS DE UM VOLUME, COM A MÁSCARA POR CIMA QUANDO ELA VEM
def showTile(img=None, mask=None, save=None):
    if img is None and mask is None:
        return print("Erro: Forneça pelo menos 'img' ou 'mask'.")

    ref_vol = img if img is not None else mask
    mid_x = ref_vol.shape[0] // 2
    mid_y = ref_vol.shape[1] // 2
    mid_z = ref_vol.shape[2] // 2

    def get_slices(vol):
        if vol is None:
            return None

        s_x = np.array(vol[mid_x, :, :]) # Plano YZ
        s_y = np.array(vol[:, mid_y, :]) # Plano XZ
        s_z = np.array(vol[:, :, mid_z]) # Plano XY
        return [s_x, np.rot90(s_z, -1), s_y]

    img_slices  = get_slices(img)
    mask_slices = get_slices(mask)
    cmap_mask_only    = ListedColormap(['black', 'red', 'green', 'blue'])
    cmap_mask_overlay = ListedColormap([(0, 0, 0, 0), 'red', 'green', 'blue'])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles    = [f'Slice X={mid_x}', f'Slice Y={mid_y}', f'Slice Z={mid_z}']

    for i, ax in enumerate(axes):
        if img is not None:
            ax.imshow(img_slices[i], cmap='gray')

        if mask is not None:
            if img is not None:
                ax.imshow(mask_slices[i], cmap=cmap_mask_overlay, vmin=0, vmax=3, alpha=0.6)
            else:
                ax.imshow(mask_slices[i], cmap=cmap_mask_only, vmin=0, vmax=3)

        ax.set_title(titles[i])

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        return plt.close(fig)

    plt.show()


# ESCALONAMENTO DOS VOLUMES NO FORMAT, ESCOLHIDO PELO scaling DO info.json E GRAVADO NA COLUNA scaling DO DataBase.csv
class Normalization:
    OPTIONS = ('normalize', 'percentile', 'standardize', None)
    DEFAULT = 'normalize'                  # O MIN-MAX DE CADA VOLUME, DECISÃO DE 06/10/2026
    UNIT    = ('normalize', 'percentile')  # OS QUE DEIXAM O VOLUME EM [0, 1], A ESCALA DOS .dat DO MARLIM

    def __init__(self, scaling=DEFAULT):
        if scaling not in self.OPTIONS:
            raise ValueError(f'scaling={scaling!r} desconhecido, use um de {self.OPTIONS}')

        self.scaling = scaling
        self.low     = None    # p01 DO CONJUNTO, SÓ NO 'percentile'
        self.high    = None    # p99 DO CONJUNTO, SÓ NO 'percentile'

    # O PERCENTIL É DO CONJUNTO INTEIRO: LÊ TODOS OS VOLUMES UMA VEZ, ANTES DO PRIMEIRO; OS OUTROS MÉTODOS NÃO LEEM NADA
    def fit(self, paths, load):
        if self.scaling != 'percentile':
            return self

        arrays = [load(path) for path in paths]
        self.low, self.high = (float(value) for value in np.percentile(arrays, [1, 99]))
        del arrays
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
        names = {'normalize': 'min-max de cada volume para [0, 1]', 'percentile': f'corte no p01/p99 do conjunto ({self.low}, {self.high}) para [0, 1]', 'standardize': 'padronização de cada volume (média 0, desvio 1)', None: 'volume cru, sem escalonamento'}
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

        if os.path.exists(main_dir):
            shutil.rmtree(main_dir)
        os.makedirs(main_dir)

    def update(self, file_path, target_dir, is_mask=False):
        folder = os.path.join(self.main_dir, target_dir)
        os.makedirs(folder, exist_ok=True)

        data = np.load(file_path)
        base_name     = os.path.splitext(os.path.basename(file_path))[0]
        z_t, y_t, x_t = self.target_size

        z_pad = (z_t - data.shape[0] % z_t) % z_t
        y_pad = (y_t - data.shape[1] % y_t) % y_t
        x_pad = (x_t - data.shape[2] % x_t) % x_t

        pad_mode    = 'constant' if is_mask else 'reflect'
        data_padded = np.pad(data, ((0, z_pad), (0, y_pad), (0, x_pad)), mode=pad_mode) if z_pad > 0 or y_pad > 0 or x_pad > 0 else data

        index = 0
        for z in range(0, data_padded.shape[0], z_t):
            for y in range(0, data_padded.shape[1], y_t):
                for x in range(0, data_padded.shape[2], x_t):
                    tile = data_padded[z:z+z_t, y:y+y_t, x:x+x_t]
                    save_path = os.path.join(folder, f'{base_name}_tile_{index:04d}.npy')

                    np.save(save_path, tile)
                    index = (index + 1)
