import os, re, io, sys, json, glob, time
import numpy as np
import pandas as pd
import torch
import nbformat
import papermill as pm
from PIL import Image
from contextlib import contextmanager, redirect_stdout, redirect_stderr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.append(ROOT)
sys.path.append(f'{ROOT}/Model')    # O Blocks.index CARREGA A REDE PELO Network/ DE Model/
from Blocks.index import BlockPredictor, SlidingWindow, TileGrid
from Metrics.ClDice.index import ClDice
from Dataset.index import Normalization


# EXECUTA UM NOTEBOOK DO PIPELINE NUMA CÓPIA NA PASTA DO log, COM O cwd NA PASTA DELE: replace TROCA TRECHOS DO CÓDIGO E parameters ENTRAM PELO PAPERMILL
def runNotebook(path, log, replace=None, parameters=None):
    nb = nbformat.read(path, as_version=4)

    for old, new in (replace or {}).items():
        cells = [cell for cell in nb.cells if cell.cell_type == 'code' and old in cell.source]

        if not cells:
            raise ValueError(f'{path}: trecho {old!r} não existe mais; a cópia gravaria no lugar errado')

        for cell in cells:
            cell.source = cell.source.replace(old, new)

    if parameters:
        config = next(cell for cell in nb.cells if cell.cell_type == 'code' and cell.source.startswith(tuple(parameters)))
        config.metadata['tags'] = ['parameters']

    temp = os.path.join(os.path.dirname(os.path.abspath(log)), os.path.basename(path).replace('.ipynb', '_tmp.ipynb'))    # NADA DE ARQUIVO NOVO NAS PASTAS DO PIPELINE
    nbformat.write(nb, temp)

    try:
        pm.execute_notebook(temp, log, parameters=parameters or {}, kernel_name='python3', cwd=os.path.dirname(os.path.abspath(path)), progress_bar=False)
    except pm.PapermillExecutionError as error:
        if error.ename != 'SystemExit':    # O Format TERMINA EM sys.exit QUANDO NÃO HÁ TILES A CORTAR
            raise
    finally:
        os.remove(temp)


# ESPERA A PLACA TER minimum GiB LIVRES: COM OUTRO TREINO NELA, O CANDIDATO CAIRIA POR FALTA DE MEMÓRIA; AVISA AO COMEÇAR E AO TERMINAR A ESPERA
def waitGpu(minimum, poll=60):
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 2 ** 30

    if free >= minimum:
        return

    print(f"{time.strftime('%d/%m %H:%M')} · a placa tem {free:.1f} GiB livres e o candidato pede {minimum}: esperando o outro job", flush=True)

    while free < minimum:
        time.sleep(poll)
        free = torch.cuda.mem_get_info()[0] / 2 ** 30

    print(f"{time.strftime('%d/%m %H:%M')} · placa livre ({free:.1f} GiB): continuando", flush=True)


# AS BARRAS E OS PRINTS DO GERADOR E DA PREDIÇÃO DOS BLOCOS, QUE EM CENTENAS DE CANDIDATOS PESARIAM O NOTEBOOK
@contextmanager
def quiet():
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


# NOTA DOS BLOCOS REAIS DO Documents/REPORT.html: clDice COM TOLERÂNCIA NO ORÇAMENTO IGUAL, NO F3_wu CONTRA O MODELO DO ARTIGO DO WU E NO MARLIM CONTRA O ESPECIALISTA NA INLINE ANOTADA
class BlockScore:
    TOL     = 3.0       # AMOSTRAS: A LARGURA DE UM TRAÇO
    BORDER  = 16        # PRIMEIROS TRAÇOS DE CADA PATCH, ONDE TODA REDE MARCA A BORDA
    OVERLAP = 0.25      # O DOS NOTEBOOKS DE Blocks/
    TUNING  = ('1200', '1300')
    TEST    = ('1400', '2600')
    F3_WU   = f'{ROOT}/Blocks/F3_wu/files'
    MARLIM  = f'{ROOT}/Blocks/Marlim/files'
    WU      = f'{ROOT}/Documents/Reference/projeto_wu/results'    # LIDO SÓ NO PRIMEIRO setup(): DEPOIS A REFERÊNCIA MORA NO CACHE

    def __init__(self, cache):
        self.cache    = cache
        self.inline   = json.load(open(f'{self.MARLIM}/cache/inline_index.json'))
        self.scores   = {}
        self.sections = {}

    def setup(self):
        if not os.path.exists(f'{self.cache}/reference.npz'):
            self.save()

        data        = np.load(f'{self.cache}/reference.npz')
        self.budget = json.load(open(f'{self.cache}/budget.json'))
        metric      = ClDice(self.TOL)
        self.masks  = {block: data[block] for block in data.files}
        self.refs   = {block: (metric.skeleton(data[block]), metric.near(data[block])) for block in data.files}

    # PREDIZ COM A REDE DE modelPath O F3_wu INTEIRO E A INLINE ANOTADA DE CADA PATCH, SEM GRAVAR NADA
    def update(self, modelPath):
        predictor     = BlockPredictor(modelPath, self.OVERLAP, skipDone=False)
        self.scores   = {}
        self.sections = {}

        if not predictor.load():
            return

        sections = {'f3_wu': self.predict(predictor, f'{self.F3_WU}/patch_metadata.json', f'{self.F3_WU}/patches')}

        for patch in self.TUNING + self.TEST:
            tiles = f'{self.MARLIM}/patches/{patch}/tiles'
            sections[f'marlim_{patch}'] = self.predict(predictor, f'{tiles}/patch_metadata.json', tiles, self.inline[patch])

        predictor.reset()
        self.sections = sections
        self.scores   = {block: self.score(block, probs) for block, probs in sections.items()}

    # O VOLUME (inline None) OU SÓ A INLINE, MONTADOS PELOS NÚCLEOS DOS TILES COMO NO BlockPredictor DOS NOTEBOOKS
    def predict(self, predictor, metaPath, tiles, inline=None):
        grid    = TileGrid(metaPath)
        target  = np.zeros(grid.shape if inline is None else grid.shape[1:], np.float32)
        sliding = SlidingWindow(predictor.window, grid.size, self.OVERLAP)

        for name, probs in predictor.process(sliding, tiles, f'{tiles}/{predictor.modelName}', grid.names(inline)):
            grid.place(target, name, probs, inline)

        return target

    def score(self, block, probs):
        skelRef, nearRef = self.refs[block]
        probs = np.asarray(probs, np.float32)
        limit = ClDice.threshold(ClDice.histogram(probs if block == 'f3_wu' else probs[:, self.BORDER:]), self.budget[block])
        pred  = probs >= limit

        if block != 'f3_wu':
            pred[:, :self.BORDER] = False

        return {**ClDice(self.TOL).update(pred, refSkeleton=skelRef, refNear=nearRef).metrics(), 'limiar': limit}

    def info(self):
        if not self.scores:
            return {}

        cl     = {block: score['cldice'] for block, score in self.scores.items()}
        marlim = lambda patches: float(np.mean([cl[f'marlim_{p}'] for p in patches]))
        return {
            'nota': (cl['f3_wu'] + marlim(self.TUNING + self.TEST)) / 2,
            'nota_ajuste': (cl['f3_wu'] + marlim(self.TUNING)) / 2,
            'marlim_ajuste': marlim(self.TUNING),
            'marlim_prova': marlim(self.TEST),
            **{block: {key: round(value, 4) for key, value in score.items()} for block, score in self.scores.items()}
        }

    # AS PREDIÇÕES QUE OS NOTEBOOKS DE Blocks/ JÁ GRAVARAM NUMA PASTA DE MODELO (f3_wu/ E marlim/patch_*/predicted.npy)
    def load(self, folder):
        sections = {'f3_wu': np.load(f'{folder}/f3_wu/predicted.npy', mmap_mode='r')}
        sections.update({f'marlim_{p}': np.load(f'{folder}/marlim/patch_{p}/predicted.npy', mmap_mode='r')[self.inline[p]] for p in self.TUNING + self.TEST})
        self.sections = sections
        self.scores   = {block: self.score(block, probs) for block, probs in sections.items()}

    # AS REFERÊNCIAS E O ORÇAMENTO DE CADA BLOCO: O MODELO DO ARTIGO DO WU NO 0,5 NO F3_wu; O ESPECIALISTA NO MARLIM, COM A FRAÇÃO QUE O fseg-70 MARCA NA MESMA INLINE
    def save(self):
        os.makedirs(self.cache, exist_ok=True)
        refs   = {'f3_wu': np.load(f'{self.WU}/wu_artigo_2018/f3_wu/predicted.npy', mmap_mode='r') >= 0.5}
        budget = {'f3_wu': float(refs['f3_wu'].mean())}

        for patch in self.TUNING + self.TEST:
            expert = np.array(Image.open(f'{self.MARLIM}/patches/{patch}/{patch}_interpretado.png').convert('L')) > 128
            fseg   = np.load(f'{self.WU}/wu_fseg70/marlim/patch_{patch}/predicted.npy', mmap_mode='r')[self.inline[patch]]
            expert[:, :self.BORDER] = False
            refs[f'marlim_{patch}']   = expert
            budget[f'marlim_{patch}'] = float((fseg[:, self.BORDER:] >= 0.5).mean())

        np.savez_compressed(f'{self.cache}/reference.npz', **refs)
        json.dump(budget, open(f'{self.cache}/budget.json', 'w'), indent=4)


# clDice DE UM DATASET ROTULADO CONTRA O RÓTULO, NO ORÇAMENTO IGUAL DA BlockScore: UM LIMIAR SÓ PARA TODOS OS TILES, NA FRAÇÃO DE FALHA DOS RÓTULOS
class SyntheticScore:
    TOL     = BlockScore.TOL
    OVERLAP = BlockScore.OVERLAP

    def __init__(self, cache, dataset):
        self.cache    = cache
        self.dataset  = dataset
        self.database = pd.read_csv(f'{ROOT}/Dataset/{dataset}/DataBase.csv')
        self.scaling  = Normalization.read(f'{ROOT}/Dataset/{dataset}/DataBase.csv')
        self.path     = f'{cache}/reference_{dataset}_{len(self.database)}.npz'
        self.result   = {}

    def setup(self):
        if not os.path.exists(self.path):
            self.save()

        data          = np.load(self.path)
        self.shape    = tuple(int(n) for n in data['shape'])
        self.fraction = float(data['fraction'])
        self.refs     = list(zip(data['skeleton'], data['near']))

    # PREDIZ COM A REDE DE modelPath TODOS OS TILES DO DATASET (EM PASSOS DE 1/255, PARA CABER NA MEMÓRIA) E MEDE O clDice; NADA É GRAVADO
    def update(self, modelPath):
        predictor   = BlockPredictor(modelPath, self.OVERLAP, skipDone=False)
        trained     = json.load(open(f'{modelPath}/info.json', encoding='utf-8')).get('processing', {}).get('scaling', Normalization.DEFAULT)
        self.result = {}

        if trained != self.scaling:
            raise ValueError(f'{modelPath} treinou com scaling={trained!r} e o {self.dataset} está em {self.scaling!r}')

        if not predictor.load():
            return

        sliding = SlidingWindow(predictor.window, self.shape, self.OVERLAP)
        tiles   = [ClDice.quantize(sliding.process(predictor.network, np.load(path))) for path in self.database.img_path]
        predictor.reset()
        self.result = self.score(tiles)

    # tiles EM uint8 (PASSOS DE 1/255), NA ORDEM DO DataBase.csv
    def score(self, tiles):
        level  = int(round(ClDice.threshold(sum(ClDice.histogram(tile) for tile in tiles), self.fraction) * 255))
        metric = ClDice(self.TOL)

        for tile, (skelRef, nearRef) in zip(tiles, self.refs):
            metric.update(tile >= level, refSkeleton=self.unpack(skelRef), refNear=self.unpack(nearRef))

        return {**metric.metrics(), 'limiar': level / 255, 'tiles': len(tiles)}

    def unpack(self, bits):
        return np.unpackbits(bits)[:int(np.prod(self.shape))].reshape(self.shape).astype(bool)

    def info(self):
        return {key: round(value, 4) if isinstance(value, float) else value for key, value in self.result.items()}

    # O ESQUELETO E A VIZINHANÇA DE CADA RÓTULO EM BITS: OS DO dataset_wu_val OCUPAM ~1 MB
    def save(self):
        os.makedirs(self.cache, exist_ok=True)
        metric   = ClDice(self.TOL)
        masks    = [np.load(path) > 0.5 for path in self.database.mask_path]
        skeleton = np.stack([np.packbits(metric.skeleton(mask)) for mask in masks])
        near     = np.stack([np.packbits(metric.near(mask)) for mask in masks])
        np.savez_compressed(self.path, shape=masks[0].shape, fraction=np.mean([mask.mean() for mask in masks]), skeleton=skeleton, near=near)


# UMA LINHA POR CANDIDATO COMPLETO EM folder: OS TRÊS clDice DO OBJETIVO, O IoU E A AUC NO target, A NOTA NOS BLOCOS, O TEMPO E AS FAIXAS keys DO GERADOR
def getRuns(folder, target, keys):
    rows  = []
    paths = sorted(glob.glob(f'{folder}/model_*/info.json'), key=lambda path: int(re.findall(r'model_(\d+)', path)[-1]))

    for path in paths:
        info = json.load(open(path))

        if 'blocos' not in info or 'sintetico' not in info:    # CANDIDATO INTERROMPIDO NO MEIO: O evaluate() REFAZ
            continue

        if info['processing'].get('synthetic') is None:        # A REFERÊNCIA (DADOS DO WU) É A CALIBRAÇÃO, NÃO UM CANDIDATO
            continue

        test, blocks, synthetic = info['test'][target], info['blocos'], info['processing']['synthetic']
        rows.append({
            'modelo': os.path.basename(os.path.dirname(path)),
            'trial': info['processing']['trial'],
            'wu': info['sintetico']['cldice'],
            'f3_wu': blocks['f3_wu']['cldice'],
            'marlim': blocks['marlim_ajuste'],
            'marlim_prova': blocks['marlim_prova'],
            'nota': blocks['nota'],
            'iou': test['iou'],
            'auc': test.get('auc'),
            'minutos': info.get('minutos'),
            **{key: synthetic[key] for key in keys},
            'synthetic': synthetic
        })

    return pd.DataFrame(rows)
