# HARNESS DE AVALIAÇÃO NO MARLIM: AS CLASSES SÃO AS DO Marlim/Analysis.ipynb (EXECUTADAS DAS CÉLULAS), A PREDIÇÃO
# GUARDA SÓ A INLINE ANOTADA. MÉTRICAS DE STICKS IDÊNTICAS ÀS DO NOTEBOOK + MÉTRICAS DE DEFEITO DA MÁSCARA
import os, re, sys, json, time
import numpy as np
import torch

ROOT = '/home/grva-mint/Projects/Falhas'
sys.path.insert(0, f'{ROOT}/Model')
from Network.index import ModelNetwork

CUT = (r'^grid = TileGrid', r"^\{'shape': grid", r'^SlidingWindow\(window', r'^annotated = ', r'^pd\.DataFrame\(annotated', r'^PATCH    = ', r'^comparer = ', r'^comparer\.show')


def loadNotebook():
    nb   = json.load(open(f'{ROOT}/Blocks/Marlim/Analysis.ipynb'))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    ns   = {}
    exec(code[0].replace("sys.path.append('../Model')", '').replace('from Network.index import ModelNetwork', ''), ns)
    ns.update({'BASE_PATH': '', 'MARLIM_DIR': f'{ROOT}/Blocks/Marlim/files/patches', 'CACHE_DIR': f'{ROOT}/Blocks/Marlim/files/cache',
               'OVERLAP': 0.25, 'WINDOW': None, 'SKIP_DONE': True, 'STICK_TOL': 0.80, 'STICK_MIN': 140, 'STICK_GAP': 59, 'ModelNetwork': ModelNetwork})

    for src in code:
        if any(k in src for k in ('class TileGrid', 'class SlidingWindow', 'class FaultStickExtractor', 'class FaultComparer')):
            lines = src.split('\n')
            keep  = [l for l in lines if not any(re.match(p, l) for p in CUT)]
            exec('\n'.join(keep), ns)

    return ns


NB      = loadNotebook()
PATCHES = ('1200', '1300', '1400', '2600')
INDEX   = json.load(open(f'{ROOT}/Blocks/Marlim/files/cache/inline_index.json'))


def loadNetwork(modelDir, state=None):
    info    = json.load(open(f'{modelDir}/info.json'))
    options = dict(info['model'])
    network = ModelNetwork(**options)
    # O model_6 DA CAMPANHA MULTISCALE É DA FAULT-SEG-NET ANTIGA (ATENÇÃO ACHATADA, SEM O GARGALO): CARREGA COM A CLASSE DO git HEAD
    if 'legacy' in modelDir or (options['network'] == 'fault_seg_net' and 'Backups/Multiscale' in modelDir):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from FaultSegNetOld import FaultSegNet as Old
        network.model = Old(1, 1, options['num_filters'], options['dropout'], options['img_size']).to(network.device)
    state   = torch.load(state or f'{modelDir}/model.pth', map_location=network.device, weights_only=False)['model']
    network.model.load_state_dict(state)
    network.model.eval()
    return network, info


# PREDIZ A INLINE ANOTADA DO PATCH: CADA TILE PASSA INTEIRO (OU PELA JANELA DO TREINO) E SÓ A FATIA DO NÚCLEO FICA
def predictSection(network, info, pid, transform=None, batch=1):
    grid   = NB['TileGrid'](f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles/patch_metadata.json')
    inline = INDEX[pid]
    window = NB['asTriple'](info['model'].get('img_size'), default=128)
    slide  = NB['SlidingWindow'](window, grid.size, overlap=0.25)
    names  = grid.names(inline)
    lo, hi = grid.ov, [p - o for p, o in zip(grid.size, grid.ov)]
    sec    = np.zeros(grid.shape[1:], np.float32)
    folder = f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles'

    def place(name, probs):
        (d, h, w), (cd, ch, cw) = grid.core(name)
        core = probs[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sec[h:h + ch, w:w + cw] = core[inline - d, :ch, :cw]

    with torch.no_grad():
        if len(slide.positions) == 1:
            for start in range(0, len(names), batch):
                chunk = names[start:start + batch]
                tiles = [NB['getDAT'](f'{folder}/{n}', grid.size) for n in chunk]
                tiles = [transform(t) if transform else t for t in tiles]
                x     = torch.from_numpy(np.stack(tiles)[:, None]).to(network.device)
                probs = torch.sigmoid(network.model(x)).float().cpu().numpy()[:, 0]
                for n, p in zip(chunk, probs):
                    place(n, p)
        else:
            for n in names:
                t = NB['getDAT'](f'{folder}/{n}', grid.size)
                t = transform(t) if transform else t
                place(n, slide.process(network, torch.from_numpy(np.ascontiguousarray(t))))

    return sec


def comparer(pid):
    preds = [{'label': 'interpretado', 'path': f'{ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png', 'color': 'blue'}]
    return NB['FaultComparer'](pid, preds)


def stickMetrics(pid, sec, label='rede'):
    cmp = comparer(pid)
    cmp.preds.append({'label': label, 'path': 'harness', 'color': 'red'})
    cmp.setSection(label, sec)
    m = cmp.metrics(label)
    return {k: v for k, v in m.items() if k != 'modelo'}, cmp


# MÉTRICAS DE DEFEITO DA MÁSCARA (LIMIAR 0.5) NA INLINE ANOTADA
def defectMetrics(pid, sec, tol=12):
    import cv2
    import scipy.ndimage as ndi
    from skimage.morphology import skeletonize
    from PIL import Image

    seismic = np.load(f'{ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
    gt      = np.array(Image.open(f'{ROOT}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    mask    = sec >= 0.5
    ext     = NB['FaultStickExtractor']()
    energy  = ext.energy(seismic)
    alive   = energy > 0.05

    # ORIENTAÇÃO LOCAL DA MÁSCARA (0 = HORIZONTAL, 90 = VERTICAL) PELO TENSOR DE ESTRUTURA DA PROBABILIDADE
    p      = cv2.GaussianBlur(sec.astype(np.float32), (0, 0), 1.5)
    Ix, Iy = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    g      = lambda a: cv2.GaussianBlur(a, (0, 0), 3.0)
    Jxx, Jyy, Jxy = g(Ix * Ix), g(Iy * Iy), g(Ix * Iy)
    lineAngle = (np.degrees(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180.0
    dip       = np.minimum(lineAngle, 180 - lineAngle)

    layer, coh = ext.layering(seismic)
    diff       = np.abs(lineAngle - layer) % 180
    diff       = np.minimum(diff, 180 - diff)

    gtSk   = skeletonize(gt)
    distM  = ndi.distance_transform_edt(~mask)
    distG  = ndi.distance_transform_edt(~gtSk)
    n      = max(mask.sum(), 1)

    lbl, count = ndi.label(mask, structure=np.ones((3, 3)))
    sizes      = ndi.sum(mask, lbl, range(1, count + 1)) if count else np.array([0])
    objs       = ndi.find_objects(lbl)
    extent     = np.array([np.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in objs]) if count else np.array([0])

    zz    = np.arange(sec.shape[0])[:, None] * np.ones((1, sec.shape[1]))
    bands = {'z<450': zz < 450, 'z450-700': (zz >= 450) & (zz < 700), 'z700-950': (zz >= 700) & (zz < 950), 'z950-1200': (zz >= 950) & (zz < 1200), 'z>1200': zz >= 1200}
    cover = {f'cob_{k}': round(float((distM[gtSk & b] <= tol).mean()), 3) if (gtSk & b).sum() > 20 else None for k, b in bands.items()}

    return {
        'mask_frac': round(float(mask[alive].mean()), 4),
        'water_frac': round(float(mask[~alive].mean()), 5),
        'cobertura': round(float((distM[gtSk] <= tol).mean()), 3),
        'perto_gt': round(float((distG[mask] <= tol).mean()), 3),
        'horizontal': round(float((dip[mask] < 25).mean()), 3),
        'no_mergulho': round(float(((dip[mask] >= 35) & (dip[mask] <= 82)).mean()), 3),
        'paralelo_camada': round(float(((diff[mask] < 15) & (coh[mask] > 0.3)).mean()), 3),
        'fragmentos': int(count),
        'frag_curtos': round(float(sizes[extent < 30].sum() / n), 3),
        'borda_x0': round(float(mask[:, :4].mean()), 3),
        **cover,
    }


def evaluate(modelDir, patches=PATCHES, transform=None, save=None, tag=None, state=None):
    network, info = loadNetwork(modelDir, state)
    rows = []

    for pid in patches:
        t0  = time.time()
        sec = predictSection(network, info, pid, transform=transform)
        dt  = time.time() - t0
        sm, _ = stickMetrics(pid, sec)
        dm    = defectMetrics(pid, sec)
        rows.append({'modelo': tag or os.path.basename(modelDir), 'rede': info['model']['network'], 'patch': pid, **sm, **dm, 'seg': round(dt)})

        if save:
            os.makedirs(save, exist_ok=True)
            np.save(f'{save}/{tag or os.path.basename(modelDir)}_{pid}.npy', sec.astype(np.float16))

        print(json.dumps(rows[-1]), flush=True)

    del network
    torch.cuda.empty_cache()
    return rows


# PREDIÇÃO RÁPIDA DA INLINE ANOTADA: MONTA O SLAB DE ENTRADA (128 INLINES REAIS) PELOS NÚCLEOS DOS TILES E PASSA JANELAS DE 128^3 COM
# PASSO stride E PESO DE HANN NO PLANO (z, xline); ~300 JANELAS POR PATCH CONTRA 910 TILES. SÓ PARA COMPARAR SONDAS ENTRE SI
SLABS = {}

def slab(pid):
    if pid in SLABS:
        return SLABS[pid]
    os.makedirs(f'{ROOT}/Synthetic/files/experiments/cache', exist_ok=True)
    cache = f'{ROOT}/Synthetic/files/experiments/cache/slab_{pid}.npy'    # CACHE DESCARTÁVEL: REFEITO SOZINHO SE FOR APAGADO
    if os.path.exists(cache):
        SLABS[pid] = np.load(cache)
        return SLABS[pid]
    grid   = NB['TileGrid'](f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles/patch_metadata.json')
    folder = f'{ROOT}/Blocks/Marlim/files/patches/{pid}/tiles'
    lo, hi = grid.ov, [p - o for p, o in zip(grid.size, grid.ov)]
    vol    = np.zeros((grid.size[0], *grid.shape[1:]), np.float32)
    for name in grid.names():
        (d, h, w), (cd, ch, cw) = grid.core(name)
        tile = NB['getDAT'](f'{folder}/{name}', grid.size)
        vol[:, h:h + ch, w:w + cw] = tile[:, lo[1]:lo[1] + ch, lo[2]:lo[2] + cw]
    np.save(cache, vol)
    SLABS[pid] = vol
    return vol


def predictFast(network, info, pid, stride=112, batch=1):
    vol    = slab(pid)
    inline = INDEX[pid] + 32
    win    = 128
    starts = lambda n: sorted(set(list(range(0, n - win + 1, stride)) + [n - win]))
    wz, wx = np.hanning(win + 2)[1:-1], np.hanning(win + 2)[1:-1]
    weight = np.maximum(wz[:, None] * wx[None, :], 1e-3).astype(np.float32)
    total  = np.zeros(vol.shape[1:], np.float32)
    norm   = np.zeros(vol.shape[1:], np.float32)
    pos    = [(z, x) for z in starts(vol.shape[1]) for x in starts(vol.shape[2])]
    with torch.no_grad():
        for i in range(0, len(pos), batch):
            chunk = pos[i:i + batch]
            x = torch.from_numpy(np.stack([vol[:, z:z + win, s:s + win] for z, s in chunk])[:, None]).to(network.device)
            p = torch.sigmoid(network.model(x)).float().cpu().numpy()[:, 0, inline]
            for (z, s), q in zip(chunk, p):
                total[z:z + win, s:s + win] += q * weight
                norm[z:z + win, s:s + win] += weight
    return total / norm


def evaluateFast(modelDir, patches=PATCHES, save=None, tag=None, state=None):
    network, info = loadNetwork(modelDir, state)
    rows = []
    for pid in patches:
        t0  = time.time()
        sec = predictFast(network, info, pid)
        dt  = time.time() - t0
        sm, _ = stickMetrics(pid, sec)
        dm    = defectMetrics(pid, sec)
        rows.append({'modelo': tag or os.path.basename(modelDir), 'rede': info['model']['network'], 'patch': pid, **sm, **dm, 'seg': round(dt)})
        if save:
            os.makedirs(save, exist_ok=True)
            np.save(f'{save}/{tag or os.path.basename(modelDir)}_{pid}.npy', sec.astype(np.float16))
        print(json.dumps(rows[-1]), flush=True)
    del network
    torch.cuda.empty_cache()
    return rows


# MESMA PREDIÇÃO RÁPIDA, MÉDIA SOBRE ESPELHAMENTOS DA JANELA (0 = inline, 2 = xline); A SAÍDA VOLTA DESESPELHADA ANTES DA SOMA
def predictFastTTA(network, info, pid, flips=((), (0,), (2,), (0, 2)), stride=112):
    vol    = slab(pid)
    inline = INDEX[pid] + 32
    win    = 128
    starts = lambda n: sorted(set(list(range(0, n - win + 1, stride)) + [n - win]))
    wz     = np.hanning(win + 2)[1:-1]
    weight = np.maximum(wz[:, None] * wz[None, :], 1e-3).astype(np.float32)
    total  = np.zeros(vol.shape[1:], np.float32)
    norm   = np.zeros(vol.shape[1:], np.float32)
    with torch.no_grad():
        for z in starts(vol.shape[1]):
            for s in starts(vol.shape[2]):
                x = torch.from_numpy(np.ascontiguousarray(vol[:, z:z + win, s:s + win]))[None, None].to(network.device)
                acc = 0
                for axes in flips:
                    dims = tuple(a + 2 for a in axes)
                    xi   = torch.flip(x, dims) if dims else x
                    p    = torch.sigmoid(network.model(xi))
                    acc  = acc + (torch.flip(p, dims) if dims else p)
                q = (acc / len(flips)).float().cpu().numpy()[0, 0, inline]
                total[z:z + win, s:s + win] += q * weight
                norm[z:z + win, s:s + win] += weight
    return total / norm


# SEÇÃO VIVA DA INLINE ANOTADA (ENERGIA LOCAL > 0.05), A MESMA DO defectMetrics
LIVE = {}

def liveMask(pid):
    if pid not in LIVE:
        seismic   = np.load(f'{ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
        LIVE[pid] = NB['FaultStickExtractor']().energy(seismic) > 0.05
    return LIVE[pid]


# MÁSCARA IGUALADA: p^k COM k TAL QUE p^k >= 0.5 MARQUE frac DA SEÇÃO VIVA; O F1 PREMIA MARCAR POUCO, ENTÃO A COMPARAÇÃO
# ENTRE MODELOS DE CALIBRAÇÃO DIFERENTE SE FAZ AQUI TAMBÉM
def equalize(pid, sec, frac=0.01):
    t = float(np.quantile(sec[liveMask(pid)], 1 - frac))
    t = min(max(t, 1e-6), 1 - 1e-6)
    k = float(np.log(0.5) / np.log(t))
    return np.power(np.clip(sec, 0, 1), k).astype(np.float32), k


# TODAS AS MEDIDAS DE UMA SEÇÃO: STICKS NO LIMIAR NATIVO, DEFEITOS DA MÁSCARA E OS STICKS COM A MÁSCARA IGUALADA EM 0.5/1/2%
def sectionMetrics(pid, sec, fracs=(0.005, 0.01, 0.02)):
    sm, _ = stickMetrics(pid, sec)
    row   = {**sm, **defectMetrics(pid, sec)}

    for frac in fracs:
        eq, k = equalize(pid, sec, frac)
        m, _  = stickMetrics(pid, eq)
        tag   = f'{frac * 100:g}'
        row.update({f'f1@{tag}': m['f1'], f'recall@{tag}': m['recall'], f'precisao@{tag}': m['precisao'], f'deteccao@{tag}': m['deteccao'], f'k@{tag}': round(k, 3)})

    return row


# AVALIA UM MODELO NOS PATCHES PEDIDOS (fast = SLAB COM JANELAS DE PASSO 112; senão O MÉTODO DO NOTEBOOK) E ACRESCENTA AO results.jsonl
def run(modelDir, patches, tag, fast=True, results=None, save=None, extra=None):
    network, info = loadNetwork(modelDir)
    rows = []

    for pid in patches:
        t0  = time.time()
        sec = predictFast(network, info, pid) if fast else predictSection(network, info, pid)
        dt  = time.time() - t0
        row = {'tag': tag, 'modelo': modelDir, 'rede': info['model']['network'], 'patch': pid, 'metodo': 'rapido' if fast else 'notebook', **(extra or {}), **sectionMetrics(pid, sec), 'seg': round(dt)}
        rows.append(row)

        if save:
            os.makedirs(save, exist_ok=True)
            np.save(f'{save}/{tag}_{pid}{"_fast" if fast else ""}.npy', sec.astype(np.float16))

        if results:
            with open(results, 'a') as file:
                file.write(json.dumps(row) + '\n')

        print(json.dumps({k: row[k] for k in ('tag', 'patch', 'metodo', 'f1', 'recall', 'precisao', 'mask_frac', 'f1@1', 'seg')}), flush=True)

    del network
    torch.cuda.empty_cache()
    return rows
