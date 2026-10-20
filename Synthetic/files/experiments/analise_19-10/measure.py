# MEDIDAS DO DOMÍNIO: DESCONTINUIDADE DE FUNDO (FORA DAS FALHAS) NO REAL E NO SINTÉTICO, FALHAS POR 128 px E ONDE CAI A MÁSCARA DA dbrnet
import os, json, cv2, numpy as np, pandas as pd
import scipy.ndimage as ndi
from PIL import Image
from skimage.morphology import skeletonize
cv2.setNumThreads(4)
R = '/home/grva-mint/Projects/Falhas'
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sections')
WATER, ALIVE = 0.50263, 0.0552

def gauss(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

# SEMBLANCE DE 5 TRAÇOS SEGUINDO O MERGULHO LOCAL (A MESMA DO Regions/Marlim): ~1 NO REFLETOR CONTÍNUO, CAI NA DESCONTINUIDADE
def semblance(sec, half=2, win=4):
    sec = sec.astype(np.float32)
    smooth = gauss(sec, 1.0)
    gz, gx = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    Jzz, Jxx, Jzx = gauss(gz * gz, 4.0), gauss(gx * gx, 4.0), gauss(gz * gx, 4.0)
    dip = np.clip(-np.tan(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)), -3, 3)
    zz, xx = np.mgrid[:sec.shape[0], :sec.shape[1]].astype(np.float32)
    traces = [cv2.remap(sec, xx + k, zz + dip * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for k in range(-half, half + 1)]
    window = np.ones((2 * win + 1, 1), np.float32)
    num = cv2.filter2D(sum(traces) ** 2, -1, window, borderType=cv2.BORDER_REFLECT)
    den = cv2.filter2D(sum(t * t for t in traces) * len(traces), -1, window, borderType=cv2.BORDER_REFLECT)
    return num / (den + 1e-12)

def runs(line):
    return int(np.count_nonzero(np.diff(np.concatenate([[0], line.astype(np.int8), [0]])) == 1))

index = json.load(open(f'{R}/Blocks/Marlim/files/cache/inline_index.json'))
real, place = [], []
for pid in ('1200', '1300', '1400', '2600'):
    sec  = np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
    gt   = np.array(Image.open(f'{R}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    prob = np.load(f'{S}/art5_dbrnet_{pid}.npy').astype(np.float32)
    rock = np.abs(sec - WATER) > 0.02
    floor = np.argmax(rock, axis=0)
    dead = ~rock.any(0)
    zz = np.arange(sec.shape[0])[:, None]
    water = (zz < floor[None, :] + 16) | dead[None, :]
    env = np.sqrt(gauss((sec - WATER) ** 2, 12))
    alive = env > ALIVE
    deadzone = ~alive & ~water
    live = ~water & alive
    disc = semblance(sec - WATER) < 0.8
    dist = ndi.distance_transform_edt(~skeletonize(gt))
    near, far = dist <= 6, dist > 40
    real.append({'patch': pid, 'disc_na_falha': disc[near & live].mean(), 'disc_fundo': disc[far & live].mean(), 'razao': disc[near & live].mean() / disc[far & live].mean(), 'falhas_por_128px': np.mean([runs(gt[z][live[z]]) / max(live[z].sum(), 1) * 128 for z in range(300, 1200, 25)]), 'rotulo_frac_vivo': gt[live].mean()})
    mask = prob >= 0.5
    edge = np.zeros_like(mask); edge[:, :8] = True; edge[:, -8:] = True
    n = mask.sum()
    place.append({'patch': pid, 'máscara/vivo': mask[live].mean(), 'água+traço morto': (mask & water).sum() / n, 'zona morta': (mask & deadzone).sum() / n, 'borda x<8': (mask & edge).sum() / n, '≤12px do especialista': (mask & live & (dist <= 12)).sum() / n, '12-40px': (mask & live & (dist > 12) & (dist <= 40)).sum() / n, '>40px (vivo)': (mask & live & (dist > 40)).sum() / n})

syn = []
for name in ('dataset_74', 'dataset_wu'):
    df = pd.read_csv(f'{R}/Dataset/{name}/DataBase.csv').sample(40, random_state=0)
    acc = {'near': [], 'far': [], 'count': [], 'frac': []}
    for img_path, mask_path in zip(df.img_path, df.mask_path):
        img, msk = np.load(img_path)[64].astype(np.float32), np.load(mask_path)[64] > 0
        disc = semblance(img - np.median(img)) < 0.8
        dist = ndi.distance_transform_edt(~msk)
        inner = np.zeros_like(msk); inner[8:-8, 8:-8] = True
        acc['near'].append(disc[(dist <= 3) & inner].mean() if ((dist <= 3) & inner).any() else np.nan)
        acc['far'].append(disc[(dist > 20) & inner].mean() if ((dist > 20) & inner).any() else np.nan)
        acc['count'].append(np.mean([runs(msk[z]) for z in (32, 64, 96)]))
        acc['frac'].append(msk.mean())
    near, far = np.nanmean(acc['near']), np.nanmean(acc['far'])
    syn.append({'dataset': name, 'disc_na_falha': near, 'disc_fundo': far, 'razao': near / far, 'falhas_por_128px': np.mean(acc['count']), 'rotulo_frac': np.mean(acc['frac'])})

pd.set_option('display.width', 220)
print('MARLIM (inline anotada, intervalo vivo):'); print(pd.DataFrame(real).round(4).to_string(index=False))
print('\nSINTÉTICO (seção central, 40 tiles):'); print(pd.DataFrame(syn).round(4).to_string(index=False))
print('\nONDE CAI A MÁSCARA ≥0.5 DA dbrnet DE REFERÊNCIA (fração da máscara):'); print(pd.DataFrame(place).round(3).to_string(index=False))
