# CONTINUIDADE LATERAL DOS REFLETORES: SEMBLANCE EM ABERTURA LONGA (17 TRAÇOS) E VARIAÇÃO DA AMPLITUDE AO LONGO DO MERGULHO, LONGE DAS FALHAS
import json, cv2, numpy as np, pandas as pd
import scipy.ndimage as ndi
from scipy.signal import hilbert
from PIL import Image
from skimage.morphology import skeletonize
cv2.setNumThreads(4)
R = '/home/grva-mint/Projects/Falhas'
WATER, ALIVE = 0.50263, 0.0552

def gauss(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def steer(sec):
    smooth = gauss(sec, 1.0)
    gz, gx = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    Jzz, Jxx, Jzx = gauss(gz * gz, 4.0), gauss(gx * gx, 4.0), gauss(gz * gx, 4.0)
    return np.clip(-np.tan(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)), -3, 3)

# TRAÇOS VIZINHOS SEGUINDO O MERGULHO: SEMBLANCE (CONTINUIDADE DA FORMA) E CV DO ENVELOPE (CONSTÂNCIA DA AMPLITUDE AO LONGO DO REFLETOR)
def lateral(sec, half):
    sec = sec.astype(np.float32)
    dip = steer(sec)
    env = np.abs(hilbert(sec, axis=0)).astype(np.float32)
    zz, xx = np.mgrid[:sec.shape[0], :sec.shape[1]].astype(np.float32)
    traces = [cv2.remap(sec, xx + k, zz + dip * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for k in range(-half, half + 1)]
    envs   = np.stack([cv2.remap(env, xx + k, zz + dip * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for k in range(-half, half + 1)])
    window = np.ones((9, 1), np.float32)
    num = cv2.filter2D(sum(traces) ** 2, -1, window, borderType=cv2.BORDER_REFLECT)
    den = cv2.filter2D(sum(t * t for t in traces) * len(traces), -1, window, borderType=cv2.BORDER_REFLECT)
    return num / (den + 1e-12), envs.std(0) / (envs.mean(0) + 1e-6), env

index = json.load(open(f'{R}/Blocks/Marlim/files/cache/inline_index.json'))
rows = []
for pid in ('1200', '1300', '1400', '2600'):
    sec = np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) - WATER
    gt  = np.array(Image.open(f'{R}/Blocks/Marlim/files/patches/{pid}/{pid}_interpretado.png').convert('L')) > 128
    rock = np.abs(sec) > 0.02
    floor, deadtr = np.argmax(rock, axis=0), ~rock.any(0)
    zz = np.arange(sec.shape[0])[:, None]
    water = (zz < floor[None, :] + 16) | deadtr[None, :]
    alive = np.sqrt(gauss(sec ** 2, 12)) > ALIVE
    live, deadzone = ~water & alive, ~water & ~alive
    sk = skeletonize(gt)
    far = ndi.distance_transform_edt(~sk) > 40
    s17, cv17, env = lateral(sec, 8)
    strong = env > np.percentile(env[live], 70)
    rows.append({'dado': f'Marlim {pid}', 'disc17_fundo': (s17 < 0.8)[far & live].mean(), 'cv_amp_refletor': np.median(cv17[far & live & strong]), 'anotacao_na_zona_morta': sk[deadzone].sum() / sk.sum()})

for name in ('dataset_74', 'dataset_wu'):
    df = pd.read_csv(f'{R}/Dataset/{name}/DataBase.csv').sample(30, random_state=0)
    d17, cvs = [], []
    for img_path, mask_path in zip(df.img_path, df.mask_path):
        img, msk = np.load(img_path)[64].astype(np.float32), np.load(mask_path)[64] > 0
        img -= np.median(img)
        s17, cv17, env = lateral(img, 8)
        far = ndi.distance_transform_edt(~msk) > 20
        inner = np.zeros_like(msk); inner[12:-12, 12:-12] = True
        strong = env > np.percentile(env[inner], 70)
        d17.append((s17 < 0.8)[far & inner].mean()); cvs.append(np.median(cv17[far & inner & strong]))
    rows.append({'dado': name, 'disc17_fundo': np.nanmean(d17), 'cv_amp_refletor': np.nanmedian(cvs), 'anotacao_na_zona_morta': np.nan})

print(pd.DataFrame(rows).round(4).to_string(index=False))
