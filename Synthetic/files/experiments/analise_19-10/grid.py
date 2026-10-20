# COMPOSIÇÃO DA SEÇÃO AVALIADA EM CÉLULAS 64x64 (O NÚCLEO DE CADA TILE): ÁGUA, ZONA MORTA, VIVO; E O TOPO DA ZONA MORTA
import cv2, numpy as np, pandas as pd
R = '/home/grva-mint/Projects/Falhas'
WATER, ALIVE = 0.50263, 0.0552
gauss = lambda img, s: cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
rows = []
for pid in ('1200', '1300', '1400', '2600'):
    sec = np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) - WATER
    rock = np.abs(sec) > 0.02
    floor, deadtr = np.argmax(rock, axis=0), ~rock.any(0)
    zz = np.arange(sec.shape[0])[:, None]
    water = (zz < floor[None, :] + 16) | deadtr[None, :]
    alive = np.sqrt(gauss(sec ** 2, 12)) > ALIVE
    live = ~water & alive
    top = sec.shape[0] - np.argmax(alive[::-1], axis=0)
    cls = []
    for z0 in range(0, 1601 - 63, 64):
        for x0 in range(0, 2240 - 63, 64):
            w, l = water[z0:z0 + 64, x0:x0 + 64].mean(), live[z0:z0 + 64, x0:x0 + 64].mean()
            cls.append('água' if w > 0.5 else ('vivo' if l > 0.5 else 'zona morta'))
    c = pd.Series(cls).value_counts(normalize=True)
    rows.append({'patch': pid, 'células água': c.get('água', 0), 'células zona morta': c.get('zona morta', 0), 'células vivas': c.get('vivo', 0), 'fundo do mar z (p50)': int(np.median(floor[~deadtr])), 'topo zona morta z (p50)': int(np.median(top[~deadtr]))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
