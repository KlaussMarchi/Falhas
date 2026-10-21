# STICKS SEM RÓTULO DA MÁSCARA IGUALADA EM 1% (Nº, COMPRIMENTO, RENDIMENTO, MERGULHO), DAS SEÇÕES SALVAS -> profile/e2plus.csv
import os, sys, glob, re
import numpy as np, pandas as pd
import scipy.ndimage as ndi
from multiprocessing import get_context

EXP = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
sys.path.insert(0, f'{EXP}/harness')
OUT = f'{EXP}/profile/e2plus.csv'


def measure(path):
    import marlim
    tag, pid = re.match(r'(.+)_(\d{4})_fast\.npy', os.path.basename(path)).groups()
    sec      = np.load(path).astype(np.float32)
    seismic  = np.load(f'{marlim.ROOT}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32)
    eq, _    = marlim.equalize(pid, sec, 0.01)
    sticks   = marlim.NB['FaultStickExtractor'](0.80, 140, 59).extract(eq, seismic=seismic)
    mask     = eq >= 0.5
    canvas   = np.zeros(sec.shape, np.uint8)
    import cv2
    for s in sticks:
        cv2.line(canvas, tuple(int(round(v)) for v in s['p0']), tuple(int(round(v)) for v in s['p1']), 1, 1)
    near     = ndi.distance_transform_edt(canvas == 0) <= 3
    lengths  = np.array([s['len'] for s in sticks]) if sticks else np.zeros(0)
    dips     = np.array([np.degrees(np.arctan2(abs(s['p1'][1] - s['p0'][1]), abs(s['p1'][0] - s['p0'][0]) + 1e-9)) for s in sticks]) if sticks else np.zeros(0)
    return {'tag': tag, 'patch': pid, 'e2_sticks': len(sticks), 'e2_len_total': float(lengths.sum()), 'e2_len_p50': float(np.median(lengths)) if len(lengths) else 0.0,
            'e2_rendimento': float(near[mask].mean()) if mask.any() else 0.0, 'e2_dip_p50': float(np.median(dips)) if len(dips) else np.nan, 'e2_ingremes': float((dips >= 40).mean()) if len(dips) else np.nan}


if __name__ == '__main__':
    done  = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['tag', 'patch'])
    seen  = set(zip(done.tag, done.patch.astype(str)))
    paths = [p for p in sorted(glob.glob(f'{EXP}/sections/*_fast.npy')) if tuple(re.match(r'(.+)_(\d{4})_fast\.npy', os.path.basename(p)).groups()) not in seen]
    with get_context('fork').Pool(6) as pool:
        rows = pool.map(measure, paths)
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(OUT, index=False)
    print(table.groupby('tag')[[c for c in table if c.startswith('e2_')]].mean().round(3).to_string())
