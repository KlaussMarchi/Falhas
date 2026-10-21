# PERÍODO MEDIDO PELA RÉGUA EM FUNÇÃO DA FREQUÊNCIA NOMINAL DO RICKER (A PARTE "MEDIDA" DA CALIBRAÇÃO, SEM AG): PARA CADA f FIXO,
# N TILES DO C0_v2 SEM COLUNA (fast) E O PERÍODO DA RÉGUA. GRAVA profile/period_map.csv
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibration
from ruler import TileRuler

FREQS = [16, 20, 26, 33, 40, 50, 65, 80, 100, 130, 160, 208, 260]
N     = 8

if __name__ == '__main__':
    spec  = json.load(open(f'{calibration.EXP}/datasets/specs.json'))['C0_v2']['regions']['tiles']['params']
    spec  = {**spec, 'seafloorProb': 0.0, 'deadZoneProb': 0.0}
    ruler = TileRuler()
    rows  = []
    for f in FREQS:
        for s in range(N):
            image, _ = calibration.getTile({**spec, 'waveletFreq': (f, f)}, 5000 + 100 * f + s)
            rows.append({'f': f, 'semente': s, 'period': ruler.get(image).get('period', np.nan)})
        print(f, np.round(np.nanpercentile([r['period'] for r in rows if r['f'] == f], [10, 50, 90]), 1), sum(np.isnan(r['period']) for r in rows if r['f'] == f), flush=True)
    pd.DataFrame(rows).to_csv(f'{calibration.EXP}/profile/period_map.csv', index=False)
