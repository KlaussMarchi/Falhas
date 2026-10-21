# CMA-ES DO PERFIL DAS FALHAS: AJUSTA REJEITO/PERÍODO, VISIBILIDADE DO RÓTULO, ARRASTO E DENSIDADE SOBRE UMA BASE DE IMAGEM FIXA
# PARA AS FALHAS ROTULADAS TEREM O PERFIL MEDIDO NO BLOCO. uso: python fault_ga.py <base> <fração rotulada> [alvo]
import sys, json
import pandas as pd
from calibration import Calibration, FaultObjective, EXP

BASES     = {'74': {}, 'C1': json.load(open(f'{EXP}/datasets/specs.json'))['C1_stdDensity']['regions']['tiles']['params']}
REFERENCE = {'faultThrowPeriod': (0.2, 1.5), 'faultVisible': 0.15, 'faultDrag': (0.0, 0.8), 'faultCount': (2, 6)}
LIMITS    = {'faultThrowPeriod': (0.1, 3.0), 'faultVisible': (0.0, 0.5), 'faultDrag': (0.0, 1.0), 'faultCount': (1, 12)}


if __name__ == '__main__':
    base, fraction = sys.argv[1], float(sys.argv[2])
    target         = pd.read_pickle(f'{EXP}/profile/faults_{sys.argv[3] if len(sys.argv) > 3 else "modelo_anc_art5_74wu"}.pkl')
    Calibration.POPULATION, Calibration.GENERATIONS, Calibration.SPREAD = 8, 12, 0.5
    calibration = Calibration(FaultObjective(target, fraction, nTiles=24), REFERENCE, LIMITS, fixed=BASES[base], name=f'falhas_{base}')
    calibration.update()
    summary = calibration.evaluate()
    print(summary.round(3).to_string())
    print(json.dumps({'base': base, 'fração': fraction, 'vencedor': calibration.winner}, default=list))
    json.dump({'base': base, 'fracao': fraction, 'vencedor': calibration.winner, 'resumo': summary.round(4).to_dict()}, open(f'{calibration.memory}/winner.json', 'w'), indent=4, default=list)
