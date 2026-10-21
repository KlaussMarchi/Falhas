# SPEC DE DATASET A PARTIR DO VENCEDOR DE UMA CAMPANHA DO fault_ga.py: BASE DE IMAGEM + PARÂMETROS DE FALHA DO VENCEDOR, 220 TILES
# uso: python make_spec.py <tag> <pasta da campanha em Synthetic/files/memory>
import sys, json
from fault_ga import BASES

EXP = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'


if __name__ == '__main__':
    tag, campaign = sys.argv[1], sys.argv[2]
    winner        = json.load(open(f'/home/grva-mint/Projects/Falhas/Synthetic/files/memory/{campaign}/winner.json'))
    specs         = json.load(open(f'{EXP}/datasets/specs.json'))
    specs[tag]    = {'seed': 20261021, 'regions': {'tiles': {'n_images': 220, 'output': 'tiles', 'params': {**BASES[winner['base']], **winner['vencedor']}}}}
    json.dump(specs, open(f'{EXP}/datasets/specs.json', 'w'), indent=1)
    print(tag, json.dumps(specs[tag]['regions']['tiles']['params']))
