# RESUMO DO results.jsonl: MÉDIA DOS PATCHES POR tag E MÉTODO, COM O F1 NO LIMIAR NATIVO, COM A MÁSCARA IGUALADA E OS DEFEITOS
# uso: python summary.py [filtro-de-tag]
import sys, json
import pandas as pd

EXP  = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
COLS = ['f1', 'recall', 'precisao', 'deteccao', 'f1@0.5', 'f1@1', 'f1@2', 'precisao@1', 'recall@1', 'mask_frac', 'perto_gt', 'frag_curtos', 'horizontal', 'borda_x0', 'cob_z700-950', 'cob_z950-1200']


def load():
    rows = [json.loads(l) for l in open(f'{EXP}/results.jsonl')]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load()
    if len(sys.argv) > 1:
        df = df[df.tag.str.contains(sys.argv[1])]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    keys = ['tag', 'metodo'] + [k for k in ('dataset', 'epochs', 'trial') if k in df]
    df   = df.copy()
    marl = df[~df.metodo.isin(['bateria', 'injecao'])]
    mean = marl.groupby(keys, dropna=False)[COLS + ['patch']].agg({**{c: 'mean' for c in COLS}, 'patch': lambda p: ','.join(sorted(set(p)))})
    print(mean.round(3).to_string())
    print()
    print(marl.pivot_table(index=['tag', 'metodo'], columns='patch', values=['f1', 'f1@1']).round(3).to_string())
    bat = df[df.metodo == 'bateria'].dropna(axis=1, how='all')
    if len(bat):
        print()
        print(bat.drop(columns=['modelo', 'metodo']).set_index('tag').round(4).to_string())
