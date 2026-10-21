# ESCOLHA HONESTA ENTRE CANDIDATOS (LEAVE-ONE-PATCH-OUT): PARA CADA PATCH, O MELHOR PELA MÉDIA DOS OUTROS 3 E A NOTA DELE NO
# PATCH DEIXADO DE FORA. A MÉDIA DESSAS NOTAS É O F1 ESPERADO DO PROCEDIMENTO DE ESCOLHA, SEM O OTIMISMO DE ESCOLHER E MEDIR NOS
# MESMOS PATCHES. uso: python lopo.py <tag> [<tag> ...]   (METRIC=f1@1 NO AMBIENTE TROCA A MÉTRICA)
import os, sys, json
import pandas as pd

EXP = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'


def table(tags, metric):
    rows = pd.DataFrame([json.loads(line) for line in open(f'{EXP}/results.jsonl')])
    rows = rows[(rows.metodo == 'rapido') & rows.tag.isin(tags)]
    return rows.assign(patch=rows.patch.astype(str)).pivot_table(index='tag', columns='patch', values=metric, aggfunc='last').dropna()


def lopo(scores):
    rows = []
    for patch in scores.columns:
        rest   = scores.drop(columns=patch).mean(axis=1)
        chosen = rest.idxmax()
        rows.append({'fora': patch, 'escolhido': chosen, 'media_dos_outros': rest[chosen], 'nota_fora': scores.loc[chosen, patch]})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    scores = table(sys.argv[1:], os.environ.get('METRIC', 'f1'))
    folds  = lopo(scores)
    pd.set_option('display.width', 200)
    print(scores.assign(media=scores.mean(axis=1)).sort_values('media', ascending=False).round(3).to_string())
    print()
    print(folds.round(3).to_string(index=False))
    print(f'\nF1 deixando um patch de fora: {folds.nota_fora.mean():.3f}   melhor média nos 4 (otimista): {scores.mean(axis=1).max():.3f}')
