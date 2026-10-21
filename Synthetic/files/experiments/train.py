# TREINO PRIVADO: CÓPIA DO Model/1 - Model.ipynb COM OS CAMINHOS REDIRECIONADOS (info.json, DataBase.csv, progress.json, Backup/)
# PARA models/<run>/, EXECUTADA POR PAPERMILL COM cwd=Model/. NUNCA ESCREVE Task/info.json NEM Model/Backup.
# uso: python train.py <run>   (models/<run>/config.json = {'info': {...}, 'database': caminho do DataBase.csv})
import os, sys, json, glob, time
import nbformat
import papermill as pm

ROOT  = '/home/grva-mint/Projects/Falhas'
EXP   = f'{ROOT}/Synthetic/files/experiments'
SRC   = f'{ROOT}/Model/1 - Model.ipynb'


def patch(run, database):
    swaps = [
        ("open('../Task/info.json', 'r')", f"open('{run}/info.json', 'r')"),
        ("DATABASE = f'../Dataset/{OPTIONS.get(\"dataset\")}/DataBase.csv'", f"DATABASE = '{database}'"),
        ("with open(f'progress.json', 'w', encoding='utf-8')", f"with open(f'{run}/progress.json', 'w', encoding='utf-8')"),
        ("os.makedirs('Backup/', exist_ok=True)", f"os.makedirs('{run}/', exist_ok=True)"),
        ("os.listdir('Backup/')", f"os.listdir('{run}/')"),
        ("folder  = f'Backup/model_{index}'", f"folder  = f'{run}/model_{{index}}'"),
    ]
    nb    = nbformat.read(SRC, as_version=4)
    count = {old: 0 for old, _ in swaps}

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for old, new in swaps:
            count[old] += cell.source.count(old)
            cell.source = cell.source.replace(old, new)

    missing = [old for old, n in count.items() if n != 1]
    if missing:
        raise ValueError(f'trechos não encontrados exatamente uma vez no notebook: {missing}')

    return nb


def main(name):
    run    = f'{EXP}/models/{name}'
    config = json.load(open(f'{run}/config.json'))

    if glob.glob(f'{run}/model_*/model.pth'):
        return print(name, 'já treinado')

    json.dump(config['info'], open(f'{run}/info.json', 'w'))
    nb = patch(run, config['database'])
    nbformat.write(nb, f'{run}/train.ipynb')

    t0 = time.time()
    pm.execute_notebook(f'{run}/train.ipynb', f'{run}/train_out.ipynb', kernel_name='python3', cwd=f'{ROOT}/Model', progress_bar=False, log_output=False)
    print(name, f'treinado em {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        main(name)
