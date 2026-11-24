import papermill as pm
import nbformat
from pathlib import Path
import os, re, json, sys

sys.path.append('..')
from Dataset.index import Normalization


def execute(path, parameters=None, cwd=None):
    p = Path(path)
    dir_path, name, ext = p.parent, p.stem, p.suffix
    print('etapa: ', name)

    os.makedirs('logs', exist_ok=True)
    out = os.path.join('logs', f'{name}_out{ext}')

    try:
        pm.execute_notebook(path, out, parameters=parameters or {}, kernel_name='python3', log_output=True, progress_bar=True, cwd=str(cwd or dir_path))
    except Exception as e:
        print(f'Error executing {path}: {e}')


# OS model_N DE Model/Backup, PARA SABER QUAIS A RODADA GRAVOU
def getModels():
    return {name for name in os.listdir('../Model/Backup') if re.fullmatch(r'model_\d+', name)} if os.path.isdir('../Model/Backup') else set()


# O 3 - Predict SÓ NOS MODELOS DA RODADA: A MATRIZ DE CONFUSÃO, A ACURÁCIA, A ROC E AS OUTRAS MÉTRICAS DE TESTE VÃO PARA O test.<dataset> DO info.json
# DE CADA UM. NUMA CÓPIA, A CÉLULA DE CONFIGURAÇÃO (A QUE COMEÇA EM BASE_PATH) VIRA A DE PARÂMETROS, E O models, QUE ELA CALCULA DO MODELS, VAI JUNTO
def predict(models):
    nb     = nbformat.read('../Model/3 - Predict.ipynb', as_version=4)
    config = next(cell for cell in nb.cells if cell.cell_type == 'code' and cell.source.startswith('BASE_PATH'))
    config.metadata['tags'] = ['parameters']

    os.makedirs('logs', exist_ok=True)
    nbformat.write(nb, 'logs/3 - Predict.ipynb')
    execute('logs/3 - Predict.ipynb', {'MODELS': models, 'models': models}, cwd='../Model')
    os.remove('logs/3 - Predict.ipynb')


with open('task.json', 'r') as file:
    tasks = json.load(file)

for i, task in enumerate(tasks):
    print(f'\n\nRodada {i+1}/{len(tasks)}')

    with open('info.json', 'w') as file:
        file.write(json.dumps(task, ensure_ascii=False, indent=4))

    with open('info.json', 'r') as file:
        info = json.load(file)

    print('info: ', info)
    dataset  = info.get('dataset')
    database = f"../Dataset/{dataset}/DataBase.csv"
    print(dataset)

    # O Format SÓ É PULADO QUANDO O DataBase.csv JÁ ESTÁ NO ESCALONAMENTO QUE A RODADA PEDE (SEM A COLUNA, É O MIN-MAX)
    formatted = os.path.exists(database) and Normalization.read(database) == info.get('scaling', Normalization.DEFAULT)
    n_trials  = int(info.get('n_trials') or 1)

    if formatted and info.get('img_size') is None:
        print(f'Format pulado: {database} ja existe')
    else:
        execute(f"../Dataset/{dataset}/Format.ipynb")

    before = getModels()

    for trial in range(n_trials):
        print(f'\nTrial {trial+1}/{n_trials}')

        with open('info.json', 'w') as file:
            file.write(json.dumps({**task, 'trial': trial}, ensure_ascii=False, indent=4))

        execute("../Model/1 - Model.ipynb")

    trained = sorted(getModels() - before, key=lambda name: int(name.split('_')[1]))
    if trained:
        predict(trained)
