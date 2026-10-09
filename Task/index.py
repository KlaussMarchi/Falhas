import papermill as pm
from pathlib import Path
import os, json, sys

sys.path.append('..')
from Dataset.index import Normalization

 
def execute(path):
    p = Path(path)
    dir_path, name, ext = p.parent, p.stem, p.suffix
    print('etapa: ', name)
    
    os.makedirs('logs', exist_ok=True)
    out = os.path.join('logs', f'{name}_out{ext}')    
    
    try:
        pm.execute_notebook(path, out, kernel_name='python3', log_output=True, progress_bar=True, cwd=str(dir_path))
    except Exception as e:
        print(f'Error executing {path}: {e}')


with open('task.json', 'r') as file:
    tasks = json.load(file)

for i, task in enumerate(tasks):
    print(f'\n\nRodada {i+1}/{len(tasks)}')

    with open('info.json', 'w') as file:
        file.write(json.dumps(task))

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
    
    for trial in range(n_trials):
        print(f'\nTrial {trial+1}/{n_trials}')

        with open('info.json', 'w') as file:
            file.write(json.dumps({**task, 'trial': trial}))

        execute("../Model/1 - Model.ipynb")
