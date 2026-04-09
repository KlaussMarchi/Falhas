# conda activate NOME_ENV
# pip install papermill
# conda install ipykernel
# python -m ipykernel install --user --name=NOME_ENV

import papermill as pm
from pathlib import Path
import os, json


def execute(path):
    p = Path(path)
    dir_path, name, ext = p.parent, p.stem, p.suffix
    print('etapa: ', name)
    
    os.makedirs('logs', exist_ok=True)
    out = os.path.join('logs', f'{name}_out{ext}')    
    
    try:
        pm.execute_notebook(path, out, kernel_name='torch-gpu', log_output=True, progress_bar=True, cwd=str(dir_path))
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
    execute("../Dataset/Format.ipynb")
    execute("../Model/Analysis.ipynb")
