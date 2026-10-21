# ESTADO DE TUDO NUMA OLHADA (PARA RETOMAR DEPOIS DE UMA COMPACTAÇÃO OU NUMA SESSÃO NOVA): FILA DA GPU, GERAÇÃO NA CPU, PLACA
import os, json, glob, subprocess

EXP = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
jobs = json.load(open(f'{EXP}/queue.json'))
done = {os.path.basename(p) for p in glob.glob(f'{EXP}/queue_done/*')}
run  = {os.path.basename(p) for p in glob.glob(f'{EXP}/queue_running/*')}
hold = [j['id'] for j in jobs if j.get('hold')]
todo = [j['id'] for j in jobs if j['id'] not in done | run and not j.get('hold')]
print(f'fila: {len(done)} feitos, rodando {sorted(run)}, {len(todo)} pendentes, parados {hold}')
print('próximos:', todo[:6])
print('datasets prontos:', sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{EXP}/datasets/*/DataBase.csv')))
gen = subprocess.run("ps -eo etime,args | grep -E '^ *[0-9:-]+ +[^ ]*bin/python make_dataset.py' | grep -v grep", shell=True, capture_output=True, text=True).stdout.strip()
print('gerando:', gen or '-')
print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
