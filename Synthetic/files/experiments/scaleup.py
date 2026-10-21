# ABRE FILAS EXTRAS DA GPU QUANDO A PLACA FICAR OCIOSA: ESPERA A CAMPANHA DO USUÁRIO SAIR (PID), DÁ 15 min PARA A FILA a ESQUENTAR
# E LANÇA A PRÓXIMA FILA (b, c) QUANDO O USO FICAR ABAIXO DE 80% COM MEMÓRIA LIVRE, TENTANDO DE NOVO ENQUANTO HOUVER FILA VIVA. SÓ ABRE, NUNCA MATA NADA.
import os, sys, time, subprocess

EXP    = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
PYTHON = '/home/grva-mint/anaconda3/envs/torch-gpu/bin/python'


def gpu():
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    util, used, total = (int(v) for v in out.strip().split(','))
    return util, total - used


def sample(n=30):
    values = []
    for _ in range(n):
        values.append(gpu())
        time.sleep(2)
    return sum(v[0] for v in values) / n, min(v[1] for v in values)


if __name__ == '__main__':
    foreign, slots = int(sys.argv[1]), sys.argv[2:] or ['b', 'c']
    while os.path.exists(f'/proc/{foreign}'):
        time.sleep(60)
    time.sleep(900)

    # A PLACA PODE ESTAR CHEIA NA HORA (O USUÁRIO USANDO, OU UM JOB QUE JÁ SATURA): TENTA DE NOVO A CADA 15 min ATÉ A FILA ESVAZIAR
    for slot in slots:
        while os.path.exists(f'{EXP}/queue.json') and any(p for p in os.popen("ps -eo args | grep 'gpuqueue.py' | grep -v grep").read().splitlines()):
            util, free = sample()
            print(time.strftime('%d/%m %H:%M'), f'uso médio {util:.0f}%, memória livre {free} MB', flush=True)
            if util < 80 and free >= 8000:
                subprocess.Popen([PYTHON, f'{EXP}/gpuqueue.py', slot], stdout=open(f'{EXP}/logs/gpuqueue_{slot}.log', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
                print(time.strftime('%d/%m %H:%M'), 'fila', slot, 'aberta', flush=True)
                time.sleep(900)
                break
            time.sleep(900)
