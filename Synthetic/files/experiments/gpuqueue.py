# FILA DA GPU (gpuqueue.py: O NOME queue.py SOMBREARIA O MÓDULO queue DA BIBLIOTECA PADRÃO): RELÊ queue.json DEPOIS DE CADA
# JOB, PULA OS FEITOS (queue_done/<id>) E OS QUE OUTRA FILA PEGOU (queue_running/<id>, CRIADO COM O_EXCL), ESPERA A PLACA FICAR
# LIVRE DE PROCESSO ALHEIO (QUEM NÃO DESCENDE DE UMA gpuqueue) E RODA CADA JOB NUM SUBPROCESSO (job.py). DUAS FILAS = DOIS JOBS
# AO MESMO TEMPO NA PLACA. RETOMÁVEL: É SÓ RELANÇAR (APAGAR O queue_running/<id> DE UM JOB QUE MORREU NO MEIO).
import os, sys, json, time, subprocess

EXP        = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
PYTHON     = '/home/grva-mint/anaconda3/envs/torch-gpu/bin/python'
FOREIGN_MB = 2000    # PROCESSO DE OUTRO DONO USANDO MAIS QUE ISSO NA PLACA...
ACTIVE     = 10      # ...E COM ESTE USO DE SM (%) EM ALGUMA AMOSTRA DE 5 s = PLACA OCUPADA; KERNEL OCIOSO SEGURANDO MEMÓRIA NÃO TRAVA
MARGIN     = 1500    # MB QUE SOBRAM NA PLACA DEPOIS DE UM JOB NOVO ENTRAR


def ancestry(pid):
    chain = []
    while pid > 1:
        try:
            chain.append(open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore'))
            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return chain


def busy():
    out  = subprocess.run(['nvidia-smi', 'pmon', '-c', '5', '-s', 'um'], capture_output=True, text=True).stdout
    load = {}
    for cols in (line.split() for line in out.splitlines() if not line.startswith('#')):
        if len(cols) >= 10 and cols[1].isdigit():
            sm, mb = (int(v) if v.isdigit() else 0 for v in (cols[3], cols[9]))
            load[int(cols[1])] = tuple(max(a, b) for a, b in zip(load.get(int(cols[1]), (0, 0)), (sm, mb)))
    return [pid for pid, (sm, mb) in load.items() if mb > FOREIGN_MB and sm >= ACTIVE and not any('gpuqueue.py' in c for c in ancestry(pid))]


# MEMÓRIA QUE UM JOB PEDE NA PLACA (MB): TREINO nf 16 MEDIDO EM 8,7 GB (21/10), nf 32 ~15 GB, AVALIAÇÃO 3,2 GB; O CAMPO need_mb DO JOB MANDA
def need(job):
    if 'need_mb' in job:
        return job['need_mb']
    return (15000 if job['info'].get('num_filters', 32) >= 32 else 9500) if job['kind'] == 'train' else 3800


def free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def pending():
    jobs = json.load(open(f'{EXP}/queue.json'))
    return [job for job in jobs if not job.get('hold') and not any(os.path.exists(f'{EXP}/{d}/{job["id"]}') for d in ('queue_done', 'queue_running'))]


def claim(jobId):
    try:
        os.close(os.open(f'{EXP}/queue_running/{jobId}', os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def hold(jobId, reason):
    jobs = json.load(open(f'{EXP}/queue.json'))
    for job in jobs:
        if job['id'] == jobId:
            job['hold'] = reason
    json.dump(jobs, open(f'{EXP}/queue.json', 'w'), indent=1)


# after = PID DE UMA CAMPANHA DO USUÁRIO QUE A FILA ESPERA TERMINAR ANTES DE TUDO (PEDIDO DELE: NADA NA PLACA ANTES DISSO)
def main(slot, after=None):
    os.makedirs(f'{EXP}/queue_done', exist_ok=True)
    os.makedirs(f'{EXP}/queue_running', exist_ok=True)

    while after and os.path.exists(f'/proc/{after}'):
        time.sleep(120)

    while True:
        while busy():
            time.sleep(120)

        jobs = [job for job in pending() if job.get('slots', 'any') in ('any', slot)]
        if not jobs:
            print(time.strftime('%d/%m %H:%M'), 'fila vazia', flush=True)
            return

        room = free() - MARGIN
        fits = [job for job in jobs if need(job) <= room]
        if not fits:
            time.sleep(120)
            continue

        job = fits[0]
        if not claim(job['id']):
            continue

        t0 = time.time()
        print(time.strftime('%d/%m %H:%M'), 'começa', job['id'], flush=True)
        code = subprocess.run([PYTHON, f'{EXP}/job.py', job['id']], stdout=open(f'{EXP}/logs/{job["id"]}.log', 'a'), stderr=subprocess.STDOUT).returncode
        print(time.strftime('%d/%m %H:%M'), 'fim', job['id'], 'código', code, f'{(time.time() - t0) / 60:.0f} min', flush=True)

        if code == 0:
            open(f'{EXP}/queue_done/{job["id"]}', 'w').write(time.strftime('%d/%m/%Y %H:%M'))
        else:
            hold(job['id'], f'falhou com código {code}')

        os.remove(f'{EXP}/queue_running/{job["id"]}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'a', sys.argv[2] if len(sys.argv) > 2 else None)
