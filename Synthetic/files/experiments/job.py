# UM JOB DA FILA: 'train' (TREINO PRIVADO + AVALIAÇÃO NOS PATCHES PEDIDOS) OU 'eval' (SÓ A AVALIAÇÃO DE UM MODELO PRONTO)
import os, sys, json, glob, subprocess

EXP = '/home/grva-mint/Projects/Falhas/Synthetic/files/experiments'
sys.path.insert(0, EXP)
sys.path.insert(0, f'{EXP}/harness')


def git():
    return subprocess.run(['git', '-C', '/home/grva-mint/Projects/Falhas', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()


def evaluate(model, tag, spec, extra):
    import marlim
    for fast in spec.get('methods', [True]):
        marlim.run(model, spec.get('patches', ['1200', '1300']), tag, fast=fast, results=f'{EXP}/results.jsonl', save=f'{EXP}/sections', extra={**extra, 'git': git()})

    # A BATERIA DE SÓSIAS (battery.py) VAI JUNTO DA AVALIAÇÃO RÁPIDA: QUANTO O MODELO MARCA ONDE NÃO HÁ FALHA, POR TIPO DE SÓSIA
    if True in spec.get('methods', [True]):
        import battery, injection
        battery.run(model, tag)
        injection.run(model, tag)


# SUBCONJUNTO FIXO DE TILES (OS MESMOS ÍNDICES EM TODO DATASET: SEMENTES COMUNS) PARA AS SONDAS BARATAS
def subset(job):
    import numpy as np, pandas as pd
    df   = pd.read_csv(job['source'])
    pick = np.sort(np.random.default_rng(0).choice(len(df), job['subset'], replace=False))
    os.makedirs(os.path.dirname(job['database']), exist_ok=True)
    df.iloc[pick].to_csv(job['database'], index=False)


def main(jobId):
    job = next(j for j in json.load(open(f'{EXP}/queue.json')) if j['id'] == jobId)
    print(json.dumps(job), flush=True)

    if job['kind'] == 'train':
        import train, time
        # O DATASET PODE AINDA ESTAR SENDO GERADO NA CPU: ESPERA ATÉ 6 h PELO DataBase.csv (O SUBCONJUNTO É ESCRITO DEPOIS DELE)
        for _ in range(360):
            if os.path.exists(job['database']) or os.path.exists(job.get('source', job['database'])) and job.get('subset'):
                break
            time.sleep(60)
        if job.get('subset'):
            subset(job)
        run = f'{EXP}/models/{job["run"]}'
        os.makedirs(run, exist_ok=True)
        json.dump({'info': job['info'], 'database': job['database']}, open(f'{run}/config.json', 'w'), indent=1)
        train.main(job['run'])
        model = sorted(glob.glob(f'{run}/model_*'))[-1]
        info  = json.load(open(f'{model}/info.json'))['trainer']
        evaluate(model, job['run'], job.get('eval', {}), {'dataset': job['info'].get('dataset'), 'val_iou': info.get('val_iou'), 'test_iou': info.get('test_iou'), 'epochs': job['info'].get('epochs'), 'trial': job['info'].get('trial', 0)})

    if job['kind'] == 'eval':
        evaluate(job['model'], job['tag'], job, job.get('extra', {}))


if __name__ == '__main__':
    main(sys.argv[1])
