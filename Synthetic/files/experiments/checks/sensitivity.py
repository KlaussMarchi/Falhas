# SONDA DE SENSIBILIDADE (CPU): A dbrnet DE REFERÊNCIA (Artigo/model_5) EM TILES SINTÉTICOS SEM FALHA, CADA UM COM UM SÓ TIPO DE
# SÓSIA (CORPO, DOBRA ÍNGREME, ÁGUA/ZONA MORTA, GANHO, PERÍODO LONGO, RUÍDO BAIXO): QUANTO ELA MARCA ONDE NÃO HÁ FALHA NENHUMA.
# O CONTROLE POSITIVO TEM FALHAS (FRAÇÃO DO RÓTULO RECUPERADA). ENTRADA NA ESCALA DO TREINO DELA (MIN-MAX DO VOLUME).
import sys, json, time
import numpy as np, torch, pandas as pd
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
ROOT = '/home/grva-mint/Projects/Falhas'
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/Model')
from Synthetic.index import SyntheticGenerator
from Network.index import ModelNetwork

MODEL = f'{ROOT}/Documents/Backups/Artigo/model_5'
CASES = {
    'camadas (sem falha)': {'faultCount': (0, 1)},
    'corpos': {'faultCount': (0, 1), 'bodyCount': (2, 4)},
    'dobras íngremes': {'faultCount': (0, 1), 'foldSigma': (6, 20), 'foldAmplitude': (-40, 40)},
    'água + zona morta + hachura': {'faultCount': (0, 1), 'seafloorProb': 1.0, 'deadZoneProb': 1.0, 'seafloorDepth': (0.1, 0.3), 'deadZoneDepth': (0.4, 0.7), 'deadZoneHatch': (0.1, 0.2)},
    'campo de ganho 0,6': {'faultCount': (0, 1), 'gainField': 0.6},
    'período longo (f 33-55)': {'faultCount': (0, 1), 'waveletFreq': (33, 55)},
    'ruído baixo (0-0,15)': {'faultCount': (0, 1), 'noiseLevel': (0.0, 0.15)},
    'falhas (controle positivo)': {},
}
SEEDS = (31, 32, 33)

info    = json.load(open(f'{MODEL}/info.json'))
network = ModelNetwork(**{**info['model']})
network.model.load_state_dict(torch.load(f'{MODEL}/model.pth', map_location='cpu', weights_only=False)['model'])
network.model.eval()
rows = []
for name, params in CASES.items():
    for seed in SEEDS:
        g = SyntheticGenerator(seed=seed)
        g.set({**params, 'device': 'cpu'})
        img, msk = g.get()
        img = np.transpose((img - img.min()) / (img.max() - img.min()), (0, 2, 1)).astype(np.float32)
        msk = np.transpose(msk, (0, 2, 1)) > 0
        t0 = time.time()
        with torch.no_grad():
            prob = torch.sigmoid(network.model(torch.from_numpy(img)[None, None])).numpy()[0, 0]
        pred = prob >= 0.5
        rows.append({'caso': name, 'semente': seed, 'marcado': float(pred.mean()), 'rotulo': float(msk.mean()), 'recall_voxel': float((pred & msk).sum() / max(msk.sum(), 1)), 'prob_media': float(prob.mean()), 'seg': round(time.time() - t0)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        np.save(f'{ROOT}/Synthetic/files/experiments/checks/sens_{len(rows) - 1:02d}.npy', np.stack([img[:, :, 64], prob[:, :, 64].astype(np.float32), msk[:, :, 64].astype(np.float32)]))
pd.DataFrame(rows).to_csv(f'{ROOT}/Synthetic/files/experiments/checks/sensitivity.csv', index=False)
print(pd.DataFrame(rows).groupby('caso', sort=False)[['marcado', 'rotulo', 'recall_voxel']].mean().round(4).to_string())
