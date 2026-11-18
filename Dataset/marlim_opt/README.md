# marlim_opt

Dataset sintético para o bloco Marlim: 495 tiles de 128³ do `SyntheticGenerator` (`Synthetic/index.py`), gerados pelo `synthetic.json` desta pasta. É a configuração
`MOPT2` de `Synthetic/files/experiments/datasets/specs.json`, com os mesmos tiles que as redes de validação viram. Entrou aqui em 11/2026 no lugar da configuração
"calma", por decisão do usuário, depois que a auditoria de 15/11 mostrou que a calma era a que menos achava falha no Marlim.

## O que tem

Dois regimes do gerador no mesmo dataset, nenhum com rótulo nas falhas menores:

- **220 tiles no regime do dataset_74** (região `g74`, `img_0000` a `img_0219`): muitas falhas por tile (5 a 9), rejeito grande, ruído de 0,015 a 0,62 e rótulo fino. É a
  configuração do `Dataset/dataset_74/synthetic.json`. É o regime que dá sensibilidade: a rede aprende a falha com exemplos claros e aguenta ruído.
- **275 tiles no regime C5**, calibrado no Marlim (regiões `z250`, `z500`, `z750`, `z1000` e as quatro `*_neg`): períodos por profundidade, coluna de água e zona morta,
  hachura de migração, mergulho de 62 a 84°, falhas menores presentes e **sem rótulo**. 55 desses tiles não têm falha principal. É o regime que dá máscara limpa.
- O contraste de cada região (`gain`, `gainJitter` e `clip`, quando houver) está no `synthetic.json`. O `clip` satura os refletores fortes, como o corte no p01/p99 faz no bloco real.
- `original/{images,masks}` é a saída do gerador. `images/`, `masks/` e `DataBase.csv` são o `Format` no modo **`'normalize'`**. A ordem das linhas do `DataBase.csv`
  define o split do `1 - Model`.

## Como treinar

No `Task/info.json`, use `"dataset": "marlim_opt"` e `"scaling": "normalize"`, com a receita R1: `lr` 1e-3, `loss` `dice_focal`, `batch_size` 1, `scheduler` `plateau`,
`dropout` 0,1, `num_filters` 32, `ema` true, 100 épocas, `patience` 15. Com 495 tiles, a época leva ~3,6 min na RTX 5070.

## O que as redes deste dataset fazem

Régua oficial do `Blocks/Marlim/Analysis.ipynb` (patches 1200, 1300, 1400 e 2600), com a máscara igualada a 1% e a 2% da seção viva, que compara redes pelo que elas
ordenam e não pelo quanto marcam:

| rede (`dbrnet` R1) | recall a 1% | F1 a 1% | recall a 2% | precisão a 2% | F1 a 2% |
|---|---|---|---|---|---|
| neste dataset, semente 42 | 0,370 | 0,397 | 0,498 | 0,312 | 0,373 |
| neste dataset, semente 43 | 0,314 | 0,335 | 0,402 | 0,271 | 0,315 |
| no dataset_74 (220 tiles), semente 42 | 0,331 | 0,364 | 0,432 | 0,258 | 0,317 |
| na versão anterior (calma), média de 17 redes | 0,228 | 0,274 | 0,341 | 0,294 | 0,305 |

A anotação do especialista é parcial (só as falhas principais), então a precisão é limite inferior. A própria anotação, dada como predição, tira recall 0,88 nesta régua.

## Uma rede só: calibre depois do treino

O ponto de operação que a rede traz do treino (limiar 0,5) muda muito com o dataset e a semente: a máscara vai de menos de 1% a vários por cento da seção. Por isso o
notebook pode mostrar uma rede boa como fraca. A calibração não treina nada: põe os espelhos laterais (TTA) e escolhe o expoente k que faz a máscara marcar 2% da seção viva das 4 inlines, sem olhar a anotação.

```bash
cd Synthetic/files/experiments
python checks/calibrar.py CAL_model_N /home/lamcegrva/Projects/Falhas/Model/Backup/model_N    # ~45 min na GPU; grava models/CAL_model_N
```

O modelo calibrado roda nos notebooks dos blocos com `BASE_PATH = '../../Synthetic/files/experiments/models/CAL_model_N'` e `MODELS = ['model_1']`.

| rede | ponto | recall | precisão | F1 |
|---|---|---|---|---|
| neste dataset, semente 42 | nativo | 0,422 | 0,353 | 0,378 |
| neste dataset, semente 42 | **calibrada** (notebook) | **0,602** | 0,274 | **0,368** |
| neste dataset, semente 43 | nativo | 0,308 | 0,350 | 0,323 |
| neste dataset, semente 43 | **calibrada** | **0,521** | 0,263 | 0,343 |
| `model_3` (versão calma) | nativo (notebook) | 0,396 | ~0,24 | 0,296 |
| `model_3` (versão calma) | calibrada (notebook) | 0,461 | 0,269 | 0,332 |
| âncoras do dataset_74 (Dissertacao/5, Marcia/1) | nativo | 0,564–0,572 | ~0,21 | 0,30–0,31 |

## O detector recomendado

Uma rede só varia muito com o sorteio dos tiles. O resultado estável vem da média de várias redes treinadas em sorteios e regimes diferentes (tipo `ensemble` do
`Model/Network`), com o ponto de operação escolhido sem rótulo: a média elevada a k, com k tal que a máscara marque 2% da seção viva do bloco.

- Estado e números atuais: `Synthetic/files/experiments/PLANO.md`.
- Como montar: `python Synthetic/files/experiments/checks/make_ensemble.py <nome> <rodadas> k=<valor>`; para outro bloco, `checks/make_ira.py <nome> <fração>`.

## Procedência

- Diário: `Synthetic/files/experiments/journal.md` (15 e 16/11/2026, META74 e H59). Números: `logs/ranking.jsonl`.
- A versão anterior (calma, `FINAL_w2mlcalm`) está inteira em `Synthetic/files/experiments/marlim_opt_calma/`. A de antes dela (gerador antigo, escala `'percentile'`) está
  no git e em `Synthetic/files/experiments/marlim_opt_antigo/`.
