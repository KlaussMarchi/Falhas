# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Segmentação de falhas geológicas em sísmica 3D: gera volumes sintéticos, treina redes 3D neles e prediz no bloco real de
Marlim. Não há build, lint nem suíte de testes — a verificação é rodar o notebook de cima para baixo e comparar os números.

## Ambiente e execução

Tudo roda no ambiente conda `torch-gpu` (torch 2.7.1+cu118, monai 1.5.2, torchmetrics, albumentations, papermill,
opencv, scikit-image, psutil). O python do `base` **não tem torch** — nunca rode um script deste projeto sem ativar:

```bash
conda activate torch-gpu           # ou: conda run -n torch-gpu python ...
```

- **Campanha de treinos (o caminho normal):** edite `Task/task.json` (lista de rodadas) e rode `cd Task && python index.py`.
  Para cada rodada ele grava a linha em `Task/info.json`, executa `Dataset/<dataset>/Format.ipynb` (pulado quando o
  `DataBase.csv` já existe e `img_size` é `null`) e depois `Model/Analysis.ipynb`, via papermill, com a saída em
  `Task/logs/<nome>_out.ipynb`. O kernel é `python3`, que resolve para o do próprio `torch-gpu`.
- **Um notebook só:** abra no Jupyter/VS Code com o kernel `torch-gpu`, ou
  `papermill <nb> /tmp/out.ipynb -k python3 --cwd <pasta do nb>` — todo caminho dentro de um notebook é relativo à
  **pasta dele**, então o `cwd` errado quebra tudo.
- GPU: `Model/Analysis.ipynb` treina com AMP desligado (`use_amp = False`), batch 2 e 100 épocas com early stopping.
  Uma rodada custa horas; nunca re-execute um treino para "verificar" sem pedir.
- `.gitignore` exclui `*.npy`, `*.pth`, `*.dat`, `*.png`: datasets, pesos e figuras são locais. Um clone limpo precisa
  regerar os dados pelo `Synthetic/Generate.ipynb` + `Format.ipynb`.

## Fluxo dos dados

Os estágios trocam **arquivos**, nunca variáveis, e cada um lê o que o anterior gravou:

1. **`Synthetic/index.py` — `SyntheticGenerator`.** Refletividade → dobra → cisalhamento → falhamento → wavelet de
   Ricker → ruído → corte da margem. `get()` devolve `(image z-scorado, mask)`; `set(options)` sobrescreve os
   parâmetros (cada atributo é uma faixa `(low, high)` sorteada por tile); `dataset(options, n_jobs)` gera em
   multiprocesso e grava `<directory>/<output>/{images,masks}/img_XXXX.npy`. O `options` é
   `{'directory', 'seed', 'regions': {nome: {'n_images', 'output', 'params'}}}` e chave desconhecida levanta erro.
   `gain`/`gainJitter`/`clip` ajustam contraste por região depois do z-score. O `Synthetic/Generate.ipynb` é quem chama
   o gerador para montar um dataset novo; `Marlim/files/Generator.ipynb` tem um gerador próprio
   (`MarlimSyntheticGenerator`), copiado do bloco real, que alimenta `Dataset/marlim_opt`.
2. **`Dataset/<nome>/Format.ipynb`.** Lê `original/*/images|masks`, normaliza para `[0,1]` e grava `images/`, `masks/` e
   o `DataBase.csv` (`id`, estatísticas, `shape`, `img_path`/`mask_path` absolutos). Também reescreve
   `Task/info.json` com o nome do dataset. A normalização é `clip(p01, p99)` do conjunto inteiro reescalado para
   `[0,1]`; em dataset por região (`dataset_regions`) os trilhos são o `clip` **declarado** no `synthetic.json`, porque
   o percentil da amostra cairia dentro deles e mudaria com o número de tiles de cada região.
3. **`Task/info.json`** é a configuração única da rodada (`network`, `dataset`, `img_size`, `lr`, `loss`, `batch_size`,
   `scheduler`, `dropout`, `num_filters`), lida como `OPTIONS` pelo `Model/Analysis.ipynb`.
4. **`Model/Analysis.ipynb`** — o notebook de treino. Split com ~4,5% para validação e ~4,5% para teste (num dataset de
   220 tiles dá 200/10/10), `CustomDataset` + `Compose` do `Transforms/`,
   `Trainer` (clip de gradiente, `ReduceLROnPlateau`/`CosineAnnealingWarmRestarts`, `EarlyStopping` sobre `val_iou` com
   paciência 15, progresso corrente em `Model/progress.json`). Salva `Model/Backup/model_N/` com `info.json`,
   `model.pth` (`{'model', 'optimizer', 'timestamp', 'history'}`), `train.png` e `predictions/`. `N` é o maior
   `model_N` existente + 1. O `img_size` do `info.json` vem do `shape` do `DataBase.csv`, não do `Task/info.json`.
5. **`Model/Predict.ipynb`** reavalia um modelo salvo no dataset dele; **`Model/Compare.ipynb`** junta todo
   `Backup/*/info.json` numa tabela e compara variações.
6. **`Marlim/1 - Predict.ipynb`** — bloco real. Lê `Dataset/marlim/patch_<id>/*.dat` (float32 cru, shape no
   `patch_metadata.json`), roda `SlidingWindow` **na janela em que a rede foi treinada** (peso de Hanning na emenda,
   `OVERLAP` configurável) e grava as máscaras em `Model/Backup/model_N/marlim/patch_<id>/masks` + `predict.json`.
7. **`Marlim/2 - Analysis.ipynb`** — remonta o volume predito, extrai sticks (`FaultStickExtractor`) e compara com a
   interpretação do especialista (`Marlim/files/patches/<id>/<id>_interpretado.png`) pelo `FaultComparer`; sai figura
   em `Marlim/files/comparisons/` e o CSV `sticks_report_<base>.csv`.
8. **`Marlim/Regions/Marlim/Analysis.ipynb`** separa o bloco real em tiles `calm`/`faulted`/`dead` por conteúdo
   (semblance, envelope RMS, distância da falha anotada) → `files/<região>/*.npy` + `files/DataBase.csv`.
9. **`Marlim/Regions/Synthetic/Analysis.ipynb`** calibra o gerador região a região: `ImageSimilarity` dá uma nota em %
   entre o lote sintético e os tiles reais da região, e o `NatureSelector` busca o genoma que maximiza a nota. O
   `README.md` da pasta documenta a função de similaridade (régua de atributos, coeficiente de energia, calibração das
   notas) — leia antes de mexer nela. As campanhas ficam em `files/memory/<região>_<hash>/` e são retomáveis.
10. **`Nature/`** — framework mono-objetivo de otimização usado pela calibração. `NatureSelector(nome, params, memory)`
    escolhe entre `genetic` (CMA-ES), `pso`, `de`, `lshade`, `lsrtde` e delega `update()`/`portrait()`/`info()`;
    `Problem` traduz `{variável: {'type', 'bounds'}}` em genoma; `Memory` persiste `state.npz`/`best.json`/`history.json`.
11. **`Marcia/model_N/`** guarda modelos antigos/externos no mesmo formato de `Model/Backup`; os dois notebooks do
    Marlim aceitam `BASE_PATH = '../Marcia'`.

## Convenções que atravessam o repositório

- **Módulo é pasta com `index.py`** exportando o tipo de mesmo nome: `from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Nature.index import NatureSelector`. Notebook fora da pasta ajusta o path
  (`sys.path.append('../Model')`, `sys.path.append('../../..')`). Importar módulo nunca executa trabalho.
- **Eixos:** os datasets são gravados em `(x, z, y)` — o `saveTile` transpõe `(0, 2, 1)` na saída do gerador, e
  `Synthetic/utils.formatAxis` faz o mesmo para visualizar. Os tiles reais das regiões são `(inline, z, xline)`.
- **Nova rede:** arquivo em `Model/Network/types/X.py` e um `if` em `ModelNetwork.get()`; o nome usado ali é o que vai
  em `Task/info.json`. Hoje: `standard`, `unet3d_v2`, `segresnet`, `resaceunet`, `resaceunet_grva`, `macnn`.
- **Nova loss:** classe em `Model/Losses/index.py` e entrada em `Losses.options` (`cross_entropy`, `dice_focal`,
  `focal`, `smooth_dice`). Toda loss força `float32` fora do autocast. No MONAI 1.5.2 o termo focal do `DiceFocalLoss`
  é sempre sigmoide, mesmo no caso multiclasse — considere isso antes de comparar campanhas `dice_focal`.
- **Augmentação:** o notebook de treino usa `Model/Transforms/index.py`; `Model/Augmentor/index.py` é uma cópia antiga
  quase idêntica — mexa no `Transforms/`. Como o `Format` já grava em `[0,1]` e val/teste não passam por transform,
  `Normalize`/`Clip` no treino dessincronizam as distribuições e derrubam o IoU.
- **Idioma:** identificadores em inglês e `camelCase`; comentários, markdown, títulos de gráfico, commits e relatórios
  em português. O relatório de qualquer trabalho feito aqui é em português.
- **`CODE_STYLE.md` na raiz é a regra de escrita** (uma linha MAIÚSCULA acima de cada classe, sem docstring, sem type
  hint, sem underscore inicial, chamada nunca quebrada em várias linhas, uma etapa por célula terminando em prova
  visível, diferença vira dado). Leia a seção pertinente antes de escrever ou reorganizar código — principalmente a §10
  (notebooks) e a §14 (assinaturas de código gerado a remover).
