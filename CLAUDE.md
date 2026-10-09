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

- **Campanha de treinos (o caminho normal):** edite `Task/task.json` (lista de rodadas, cada uma no formato de
  `Task/info.json`) e rode `cd Task && python index.py`. Para cada rodada ele grava `Task/info.json`, executa
  `Dataset/<dataset>/Format.ipynb` (pulado quando o `DataBase.csv` já existe, `img_size` é `null` e a coluna
  `scaling` dele bate com a da rodada) e depois
  `Model/1 - Model.ipynb` uma vez por trial (`n_trials`; o trial vai no `info.json` e a semente é `42 + trial`), via
  papermill, com a saída em `Task/logs/<nome>_out.ipynb`. O kernel é `python3`, que resolve para o do `torch-gpu`.
- **Um notebook só:** abra no Jupyter/VS Code com o kernel `torch-gpu`, ou
  `papermill <nb> /tmp/out.ipynb -k python3 --cwd <pasta do nb>` — todo caminho dentro de um notebook é relativo à
  **pasta dele**, então o `cwd` errado quebra tudo.
- GPU: o treino roda com AMP só quando o `info.json` tem `"amp": true` (padrão desligado), `epochs` do `info.json` (padrão 100) e early stopping sobre
  `val_iou` com `patience` do `info.json` (padrão 15). Uma rodada custa horas; nunca re-execute um treino para "verificar" sem pedir.
- `.gitignore` exclui `*.npy`, `*.pth`, `*.dat`, imagens e `*.zip`: datasets, pesos e figuras são locais. Um clone
  limpo precisa regerar os dados pelo `Synthetic/Generate.ipynb` + `Format.ipynb`.

## Fluxo dos dados

Os estágios trocam **arquivos**, nunca variáveis, e cada um lê o que o anterior gravou:

1. **`Synthetic/index.py` — `SyntheticGenerator`.** Refletividade → dobra → cisalhamento → falhamento → wavelet de
   Ricker → ruído → corte da margem. `get()` devolve `(image z-scorado, mask)`; `set(options)` sobrescreve os
   parâmetros (cada atributo é uma faixa `(low, high)` sorteada por tile); `dataset(options, n_jobs)` gera em
   multiprocesso e grava `<directory>/<output>/{images,masks}/img_XXXX.npy`. O `options` é
   `{'directory', 'seed', 'regions': {nome: {'n_images', 'output', 'params'}}}` e chave desconhecida levanta erro.
   `gain`/`gainJitter`/`clip` ajustam contraste por região depois do z-score. O `Synthetic/Generate.ipynb` monta um
   dataset novo; `Marlim/files/Generator.ipynb` tem um gerador próprio (`MarlimSyntheticGenerator`), copiado do bloco
   real, que alimenta `Dataset/marlim_opt`.
2. **`Dataset/<nome>/Format.ipynb`.** Lê `original/*/images|masks`, escalona e grava `images/`, `masks/` e
   o `DataBase.csv` (`id`, estatísticas, `shape`, `img_path`/`mask_path` absolutos, `scaling`). Também reescreve
   `Task/info.json` com o nome do dataset. O que se repete entre os `Format` mora no `Dataset/index.py`: `getFiles`,
   `setFolder`, `showTile`, `TilesBuilder` e a `Normalization`, escolhida pelo `scaling` do `info.json`:
   `'normalize'` (o padrão sem a chave: **min-max de cada volume** para `[0,1]`, decisão de 06/10/2026), `'percentile'`
   (corte no p01/p99 do conjunto inteiro, reescalado para `[0,1]`), `'standardize'` (média 0 e desvio 1 por volume, o
   passo do artigo da ResACEUnet) ou `null` (volume cru). A escolha fica na coluna `scaling` do `DataBase.csv`
   (`Normalization.read`: sem ela é o min-max, célula vazia é `null`), e o `1 - Model`, o `Model_CV` e o `3 - Predict`
   param com erro se ela não bater com a rodada. O `marlim_opt` é `'percentile'`: o min-max por volume desfaria o ganho
   global do gerador. O `dataset_74_wu` não tem `images/` próprios: junta os do `dataset_74` e do `dataset_wu` e roda,
   por papermill, o `Format` da fonte que não estiver no `scaling` da rodada (no percentil, cada fonte com o seu). Os
   modelos salvos antes de 06/10/2026 têm `scaling: "percentile"`. O Marlim continua no p01/p99 do próprio bloco;
   medido: a `dbrnet` treinada com percentil perde só 0,005 de IoU no dado com min-max. O `dataset_zu` é o `200-20.zip` que os
   autores da ResACEUnet publicaram (Zenodo 20339874): o próprio FaultSeg3D do `dataset_wu`, bit a bit e só renumerado
   (o `Compare.ipynb` da pasta prova; mapa no `README.md`).
3. **`Task/info.json`** é a configuração única da rodada (`network`, `dataset`, `img_size`, `lr`, `loss`, `batch_size`,
   `scheduler`, `dropout`, `num_filters`, `ema`, `n_trials`, opcionais `epochs`, `patience`, `amp`, `scaling` e
   `augmentations`), lida como `OPTIONS`.
4. **`Model/1 - Model.ipynb`** — o treino. Split fixo (`random_state=42`) com ~4,5% para validação e ~4,5% para teste
   (220 tiles dão 200/10/10), `CustomDataset`, `Trainer` (clip de gradiente,
   `plateau` = `ReduceLROnPlateau` por época, `cosine` = aquecimento 1e-6 → `lr` em 10 épocas e cosseno até 1e-7 a cada
   batch (a agenda da ResACEUnet), `ModelEMA` do `Model/EMA/` quando `ema` é true, progresso
   corrente em `Model/progress.json`). Salva `Model/Backup/model_N/` com `info.json`, `model.pth`
   (`{'model', 'optimizer', 'timestamp', 'history'}`), `train.png` e `predictions/`; `N` é o maior `model_N` + 1. No
   `info.json` salvo, `processing` é o `Task/info.json` exatamente como veio (copiar e colar reproduz a rodada),
   `division` guarda o split (`val_size`, `test_size`, `n_images`, e `k_fold` no CV) e `model` os argumentos da rede: o
   `model.img_size` vem do `shape` do `DataBase.csv` ou, com `crop` no `augmentations`, da janela do recorte. O
   `3 - Predict` lê o split do `division` e não tem valor próprio.
5. **`Model/2 - Compare.ipynb`** junta todo `Backup/*/info.json` numa tabela e compara variações (média±std entre
   trials); **`Model/3 - Predict.ipynb`** reavalia um modelo salvo no dataset dele e, se ele treinou em recorte, mede
   também pelo protocolo da Tabela 2 da ResACEUnet (recortes 96³ metade em falha, média por recorte; no código deles
   precisão e recall estão trocados).
6. **`Marlim/1 - Predict.ipynb`** — bloco real. Lê `Marlim/files/patches/<id>/tiles/*.dat` (float32 cru, shape no
   `patch_metadata.json` da mesma pasta), roda `SlidingWindow` **na janela em que a rede foi treinada** (peso de Hanning na emenda,
   `OVERLAP` configurável) e grava as máscaras em `Model/Backup/model_N/marlim/patch_<id>/masks` + `predict.json`;
   só prediz modelo com `scaling` `'normalize'` ou `'percentile'`, porque o Marlim está em `[0,1]`.
7. **`Marlim/2 - Analysis.ipynb`** — remonta o volume predito, extrai sticks (`FaultStickExtractor`) e compara com a
   interpretação do especialista (`FaultComparer`, contra `Marlim/files/patches/<id>/<id>_interpretado.png`, inline
   anotada em `Marlim/files/cache`); sai figura em `Marlim/files/comparisons/` e o CSV `sticks_report_<base>.csv`.
8. **`Marlim/Regions/Marlim/Analysis.ipynb`** separa o bloco real em tiles `calm`/`faulted`/`dead` por conteúdo
   (semblance, envelope RMS, distância da falha anotada) → `files/<região>/*.npy` + `files/DataBase.csv`.
9. **`Marlim/Regions/Synthetic/Analysis.ipynb`** calibra o gerador contra o `faulted` real. A `ImageSimilarity` mede a
   distância pelos olhos de uma rede de falhas (KID do gargalo da `Unet3D_V2` em `Regions/Synthetic/model/`, treinada no
   `dataset_wu`); o objetivo da `Calibration` é a média ponderada dessa similaridade com o IoU da rede nos rótulos do
   lote. A busca parte da configuração do `dataset_74`, grava o `Dataset/dataset_regions` (220 tiles, com
   `synthetic.json`) e guarda as campanhas retomáveis em `files/memory/`.
10. **`Nature/`** — framework mono-objetivo de otimização usado pela calibração. `NatureSelector(nome, params, memory)`
    escolhe entre `genetic` (CMA-ES, aceita `mean` para a primeira nuvem nascer num ponto conhecido), `pso`, `de`,
    `lshade`, `lsrtde` e delega `update()`/`portrait()`/`info()`; `Problem` traduz `{variável: {'type', 'bounds'}}` em
    genoma; `Memory` persiste `state.npz`/`best.json`/`history.json`.
11. **`Documents/Marcia/model_N/`** e **`Documents/Backup/`** guardam modelos antigos/externos no formato de
    `Model/Backup`; os notebooks do Marlim trocam a pasta pelo `BASE_PATH`. `Documents/Articles/` tem os PDFs das redes
    implementadas (fora do git).

## Convenções que atravessam o repositório

- **Módulo é pasta com `index.py`** exportando o tipo principal: `from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Nature.index import NatureSelector`. Notebook fora da pasta ajusta o path
  (`sys.path.append('../Model')`, `sys.path.append('../../..')`). Importar módulo nunca executa trabalho.
- **Eixos:** os datasets são gravados em `(x, z, y)` — o `saveTile` transpõe `(0, 2, 1)` na saída do gerador, e
  `Synthetic/utils.formatAxis` faz o mesmo para visualizar. Os tiles reais das regiões são `(inline, z, xline)`.
- **Nova rede:** arquivo em `Model/Network/types/X.py`, import e um `if` em `ModelNetwork.get()` com uma linha
  MAIÚSCULA citando a origem; o nome usado ali é o que vai em `Task/info.json`. Hoje: `unet_3d`, `dbrnet`,
  `segresnet`, `resaceunet_grva`, `resaceunet_wu` (a `ResACEUNet2` atual do repositório dos autores),
  `resaceunet_zu` (a do artigo, 3 estágios, do primeiro commit), `macnn`, `fault_seg_net`, `nru_net`, `fault_edge_former`.
- **Nova loss:** classe em `Model/Losses/index.py` e entrada em `Losses.options` (`cross_entropy`, `dice_focal`,
  `dice_ce`, `focal`, `smooth_dice`, `compound`, `tversky`). Toda loss força `float32` fora do autocast. No MONAI 1.5.2 o termo
  focal do `DiceFocalLoss` é sempre sigmoide, mesmo no caso multiclasse — considere isso antes de comparar campanhas
  `dice_focal`.
- **Aumentação:** só pelo `augmentations` do `info.json` (`null`/ausente = tiles do `Format` como estão, bit a bit igual
  ao pipeline sem ela). `Model/Transforms/index.py` reproduz o `build_data.py` da ResACEUnet (gamma, rot90, flip,
  rotação, suavização, ruído, recorte falha/fundo) mais o `zoom` isotrópico (fator ≥ 1, para cobrir o período de 17–25 px
  das falhas do Marlim), sorteado no `DataLoader` com semente `(42 + trial, época, índice)`,
  na ordem do JSON e só no treino; `n_aug` (padrão 1, como nos autores) é quantas variações de cada tile entram por
  época. Os eixos do config são os nossos `(x, z, y)`: giros e flips em `[0, 2]`. Com `crop`, a
  rede é montada na janela e o `Trainer`/`3 - Predict` predizem o tile inteiro por `Transforms.infer` (janela deslizante).
- **Pool por `fork` depois de cv2:** notebook que usa cv2 no processo principal e depois cria pool por `fork` chama
  `cv2.setNumThreads(1)` na primeira célula — sem isso os filhos travam em futex e o `pool.map` espera para sempre.
- **Idioma:** identificadores em inglês e `camelCase`; comentários, markdown, títulos de gráfico, commits e relatórios
  em português. O relatório de qualquer trabalho feito aqui é em português.
- **Estilo de escrita:** uma linha de comentário MAIÚSCULA acima de cada classe, sem docstring, sem type hint, sem
  underscore inicial, chamada nunca quebrada em várias linhas; nos notebooks, uma etapa por célula terminando numa prova
  visível (print, tabela ou figura), e a explicação vai no markdown da seção, em tópicos.
