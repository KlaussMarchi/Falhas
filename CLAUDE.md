# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Segmentação de falhas geológicas em sísmica 3D: gera volumes sintéticos, treina redes 3D neles e prediz nos blocos reais
de `Blocks/` — Marlim (com a interpretação do especialista, onde o resultado é medido) e Iracema (sem interpretação). Não
há build, lint nem suíte de testes — a verificação é rodar o notebook de cima para baixo e comparar os números.

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
- GPU (Quadro P6000 de 24 GB, Pascal: AMP não acelera e float64 roda a 1/32 do float32): o treino roda com AMP só
  quando o `info.json` tem `"amp": true` (padrão desligado), `epochs` do `info.json` (padrão 100) e early stopping sobre
  `val_iou` com `patience` do `info.json` (padrão 15). Uma rodada custa horas; nunca re-execute um treino para "verificar" sem pedir.
- `.gitignore` exclui `*.npy`, `*.pth`, `*.dat`, imagens, `*.zip` e `Articles/`: datasets, blocos reais, pesos e figuras
  são locais. Um clone limpo precisa regerar os dados pelo `Synthetic/Analysis.ipynb` (seção "APLICANDO DATASET") +
  `Format.ipynb`.
- Disco: os patches do Marlim ocupam 29 GB e os do Iracema 302 GB. Predizer um modelo no Marlim grava ~7,3 GB de
  máscaras por patch, mais 0,9 GB do `predicted.npy`; o Iracema inteiro (~308 GB) não cabe. Confira o espaço antes.
- Rodadas longas: log fora do `/tmp`, que é apagado no boot. Para encerrar um processo use o PID explícito, porque
  `pkill -f` e `pgrep -f` casam com a própria linha de comando.

## Fluxo dos dados

Os estágios trocam **arquivos**, nunca variáveis, e cada um lê o que o anterior gravou:

1. **`Synthetic/index.py` — `SyntheticGenerator`.** O fluxo do FaultSeg3D (Wu et al. 2019,
   `Documents/Articles/fault_seg (wu).pdf`): refletividade → dobra → cisalhamento → falhamento → wavelet de Ricker →
   ruído → corte da margem.
   - O cubo de 256³ é calculado na GPU em torch float64 e reproduz a versão scipy antiga bit a bit.
   - `fast = True` sorteia os campos de ruído em float32 na placa: mesma distribuição, outro tile, ~0,7 s por tile
     contra 2–7 s. Serve só para buscas.
   - `get()` devolve `(image z-scorado, mask)`.
   - `set(options)` sobrescreve os parâmetros. Cada atributo é uma faixa `(low, high)` sorteada por tile; o teto do
     `randint` é exclusivo, então `faultCount (0, 1)` é tile sem falha.
   - `dataset(options)` gera em série na placa e grava `<directory>/<output>/{images,masks}/img_XXXX.npy`. O `options` é
     `{'directory', 'seed', 'regions': {nome: {'n_images', 'output', 'params'}}}`, e chave desconhecida levanta erro.
   - `gain`/`gainJitter`/`clip` ajustam o contraste por região depois do z-score.
   - Os padrões da classe são a configuração do `dataset_74`, e a ordem dos sorteios do `np.random` define cada tile.
     Ao mudar uma etapa, confira um tile contra o `git show HEAD:Synthetic/index.py`: a máscara tem de sair idêntica.
   - O `Synthetic/Analysis.ipynb` mostra cada etapa e monta um dataset novo.
   - `Synthetic/Regions/Generator.ipynb` tem um gerador próprio (`MarlimSyntheticGenerator`), copiado do bloco real,
     que alimenta o `Dataset/marlim_opt`.
   - O `Synthetic/PROMPT.md` é o roteiro da próxima tarefa: calibrar o gerador por bloco.
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
   trials); **`Model/3 - Predict.ipynb`** reavalia os modelos de um `BASE_PATH` no dataset em que cada um treinou (ou
   num `DATASET` comum, sem os tiles vistos no treino), somando os voxels de todos os tiles de teste antes de dividir —
   a mesma conta do `test_iou` — e grava as métricas no `test.<dataset>` do `info.json`. O protocolo da Tabela 2 da
   ResACEUnet (recortes 96³) saiu dele em 08/10/2026 e só aparece nos relatórios de `Documents/`. A matriz de confusão e
   a curva ROC (opcional, `ROC_CURVE`; histograma de log-odds por voxel, acumulado na mesma passada da rede) moram em
   `Model/Metrics/ConfusionMatrix` e `Model/Metrics/ROC`. Com `SAVE_PATH`, cada modelo grava `matrix.png` e `roc.png` na
   própria pasta (`_<dataset>` no nome quando testado fora do dataset do treino), e a `auc` entra no `test.<dataset>` do
   `info.json`.
6. **`Blocks/Marlim/Analysis.ipynb`** — o bloco real com interpretação. Predição e análise ficam num notebook só.
   - **Configuração:** `BASE_PATH` (a pasta dos `model_N/`: `../../Model/Backup` ou `../../Documents/Backups/<campanha>`),
     `MODELS`, `PATCH_IDS`, `OVERLAP`, `WINDOW` e os botões do extrator (`STICK_TOL 0.80`, `STICK_MIN 140`,
     `STICK_GAP 59`, calibrados no patch 1200).
   - **Entrada:** `files/patches/<id>/tiles/*.dat`, float32 cru em `[0,1]` (p01/p99 do bloco), eixos
     `(inline, z, xline)`, shape no `patch_metadata.json` da mesma pasta. Os 128 inlines de cada `.dat` são reais; o
     `original_shape` de 64 é só o miolo.
   - **Predição:** roda a `SlidingWindow` **na janela em que a rede foi treinada**, com peso de Hanning na emenda, e
     grava `masks/` + `predict.json` + `predicted.npy` em `<BASE_PATH>/<modelo>/marlim/patch_<id>`. Só prediz modelo com
     `scaling` `'normalize'` ou `'percentile'`.
   - **Análise:** extrai sticks (`FaultStickExtractor`) e compara com o especialista (`FaultComparer`, contra
     `files/patches/<id>/<id>_interpretado.png`). A inline anotada não é a central; o índice dela está em
     `files/cache/inline_index.json`.
   - **Saída:** figuras e `sticks_report.csv` em `<BASE_PATH>/outputs/marlim`.
   - A anotação é parcial (só as falhas principais): a precisão é limite inferior e o F1 premia marcar pouco.
7. **`Blocks/Iracema/Analysis.ipynb`** — o mesmo notebook para o Iracema, que não tem interpretação: só figuras e o
   resumo da inline. Os tiles ficam soltos em `files/patches/`, com um `files/patch_metadata.json` (2751 × 2000 × 1754,
   38 528 tiles). Com `FULL_PREDICT = False` ele prediz só os tiles que cobrem a `INLINE`.
8. **`Synthetic/Regions/Marlim/Analysis.ipynb`** separa o bloco real em tiles `calm`/`faulted`/`dead` por conteúdo
   (semblance, envelope RMS, distância da falha anotada) → `files/<região>/*.npy` + `files/DataBase.csv`, com
   `img_path` absoluto. Os patches continuam sendo lidos de `Blocks/Marlim/files/patches`.
9. **`Synthetic/Regions/Synthetic/Analysis.ipynb`** calibra o gerador contra o `faulted` real com o CMA-ES do
   `Synthetic/Nature`.
   - A `ImageSimilarity` mede a distância pelos olhos de uma rede de falhas: o KID do gargalo da `Unet3D_V2` em
     `Regions/Synthetic/model/`, treinada no `dataset_wu`, mais a fração detectada e o período vertical.
   - O objetivo da `Calibration` soma essa similaridade à largura dos intervalos (genoma em centro + log da largura). O
     IoU da rede nos rótulos entra só como piso por tile.
   - A busca parte da configuração do `dataset_74`, grava o `Dataset/dataset_regions` (220 tiles, com `synthetic.json`) e
     guarda as campanhas retomáveis em `files/memory/`.
   - O `model/info.json` ainda chama a rede de `unet3d_v2`, nome que o `ModelNetwork` não reconhece mais.
10. **`Synthetic/Nature/`** — framework mono-objetivo de otimização usado pela calibração.
    - É importado com a raiz do repositório no path: `from Synthetic.Nature.index import NatureSelector`.
    - `NatureSelector(nome, params, memory)` escolhe entre `genetic` (CMA-ES; aceita `mean` para a primeira nuvem nascer
      num ponto conhecido), `pso`, `de`, `lshade` e `lsrtde`, e delega `update()`/`portrait()`/`info()`.
    - `Problem` traduz `{variável: {'type', 'bounds'}}` em genoma.
    - `Memory` persiste `state.npz`/`best.json`/`history.json`.
11. **`Documents/Backups/<campanha>/model_N/`** (`Artigo`, `Marcia`, `Multiscale`) guarda modelos de campanhas
    anteriores no formato de `Model/Backup`, com as figuras dos blocos em `outputs/`; os notebooks dos blocos trocam a
    pasta pelo `BASE_PATH`. `Documents/Articles/` tem os PDFs das redes implementadas e o do FaultSeg3D (fora do git).

## Convenções que atravessam o repositório

- **Módulo é pasta com `index.py`** exportando o tipo principal: `from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Synthetic.Nature.index import NatureSelector`. Notebook fora da pasta ajusta
  o path (`sys.path.append('../../Model')` nos `Blocks/`, `sys.path.append('../../..')` no `Synthetic/Regions/Synthetic`).
  Importar módulo nunca executa trabalho.
- **Eixos:** os datasets são gravados em `(x, z, y)` — o `saveTile` transpõe `(0, 2, 1)` na saída do gerador, e
  `Synthetic/utils.formatAxis` faz o mesmo para visualizar. Os tiles reais (os `.dat` dos blocos e as regiões) são
  `(inline, z, xline)`, o mesmo arranjo.
- **Nova rede:** arquivo em `Model/Network/types/X.py`, import e um `if` em `ModelNetwork.get()` com uma linha
  MAIÚSCULA citando a origem; o nome usado ali é o que vai em `Task/info.json`. Hoje: `unet_3d`, `dbrnet`,
  `segresnet`, `resaceunet_grva`, `resaceunet_wu` (a `ResACEUNet2` atual do repositório dos autores),
  `resaceunet_zu` (a do artigo, 3 estágios, do primeiro commit), `macnn`, `fault_seg_net`, `nru_net`, `fault_edge_former`,
  `nra_sanet`. A `dbrnet` é a antiga `unet3d_v2`: um `info.json` com o nome velho só carrega trocando o nome.
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
