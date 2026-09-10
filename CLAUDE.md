# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

`Falhas` é um projeto de **detecção de falhas sísmicas em volumes 3D** (segmentação binária voxel-a-voxel,
`classes=1`, sigmoid + IoU). O modelo é treinado em **dados sintéticos** e avaliado no **bloco real de
Marlim** contra a interpretação de um especialista. Não há aplicação, servidor nem pacote instalável: tudo é
notebook Jupyter + folder-modules (`Pasta/index.py`), orquestrados por papermill.

Fontes de dado que convivem em `Dataset/`:
- **sintético geral** (`Synthetic/Generate.ipynb` → `dataset_74`, 220 volumes; `dataset_74_2000`, 2000) —
  gerador paramétrico `SyntheticGenerator`;
- **sintético calibrado para Marlim** — duas linhagens: `Marlim/Generator.ipynb` (classe própria
  `MarlimSyntheticGenerator` → `marlim_opt`) e `Marlim/Generator2.ipynb` (o `SyntheticGenerator` global com
  uma camada de domínio por região → `dataset_marlim_opt_2`, 660 tiles = 220 × 3 regiões);
- **Wu** (`dataset_wu`) — dataset público de referência, já vem pronto em `original/`;
- **mixes** (`dataset_74_mar`, `dataset_wu_mar`, `dataset_74_wu`, `dataset_74_wu_mar`) — só **concatenam**
  dois datasets **já formatados**; não renormalizam nada.

## Ambiente e comandos

- **Interpretador: `~/anaconda3/envs/torch-gpu/bin/python`** (Python 3.11, torch 2.7.1+cu118, monai 1.5.2,
  papermill 2.7, torchmetrics 1.8, scikit-image 0.26, opencv 4.12). O conda **`base`** é o primeiro no `PATH`
  e tem o `jupyter`, mas **não** tem torch nem papermill — nada roda nele. Há um kernelspec `torch-gpu` (use
  esse ao abrir notebooks no Jupyter) e o `python3`, que resolve para o env `torch-gpu` quando o processo pai
  já é o interpretador do env (é por isso que `KERNEL_NAME='python3'` funciona no `Task/index.py`).
- **Não há linter, build nem testes.** A validação é rodar o notebook (célula a célula ou headless) e olhar
  as figuras/métricas que cada célula imprime (CODE_STYLE.md: "toda célula termina com prova visual").
- **Pipeline completo de treino** (o jeito normal de rodar um experimento):
  ```
  cd Task && ~/anaconda3/envs/torch-gpu/bin/python index.py
  ```
  Lê `Task/task.json` (lista de configs), escreve cada uma em `Task/info.json` e executa, via papermill,
  `../Dataset/<dataset>/Format.ipynb` e depois `../Model/Analysis.ipynb`. As saídas executadas vão para
  `Task/logs/<nome>_out.ipynb` — é lá que se lê o erro quando uma rodada falha (o `execute()` **engole a
  exceção** e segue para a próxima etapa).
- **Notebook isolado headless** (o `cwd` importa: todo caminho nos notebooks é relativo à pasta do próprio
  notebook, e é isso que o papermill garante com `cwd=`):
  ```
  cd Model && ~/anaconda3/envs/torch-gpu/bin/jupyter nbconvert --to notebook --execute Analysis.ipynb --output /tmp/out.ipynb
  ```
- **Acompanhar um treino**: `Model/progress.json` é reescrito a cada época (`epoch`, `train_loss`,
  `val_loss`, `train_iou`, `val_iou`, `lr`).
- **GPU**: `ModelNetwork` cai para CPU sozinho se `cuda` não estiver disponível, mas nenhum notebook é
  utilizável na prática sem GPU (volumes 128³, `batch_size=2`).

## Fluxo de dados (o contrato que amarra tudo)

```
Synthetic/Generate.ipynb  ─┐
Marlim/Generator.ipynb    ─┼→ Dataset/<ds>/original/{images,masks}/*.npy   (volumes crus, float, eixo (x,z,y))
Marlim/Generator2.ipynb   ─┤
(dataset_wu já vem pronto)─┘
                             ↓  Dataset/<ds>/Format.ipynb   (p01/p99 global → clip → [0,1];  tiling opcional)
                          Dataset/<ds>/{images,masks}/*.npy  (ou tiles/)  +  Dataset/DataBase.csv
                             ↓  Model/Analysis.ipynb   (treino)
                          Model/Backup/model_<n>/{info.json, model.pth, train.png, predictions/}
                             ↓  Marlim/Predict.ipynb   (SlidingWindow na janela do modelo)
                          Model/Backup/model_<n>/marlim/patch_<id>/masks/*.dat   (probs sigmoid float32)
                             ↓  Marlim/Analysis.ipynb   (FaultStickExtractor + FaultComparer)
                          Marlim/files/comparisons/*.png  +  métricas stick-a-stick contra o especialista
```

Pontos não óbvios desse contrato:

- **`Task/info.json` é a config viva de uma rodada.** Todo notebook o lê no início. Chaves: `network`,
  `dataset`, `img_size`, `lr`, `loss`, `batch_size`, `scheduler`, `dropout`, `num_filters` (e `augmentations`
  opcional). O `Format.ipynb` **reescreve** o arquivo carimbando `dataset` com o nome da própria pasta.
- **`img_size` liga/desliga o tiling.** `img_size: null` → `Format.ipynb` gera o `DataBase.csv` sobre os
  volumes 128³ inteiros e **encerra com `sys.exit`** (sob papermill isso vira `PapermillExecutionError` —
  esperado). `img_size: [d,h,w]` → a classe `TilesBuilder` corta cada volume em blocos **sem sobreposição**
  em `<ds>/tiles/{images,masks}` e o `DataBase.csv` aponta para lá; a pasta `tiles/` é **reescrita a cada
  rodada**. O `task.json` atual roda uma campanha de janelas pequenas (8³–64×128×128) sobre `dataset_wu`.
- **`Dataset/DataBase.csv` é global e único** — cada `Format.ipynb` o sobrescreve com **caminhos absolutos**
  (`id,img_min,img_max,img_mean,img_std,msk_min,msk_max,shape,img_path,mask_path`). É o único acoplamento
  entre dataset e treino — o `Analysis.ipynb` lê só esse CSV e **sobrescreve o `IMG_SIZE`** com
  `df['shape'].iloc[0]`, ignorando o do `info.json`. Rodar dois
  experimentos em paralelo corrompe a rodada — o `Task/index.py` e o `Searcher` são sequenciais por isso.
- **Normalização é p01/p99 global do dataset inteiro**, calculada carregando todos os volumes de uma vez no
  `Format.ipynb`. Os **mixes não renormalizam**: concatenam os `images/`/`masks/` já formatados dos dois
  datasets de origem e só escrevem o `DataBase.csv` (as duas origens precisam já ter sido formatadas).
- **Split**: `train_test_split(random_state=42)` em dois estágios, `TEST_SIZE = VAL_SIZE = 1/22` (o
  `VAL_SIZE` ainda é reescalado por `1/(1-TEST_SIZE)` para valer sobre o resto) — 220 volumes viram
  **200/10/10**. Não é 10%: val e test têm **10 volumes cada**, e é sobre esses 10 que sai todo `val_iou`.
- **`Model/Backup/model_<n>` é numerado por varredura** (maior índice + 1). O `info.json` salvo tem
  `trainer` (`loss`/`scheduler`/`epochs`/`batch_size`/`val_iou`/`test_iou`), `processing` (o `OPTIONS` da
  rodada + `n_images`), `model` (os kwargs exatos do `ModelNetwork`) e `iou`. É o bloco `model` que
  `Predict.ipynb` e `Marlim/Predict.ipynb` usam para reinstanciar a rede antes de carregar `model.pth`.
- **O `model.pth` não é só o peso**: é um dict com `model` (state_dict), `optimizer`, `timestamp` e
  **`history`** — a lista época-a-época completa. O `progress.json` só guarda a última época e é
  sobrescrito pela rodada seguinte, então `model.pth['history']` é o **único** registro durável da curva de
  treino de um modelo. Para ranquear modelos de verdade, leia dali, não do `info.json` (ver a armadilha do
  `val_iou` abaixo).

Detalhes finos do contrato Marlim (eixos, offset da inline anotada, patches `.dat` já em `[0,1]`) estão nas
memórias do projeto — ver `marlim-pipeline-contract`.

## Arquitetura por diretório

### `Task/` — orquestrador
`index.py` (39 linhas) é todo o runner: `task.json` (fila de experimentos) → `info.json` (rodada atual) →
papermill roda `Format.ipynb` + `Analysis.ipynb`. Editar `task.json` é como se agenda uma bateria de treinos.

### `Dataset/` — datasets e formatação
Cada `dataset_*/` tem seu próprio `Format.ipynb` (copiado, genérico via `cwd` — a pasta descobre o próprio
nome com `Path().resolve().name`). Os `Format.ipynb` **duplicam de propósito** helpers como `getFiles` /
`formatAxis` para serem autocontidos. `marlim/` guarda os patches reais do bloco; `marlim_opt` e
`dataset_marlim_opt_2` são os sintéticos calibrados para Marlim.

**O que está de fato em disco hoje** (nada disso é versionado, então confira antes de rodar):
`dataset_74` 220, `dataset_74_2000` 2000 e `dataset_wu` 220 estão gerados **e formatados**;
`dataset_marlim_opt_2` tem os 660 em `original/` mas **ainda não foi formatado**; os quatro mixes
(`dataset_74_mar`, `dataset_wu_mar`, `dataset_74_wu`, `dataset_74_wu_mar`) estão **vazios** — só existem
como `Format.ipynb`, e o mix se materializa quando o `Format` roda. Em `marlim_opt`, `original/` tem 110
volumes e `images/` tem 220: o formatado é de uma geração anterior e **não corresponde** ao `original/`
atual — reformate antes de usar.

### `Model/` — treino e inferência
`Analysis.ipynb` é o único notebook que produz modelos. Folder-modules importados **com o cwd em `Model/`**:
- `Network/index.py` — `ModelNetwork` é a fachada: escolhe a arquitetura por nome em `get()` (`'standard'`
  → `UNet3D`, `'unet3d_v2'` → `Unet3D_V2`, `'segresnet'` → MONAI `SegResNet`, `'resaceunet'` → `ResACEUnet`,
  `'resaceunet_grva'` → `ResACEUnet_GRVA`, `'macnn'` → `MACNN`), move para o device, cria
  `AdamW(lr, weight_decay=1e-4)` e a
  métrica `BinaryJaccardIndex`. Arquiteturas concretas em `Network/types/*.py`. Adicionar uma rede =
  arquivo em `types/` + um `if` em `get()`.
- `Network/types/ResACEUnet_GRVA.py` — funde a `Unet3D_V2` com a `ResACEUnet` **rebalanceando a capacidade
  para a borda**, que é onde 92% do erro medido mora (ver a memória `wu-erro-de-borda-e-capacidade`).
  Reusa `ResACEBlock`/`ACE3D`/`AttentionGate3D` de `resaceunet.py`. Três decisões que a distinguem:
  `poolPlan` reduz **z primeiro** (é o eixo contínuo da falha — espessura 7 vox contra 2 em x),
  as larguras param de dobrar no fim (o gargalo tem teto de IoU 0.14), e a decisão final é tomada por uma
  **cabeça de fusão em resolução plena**, com saída de tensor único (não quebra o `Trainer`).
- `Network/types/MACNN.py` — a rede do `article.pdf` (Gao et al., 2022, *GEOPHYSICS* 87/1): U-Net de três
  níveis em que **cada encoder é refinado por atenção espaço-canal construída a partir dos três encoders**,
  não só do seu nível, antes de concatenar no decoder. Blocos de tripla convolução (a do meio dilatada).
  O artigo é ambíguo em dois pontos e a memória `macnn-implementacao` guarda como foram resolvidos — não
  os re-derive. Backbone com 11.7012 M de parâmetros contra os 11.7 M publicados.
- `Losses/index.py` — `Losses(name, multiclass)` via `__new__` + dict `options`: `'cross_entropy'`
  (`BCEWithLogitsLoss` no binário), `'dice_focal'` (`monai.losses.DiceFocalLoss`, sigmoid), `'focal'`
  (`monai.losses.FocalLoss`, recebe `alpha=0.93`) e `'smooth_dice'` (o dice suavizado do MACNN, eq. 7 do
  artigo: `1 - (2*sum(p*g)+1)/(sum(p)+sum(g)+1)`, **soma global sobre o lote**, suavizador 1 — equivale ao
  `monai DiceLoss(smooth_nr=1, smooth_dr=1, batch=True)`). Todos forçam `autocast(enabled=False)` e float32.
- `EarlyStopping/index.py` — `ready(model, metric)` acumula paciência e guarda `best_state`;
  `restore_best()` no fim. Usado com `mode='max'` sobre `val_iou`, `patience=15`.
- `Transforms/index.py` — `Compose(config)` monta a pilha de augmentation 3D a partir de
  `OPTIONS['augmentations']` (lista de dicts). **Só o dataset de treino recebe augmentação.**
- `utils/index.py` — `getFiles`, `setFolder`, `showTile`. `utils/Plotter/index.py` — `Plotter(metrics, ...)`,
  gráfico de barras para comparar métricas entre modelos.
- `Predict.ipynb` — carrega um `model_<n>` e roda sobre um dataset inteiro (recorta os volumes na janela do
  modelo na hora). `PostProc.ipynb` — varre `Backup/*/info.json` e compara experimentos.
- `Augmentor/index.py` é uma **cópia morta** de `Transforms/index.py`; o treino importa `Transforms`. Não é
  cópia exata — o `Transforms/` tem `Contrast`, `RandomShift` e `RandomZoom` que o `Augmentor/` não tem.
- A célula 0 do `Analysis.ipynb` importa `albumentations` e o pacote de detecção do `torchvision`
  (`FasterRCNN`, `AnchorGenerator`, `nms`) — **nada disso é usado**, mas o import está no topo: sem
  `albumentations` instalado o notebook morre na primeira célula.

**Estado dos experimentos** (19 modelos em `Backup/`, todos `dataset_wu`, `dice_focal`, `plateau`,
`lr=1e-3`, `num_filters=32`): o melhor por `val_iou` real é o **model_19** (`resaceunet_grva`, 0.8153) —
a rede nova bate os `unet3d_v2` a 128³, que se agrupam em 0.78–0.80. `resaceunet` (model_13, 0.7789) e
`segresnet` (model_14, 0.7697) ficaram atrás. Os de janela pequena (model_16/17/18) **não são comparáveis**
— ver a armadilha do vazamento de tiles.

### `Synthetic/` — gerador sintético geral
`index.py` → `SyntheticGenerator`: `genReflectivity` → `applyFolding` → `applyShearing` → `applyFaulting` →
`applyWavelet` → `applyNoise` → `crop` (margem 64 absorve as dobras extremas). `get()` monta um volume,
`set(options)` sobrescreve em bloco, `dataset(n, outputDir)` gera em paralelo. `Utils/index.py` traz
`formatAxis` (convenção de eixos do projeto), `showTile`, `showSteps`. `Generate.ipynb` aplica um preset e
gera; `Adjustments.ipynb` é um notebook de exploração de parâmetros (com `SAVE_DATA=False` de guarda).

### `Marlim/` — o bloco real
Notebooks **autocontidos por decisão de projeto**: as classes vivem dentro dos `.ipynb`, não há `.py` nesta
pasta (única exceção: `sys.path.append('../Model')` no `Predict.ipynb`, para reusar `ModelNetwork`).
- `Generator.ipynb` — `MarlimSyntheticGenerator`, classe própria calibrada contra medições do bloco.
- `Generator2.ipynb` — reproduz Marlim com o `SyntheticGenerator` **global** sem alterá-lo; cada região
  (`calmo`/`falhado`/`morto`) é um dict de opções + `MarlimRegion` (gain/jitter/ramp/clip). → `dataset_marlim_opt_2`.
- `Predict.ipynb` — `SlidingWindow` (anda na janela do `info.json['model']['img_size']`, soma com peso Hann,
  knob `OVERLAP`) + `MarlimPredictor`. `BASE_PATH` aponta para `../Model/Backup` **ou** `../Marcia`.
- `Analysis.ipynb` — `FaultStickExtractor` (mapa de probabilidade → falhas retas estilo especialista, knob
  `tolerance` 0..1) e `FaultComparer` (`show()`, `metrics()`), comparando na inline central contra o
  `<pid>_interpretado.png`. Calibração e métrica stick-a-stick: ver memórias `marlim-stick-extractor-v5/v6`.

### `Searcher/` — otimização metaheurística
`Searcher/README.md` é o guia completo da campanha — **leia antes de mexer**. `Analysis.ipynb` procura a
config do `Synthetic/index.py` que maximiza o IoU no `dataset_wu` de um modelo treinado só em sintético;
cada avaliação é o **pipeline inteiro** rodado por papermill (4–9 h). O otimizador é `Nature/` (framework de
metaheurísticas, DE auto-adaptativa `lshade`); a memória da campanha fica em `Searcher/files/`.
`Searcher/Nature/` **não é mais um repositório git aninhado vivo** (não tem `.git`) — ele entra no índice do
repo externo como um gitlink opaco, então seu conteúdo não é versionado aqui; não tente `git add` lá dentro.

### `Marcia/` — entregas
Não é código: cópias de modelos selecionados (`info.json`, `model.pth`, `marlim/`, `predictions/`,
`train.png`) para compartilhar — hoje 11 (`model_1..14`, com buracos). Substituiu a pasta `Lucas/`
(removida). É uma **cópia**, não um link: um `model_<n>` daqui pode divergir do de `Model/Backup/`, e o
`Marlim/Predict.ipynb` roda sobre os dois via `BASE_PATH`.

## Convenções

- **O guia de estilo canônico é `CODE_STYLE.md` na raiz** e vale para o repositório inteiro — leia antes de
  gerar ou refatorar código. Em uma linha: sem type hints, sem docstrings, sem underscore inicial;
  `camelCase` para o que é do projeto / spelling da lib para o que é dela / notação do campo para
  math-física; um vocabulário fixo de métodos (`update`/`get`/`set`/`info`/`plot`/`setup`/`start`/`stop`);
  `__init__` leve e `update()` pesado; `if` sequencial + `return` em vez de `elif`; diferença vira dado;
  `=` alinhados; comentário = **uma linha MAIÚSCULA em português** acima da definição, só onde o código não
  se explica.
- **Idioma**: código e nomes em inglês; markdown de notebook, comentários, mensagens, títulos de gráfico e
  commits em **português** (seções de notebook em `# MAIÚSCULAS`).
- **Folder-as-module**: todo componente é `Pasta/index.py`, sem `__init__.py` — imports pelo caminho
  completo (`from Network.index import ModelNetwork`), dependentes do cwd.
- **Notebooks**: classe definida e usada na mesma célula; toda célula termina com prova visual (DataFrame,
  figura ou print curto de shapes). Alguns notebooks **duplicam helpers de propósito** para serem
  autocontidos — não "consolide" isso num módulo compartilhado sem combinar antes.

## Armadilhas conhecidas

- **`.gitignore` ignora `*.npy`, `*.pth`, `*.dat`, `*.png`/`*.jpg`, `*.zip`.** Nenhum dado, peso ou figura
  está versionado — só notebooks, `.py`, JSONs e CSVs. Clonar o repo **não** dá um projeto executável.
- **`Task/index.py` engole exceções** (`print` e segue). Uma rodada pode "terminar" sem ter treinado nada —
  confira `Task/logs/*_out.ipynb` e se apareceu um `model_<n>` novo em `Model/Backup/`.
- **Tiling degrada em silêncio na inferência.** Uma rede treinada em janela pequena (32³, 64³) recebendo o
  tile 128³ do Marlim: o `Unet3D_V2` é FCN e aceita, mas o GroupNorm normaliza sobre todo o volume. Use a
  janela do modelo — `Marlim/Predict.ipynb` tem `SlidingWindow` para isso; `Model/Predict.ipynb` recorta na
  hora.
- **`seed_everything()` liga `cudnn.deterministic=True`, e conv 3D dilatada não tem algoritmo
  determinístico barato.** O `MACNN` pede 6.75 GiB de workspace a mais nesse modo e estoura a P6000 (24 GiB).
  Há uma célula no `Analysis.ipynb`, logo depois do `OPTIONS`, que relaxa a flag **só quando
  `network == 'macnn'`** — as outras redes seguem determinísticas. Não remova sem trocar a rede de janela.
- **`F.adaptive_max_pool3d(x, 1)` é 265× mais lento que `x.amax(dim=(2,3,4))`** para resultado bit a bit
  idêntico (0.37 s contra 0.0014 s num tensor 2×32×128³). Vale para qualquer rede deste repo.
- **`Trainer` chama `self.scheduler.step(val_loss)` incondicionalmente.** Certo para `scheduler='plateau'`;
  com `'cosine'` o `CosineAnnealingWarmRestarts` interpreta o argumento como *epoch*. Todas as configs usam
  `plateau`.
- **O `val_iou` do `info.json` é o da ÚLTIMA época, não o do melhor modelo.** O `Analysis.ipynb` grava
  `trainer.history[-1]['val_iou']`, mas o `EarlyStopping.restore_best()` já devolveu os **pesos da melhor
  época** antes disso — ou seja, o `model.pth` é melhor do que o número que o acompanha (medido: até
  +0.005 de IoU, model_1 vale 0.8008 e o `info.json` diz 0.7961). Ranquear modelos pelo `info.json`
  compara ruído de última época; o valor real está em `max(h['val_iou'] for h in model.pth['history'])`.
  O `test_iou`, esse sim, é medido depois do `restore_best` e corresponde aos pesos salvos.
- **`epochs=100` e `use_amp=False` estão fixos na célula do `Trainer`**, não no `info.json`. O teto de 100
  **corta o treino antes da convergência**: dos 19 modelos em `Backup/`, **11 chegaram às 100 épocas sem o
  early stopping disparar, e nos 11 o melhor `val_iou` caiu da época 90 em diante** (quem encerrou foi o
  teto, não a paciência) — inclusive os melhores: model_19 (época 97, 0.8153), model_1 (99, 0.8008),
  model_5 (99, 0.8004), model_9 (90, 0.8002). Os outros 8 pararam sozinhos por paciência. Subir o teto é o
  ganho de IoU mais barato disponível; custa só GPU.
- **O ramo multiclasse está morto.** `MULTICLASS = False` fixo; `getClasses(...)` é chamado no
  `Analysis.ipynb` (célula final) e **não existe em lugar nenhum** — só roda se `network.multiclass`. Ligar
  multiclasse exige definir essa função.
- **`Model/Augmentor/index.py` é código morto** (quase idêntico a `Transforms/index.py`). Edite `Transforms/`.
- **Os defaults do `Transforms/index.py` destroem a física destes dados.** `Rot90` usa `axes=[1,2]` e
  `Transpose` inclui pares com o eixo 1 — os dois **giram o eixo vertical z para lateral** (medido: 10 de 12
  amostras saem com o eixo contínuo trocado). Só a simetria do **plano lateral (eixos 0 e 2)** é válida:
  `RandomFlip(axes=[0,2])`, `Rot90(axes=[0,2])`, `Transpose(axes=[[0,2]])`. Nenhum config em `task.json` usa
  augmentation hoje.
- **Tiling vaza entre treino e teste.** O `TilesBuilder` grava `<volume>_tile_NNNN.npy` numa pasta única e o
  `Analysis.ipynb` faz `train_test_split` **por tile**, sem agrupar por volume de origem — tiles do mesmo
  volume caem nos dois lados. Por isso model_16 (0.7916) e model_17 (0.7863) parecem bater o model_1 a 128³
  (0.7784) e são justamente os piores no Marlim real. Comparar tile com volume é comparar coisas diferentes.
- **Compartilhar estado global entre rodadas**: `Task/info.json`, `Dataset/DataBase.csv`,
  `Dataset/dataset_trial/` e a GPU são recursos únicos — nunca rode dois experimentos ao mesmo tempo.
