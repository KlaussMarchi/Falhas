# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

`Falhas` é um projeto de **detecção de falhas sísmicas em volumes 3D** (segmentação binária voxel-a-voxel,
`classes=1`, sigmoid + IoU). O modelo é treinado em **dados sintéticos** e avaliado no **bloco real de Marlim**
contra a interpretação de um especialista. Não há aplicação, servidor nem pacote instalável: tudo é notebook
Jupyter + folder-modules (`Pasta/index.py`), orquestrados por papermill.

Três fontes de dado convivem:
- **sintético próprio** (`Synthetic/` → `Dataset/dataset_74*`) — gerador paramétrico geral;
- **sintético calibrado para Marlim** (`Marlim/Generator.ipynb` → `Dataset/marlim_opt/`);
- **Wu** (`Dataset/dataset_wu/`) — dataset público de referência, já vem pronto em `original/`;
- e os **mixes** (`dataset_74_mar`, `dataset_wu_mar`, `dataset_74_wu`, `dataset_74_wu_mar`), que só concatenam
  dois datasets já formatados.

## Comandos

- **Ambiente: `~/anaconda3/bin/python`** (conda `base`, primeiro no `PATH`). Não há `requirements.txt` nem
  `environment.yml`. Pacotes usados: `torch`, `monai`, `torchmetrics`, `papermill`, `numpy`, `pandas`,
  `scipy`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `tqdm`, `Pillow`.
- **Não há linter, build nem testes automatizados.** A validação é rodar o notebook (célula a célula ou
  headless) e olhar as figuras/métricas que cada célula imprime.
- **Pipeline completo de treino** (o jeito normal de rodar um experimento):
  ```
  cd Task && ~/anaconda3/bin/python index.py
  ```
  Lê `Task/task.json` (lista de configs), escreve cada uma em `Task/info.json` e executa, via papermill,
  `../Dataset/<dataset>/Format.ipynb` e depois `../Model/Analysis.ipynb`. Saídas executadas vão para
  `Task/logs/<nome>_out.ipynb` — é lá que se lê o erro quando uma rodada falha (o `execute()` engole a
  exceção e segue para a próxima etapa).
- **Rodar um notebook isolado headless** (o `cwd` importa: todo caminho nos notebooks é relativo à pasta do
  próprio notebook, e é isso que o papermill garante com `cwd=`):
  ```
  cd Model && ~/anaconda3/bin/jupyter nbconvert --to notebook --execute Analysis.ipynb --output /tmp/out.ipynb
  ```
- **Acompanhar um treino em andamento**: `Model/progress.json` é reescrito a cada época
  (`epoch`, `train_loss`, `val_loss`, `train_iou`, `val_iou`, `lr`).
- **GPU**: `ModelNetwork` cai para CPU sozinho se `cuda` não estiver disponível, mas nenhum notebook é
  utilizável na prática sem GPU (volumes 128³, `batch_size=2`).

## Fluxo de dados (o contrato que amarra tudo)

```
Synthetic/Generate.ipynb  ─┐
Marlim/Generator.ipynb    ─┼→ Dataset/<ds>/original/{images,masks}/*.npy   (volumes crus, float)
(dataset_wu já vem pronto)─┘
                             ↓  Dataset/<ds>/Format.ipynb
                          Dataset/<ds>/{images,masks}/*.npy  +  Dataset/DataBase.csv
                             ↓  Model/Analysis.ipynb  (treino)
                          Model/Backup/model_<n>/{info.json, model.pth, train.png, predictions/}
                             ↓  Marlim/Predict.ipynb
                          Model/Backup/model_<n>/marlim/patch_<id>/masks/*.dat
                             ↓  Marlim/Analysis.ipynb
                          Marlim/files/comparisons/*.png  +  métricas contra o especialista
```

Pontos não óbvios desse contrato:

- **`Task/info.json` é a config viva de uma rodada.** Todo notebook o lê no início (`../Task/info.json` ou
  `../../Task/info.json`). Chaves: `network`, `dataset`, `img_size`, `lr`, `loss`, `batch_size`, `scheduler`,
  `dropout`, `num_filters`. O `Format.ipynb` **reescreve** o arquivo carimbando `dataset` com o nome da
  própria pasta, e o `Analysis.ipynb` acrescenta `img_size` e `n_images` só em memória, antes de salvar no
  `info.json` do modelo. `img_size: null` faz o `Format` gerar o `DataBase.csv` e **encerrar com `sys.exit`**
  antes da etapa de tiles — é o modo "sem tiling", que é como todos os datasets atuais rodam.
- **`Dataset/DataBase.csv` é global e único** — cada `Format.ipynb` o sobrescreve com **caminhos absolutos**.
  É o único acoplamento entre o dataset e o treino: o `Analysis.ipynb` lê só esse CSV e tira o `IMG_SIZE` de
  `df['shape'].iloc[0]`. Rodar dois experimentos em paralelo corrompe a rodada — o `Task/index.py` é
  sequencial por isso.
- **Normalização é p01/p99 global do dataset inteiro**, calculada carregando todos os volumes de uma vez em
  `Format.ipynb`, clipando e mapeando para `[0, 1]`. Os **datasets de mix não renormalizam nada**: eles
  concatenam os `images/`/`masks/` já formatados dos dois datasets de origem e só escrevem o `DataBase.csv`.
  Consequência: os dois datasets de origem precisam já ter sido formatados antes.
- **Split**: `train_test_split` com `random_state=42`, `TEST_SIZE = VAL_SIZE = 1/22` — 220 volumes viram
  200/10/10.
- **`Model/Backup/model_<n>` é numerado por varredura** (maior índice existente + 1). O `info.json` salvo tem
  três blocos: `trainer` (loss/scheduler/epochs/ious), `processing` (o `OPTIONS` completo da rodada) e
  `model` (os kwargs exatos do `ModelNetwork`) — é esse bloco `model` que `Predict.ipynb` e
  `Marlim/Predict.ipynb` usam para reinstanciar a rede antes de carregar `model.pth`.

## Arquitetura por diretório

### `Model/` — treino e inferência
`Analysis.ipynb` é o notebook de treino (o único que produz modelos). Os folder-modules ao redor são
importados **com o cwd em `Model/`**:
- `Network/index.py` — `ModelNetwork` é a fachada: escolhe a arquitetura por nome em `get()`
  (`'standard'` → `UNet3D`, `'unet3d_v2'`, `'segresnet'` → MONAI `SegResNet`, `'resaceunet'`), move para o
  device, cria o `AdamW(lr, weight_decay=1e-4)` e a métrica `BinaryJaccardIndex`/`MulticlassJaccardIndex`.
  Arquiteturas concretas ficam em `Network/types/*.py`. Adicionar uma rede = arquivo em `types/` + um `if`
  em `get()`.
- `Losses/index.py` — `Losses(name, multiclass)` via `__new__`, wrappers finos sobre `monai.losses` que
  forçam `autocast(enabled=False)` e float32.
- `EarlyStopping/index.py` — `ready(model, metric)` acumula paciência e guarda `best_state`;
  `restore_best()` no fim do treino. Usado com `mode='max'` sobre `val_iou`, `patience=15`.
- `Transforms/index.py` — `Compose(config)` monta a pilha de augmentation 3D a partir de uma lista de dicts
  vinda de `OPTIONS['augmentations']`; cada transform é uma subclasse de `Transform3D` com `apply(img, mask)`
  e probabilidade `p`. **Só o dataset de treino recebe augmentação.**
- `utils/index.py` — `getFiles`, `setFolder`, `showTile` (as três fatias ortogonais no meio do volume, com
  overlay de máscara). `showTile` é copiado de propósito dentro de vários notebooks para mantê-los
  autocontidos.
- `Predict.ipynb` — carrega um `model_<n>` e roda sobre um dataset `.npy` inteiro (IoU + overlays
  GT azul / predição vermelho / acerto verde). `PostProc.ipynb` — varre `Backup/*/info.json` e compara
  experimentos em tabela/gráfico.

### `Marlim/` — o bloco real
Notebooks **autocontidos por decisão de projeto**: as classes vivem dentro dos `.ipynb`, não há `.py` nesta
pasta (a única exceção é `sys.path.append('../Model')` no `Predict.ipynb`, para reusar `ModelNetwork`).
- Os patches reais vivem em `Dataset/marlim/patch_{1200,1300,1400,2600}/*.dat`: `float32`, 128³, **já em
  `[0, 1]` — nenhum pré-processamento extra antes de predizer**. `patch_metadata.json` descreve o slab
  original `(64, 1601, 2240)`, `overlap_voxels=32`, `stride=64`.
- `MarlimBlock.reconstruct()` remonta o slab colando **só o miolo** de cada patch (descarta as bordas de
  overlap); `center_section(pid)` devolve a inline central `(1601, 2240)` com cache em `files/cache/`.
- `Generator.ipynb` — `MarlimSyntheticGenerator`, calibrado contra medições do bloco real (bandas em z,
  espectro por profundidade, geometria das falhas). Escreve tiles **crus** em `Dataset/marlim_opt/original/`;
  a normalização é do `Format.ipynb` do dataset, como em qualquer outro.
- `Predict.ipynb` — para cada modelo em `../Model/Backup/`, prediz os patches e grava probabilidades
  (sigmoid, float32) em `Backup/<modelo>/marlim/patch_<id>/masks/*.dat`.
- `Analysis.ipynb` — `FaultStickExtractor` (mapa de probabilidade → falhas retas estilo especialista, com
  `tolerance` de 0=conservador a 1=permissivo interpolando os hiperparâmetros internos) e `FaultComparer`
  (`show(showType='seismic'|'mask'|'both')` e `metrics()`), comparando na inline central contra
  `files/patches/<pid>/<pid>_interpretado.png` (máscara booleana, threshold 128).

### `Synthetic/` — gerador sintético geral
`index.py` → `SyntheticGenerator`: reflectividade estratigráfica → dobramento → cisalhamento → falhamento →
convolução com wavelet → ruído → crop da margem (`margin=64` existe para absorver as dobras extremas).
Todos os parâmetros são atributos do `__init__` documentados em português com o efeito de subir/descer cada
um; `set(options)` sobrescreve em bloco (é assim que o `Generate.ipynb` aplica um preset otimizado) e
`dataset(n, outputDir)` gera em paralelo. `Utils/index.py` traz `formatAxis` (reordena os eixos para a
convenção do resto do projeto), `showTile`, `showSteps`.

### `Task/` — orquestrador
`index.py` (39 linhas) é todo o "runner": `task.json` (lista) → `info.json` (rodada atual) → papermill.
`task.json` é a fila de experimentos; editá-lo é como se agenda uma bateria de treinos.

### `Lucas/` — entregas
Não é código: cópias de resultados de modelos selecionados (`info.json`, `synthetic.json`, `predictions/`,
`marlim/`) exportadas para compartilhar. Nomenclatura `model_<origem>_<rede>_<versão>`
(`sy`=sintético, `wu`=Wu, `mix`=misto; `u3d`=UNet3D, `seg`=SegResNet).

### `Searcher/` — otimização metaheurística
Tem **CLAUDE.md próprio** (`Searcher/CLAUDE.md`) — leia-o antes de mexer ali. Dois pontos que afetam quem
está na raiz:
- `Searcher/Nature/` é um **repositório git próprio aninhado** neste; commits do framework vão no repo de
  dentro.
- O conteúdo antigo de `Searcher/` (`Dataset/`, `Model/`, `Model/Backup/model_*`) foi apagado da árvore de
  trabalho mas continua rastreado — é a origem do `git status` gigante de deleções. **É intencional; não
  "restaure" esses arquivos.**

## Convenções

- O guia de estilo canônico do autor é **`Searcher/CODE_STYLE.md`** e vale para o repositório inteiro — leia
  antes de gerar ou refatorar código. Resumo do que mais aparece aqui: sem type hints e sem docstrings;
  `camelCase` para variáveis/funções, `PascalCase` para classes/pastas, `UPPER_SNAKE` para constantes,
  `snake_case` para chaves de dict/JSON; `=` alinhados em blocos de atribuições relacionadas; expressão
  completa em uma linha (~110 colunas), inclusive chamadas com muitos kwargs; `if` sequencial + `return` em
  vez de `elif/else` para despacho; `__init__` leve e `update()` pesado; vocabulário fixo de métodos
  (`update`/`get`/`set`/`info`/`plot`/`print`/`setup`/`start`/`stop`); todo `plot()` aceita `save=None`.
- **Idioma**: código e nomes em inglês; markdown de notebook, comentários de decisão, mensagens ao usuário,
  títulos de gráfico e commits em **português** (seções de notebook em `# MAIÚSCULAS`).
- **Folder-as-module**: todo componente é `Pasta/index.py`. Não há `__init__.py`, não há barril — os imports
  são pelo caminho completo (`from Network.index import ModelNetwork`) e dependem do cwd.
- **Notebooks**: classe definida e usada **na mesma célula**, separadas por duas linhas em branco; toda
  célula termina com prova visual (DataFrame, figura ou print curto de shapes).
- Alguns notebooks (Marlim, Format) **duplicam de propósito** helpers como `showTile`/`getFiles` para serem
  autocontidos. Não "consolide" isso em um módulo compartilhado sem combinar antes.

## Armadilhas conhecidas

- **`.gitignore` ignora `*.npy`, `*.pth`, `*.dat`, `*.png`.** Ou seja: nenhum dado, peso ou figura está
  versionado — só notebooks, `.py`, JSONs e CSVs. Clonar o repo não dá um projeto executável; os datasets e
  os `Backup/*/model.pth` precisam vir de fora.
- **`Trainer` chama `self.scheduler.step(val_loss)`** incondicionalmente. Isso está certo para
  `scheduler='plateau'`, mas com `'cosine'` o `CosineAnnealingWarmRestarts` interpreta o argumento como
  *epoch* — todas as configs atuais usam `plateau`.
- **`getClasses(...)` é chamado no `Analysis.ipynb` no ramo multiclasse e não existe em lugar nenhum.** O
  projeto é binário (`MULTICLASS = False`), então o ramo nunca roda; ligar multiclasse exige definir a função.
- **Em `Losses.binary`, `'focal'` e `'dice_focal'` apontam para a mesma `BinaryDiceFocalLoss`** (a classe é
  definida duas vezes no arquivo; a segunda vence). Não há focal binária pura.
- **`Model/Augmentor/index.py` é uma cópia antiga e quase idêntica de `Model/Transforms/index.py`**; o
  notebook de treino importa de `Transforms`. Edite `Transforms/`; `Augmentor/` está morto.
- **`Task/index.py` engole exceções** (`print` e segue). Uma rodada pode "terminar" sem ter treinado nada —
  confira `Task/logs/*_out.ipynb` e o índice novo em `Model/Backup/`.
- `Trainer` roda com `use_amp = False` (AMP desligado de propósito) e `epochs=100` fixo na célula do
  notebook, não no `info.json`.
