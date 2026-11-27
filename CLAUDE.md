# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Segmentação de falhas geológicas em sísmica 3D: gera volumes sintéticos, treina redes 3D neles e prediz nos blocos reais
de `Blocks/` — Marlim (com a interpretação do especialista, onde o resultado é medido), Iracema e F3 (sem interpretação). Não
há build, lint nem suíte de testes — a verificação é rodar o notebook de cima para baixo e comparar os números.

## Ambiente e execução

Tudo roda no ambiente conda `torch-gpu` (torch 2.7.1, cu128 como no `requirements.txt`, monai 1.5.2, torchmetrics, albumentations, papermill,
opencv, scikit-image, psutil). O python do `base` **não tem torch** — nunca rode um script deste projeto sem ativar:

```bash
conda activate torch-gpu           # ou: conda run -n torch-gpu python ...
```

- **Campanha de treinos (o caminho normal):** edite `Task/task.json` (lista de rodadas, cada uma no formato de
  `Task/info.json`) e rode `cd Task && python index.py`. Para cada rodada ele grava `Task/info.json`, executa
  `Dataset/<dataset>/Format.ipynb` (pulado quando o `DataBase.csv` já existe, `img_size` é `null` e a coluna
  `scaling` dele bate com a da rodada) e depois
  `Model/1 - Model.ipynb` uma vez por trial (`n_trials`; o trial vai no `info.json` e a semente é `42 + trial`), via
  papermill, com a saída em `Task/logs/<nome>_out.ipynb`. No fim da rodada, roda o `Model/3 - Predict.ipynb` só nos
  modelos que ela gravou (uma cópia com a lista trocada por parâmetro): a matriz de confusão, a acurácia, a ROC e as outras
  métricas de teste vão para o `test.<dataset>` do `info.json` de cada um. O kernel é `python3`, que resolve para o do `torch-gpu`.
- **Um notebook só:** abra no Jupyter/VS Code com o kernel `torch-gpu`, ou
  `papermill <nb> /tmp/out.ipynb -k python3 --cwd <pasta do nb>` — todo caminho dentro de um notebook é relativo à
  **pasta dele**, então o `cwd` errado quebra tudo.
- GPU: desde 21/10/2026 o projeto roda na `mint-alien` (RTX 5070 de 12 GB). Nela a `dbrnet` com `num_filters` 32 e
  lote 2 estoura a memória mesmo com AMP: use lote 1 (10,9 GB); nf 16 com lote 2 usa 10,7 GB. Só um job de GPU por vez.
  A máquina antiga do usuário é uma Quadro P6000 de 24 GB (Pascal: AMP não acelera). Nas duas o float64 roda a uma
  fração do float32. O treino roda com AMP só quando o `info.json` tem `"amp": true` (padrão desligado), `epochs` do
  `info.json` (padrão 100) e early stopping sobre `val_iou` com `patience` do `info.json` (padrão 15). Uma rodada custa
  horas; nunca re-execute um treino para "verificar" sem pedir.
- `.gitignore` exclui `*.npy`, `*.npz`, `*.pkl`, `*.pth`, `*.dat`, `*.zip`, `Documents/Reference` e `Synthetic/files/`
  (só alguns arquivos antigos dela seguem versionados): datasets, blocos reais, pesos, perfis e experimentos são locais.
  Os PNG (curvas, matrizes, figuras dos blocos) e os `info.json` dos modelos entram no git. Um clone limpo precisa
  regerar os dados pelo `SyntheticGenerator` com o `synthetic.json` de cada dataset (o `dataset_74`: semente base 121958,
  220 tiles) + `Format.ipynb`.
- Disco: os patches do Marlim ocupam 29 GB; o Iracema inteiro teria 302 GB, mas só os 896 tiles da inline 1375 estão
  em disco (7,1 GB), e o F3 ocupa 11 GB. Predizer um modelo no Marlim grava 0,9 GB do
  `predicted.npy` por patch, mais ~7,3 GB de máscaras com `SAVE_TILES = True`; as máscaras do Iracema inteiro (~308 GB)
  não cabem. Confira o espaço antes.
- Rodadas longas: log fora do `/tmp`, que é apagado no boot. Para encerrar um processo use o PID explícito, porque
  `pkill -f` e `pgrep -f` casam com a própria linha de comando.

## Fluxo dos dados

Os estágios trocam **arquivos**, nunca variáveis, e cada um lê o que o anterior gravou:

1. **`Synthetic/Generator/index.py` — `SyntheticGenerator`.** O fluxo do FaultSeg3D (Wu et al. 2019,
   `Documents/Reference/projeto_wu/article/fault_seg (wu).pdf`): refletividade → dobra → cisalhamento → falhamento → wavelet de Ricker →
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
     Ao mudar uma etapa, confira um tile contra o `git show HEAD:Synthetic/Generator/index.py`: a máscara tem de sair idêntica.
   - Desde 24/11 a pasta tem só `Generator/`, `Nature/`, `Utils/`, `Analysis.ipynb` e o roteiro `prompt` (com os ajustes de
     26/11; o original em `files/prompt_original`). Os `1 - Profile`/`2 - Calibration`, o `Regions/` e o `files/experiments/`
     da calibração por medida de 10–11/2026 saíram (estão no histórico do git).
   - **`Synthetic/Analysis.ipynb` (26/11): otimização das faixas do gerador pelo L-SHADE do `Nature`.** Cada candidato
     gera 200 tiles em `Dataset/dataset_syntest` (sementes iguais para todos: tile i = 121958 + i), roda o `Format.ipynb` da
     pasta (cópia do do `dataset_74`, que só é lido), treina um proxy pelo `Model/1 - Model.ipynb` (`unet_wu_z`, receita do Wu,
     30 épocas, AMP, EMA). O objetivo (27/11) é o clDice (`Model/Metrics/ClDice`) em três lugares, somado com os pesos `WEIGHTS`
     (0,2 / 0,4 / 0,4): as 20 validações do Wu contra o rótulo (`SyntheticScore`, em `sintetico`), o F3_wu contra o modelo do artigo
     do Wu e os patches 1200/1300 do Marlim contra o especialista (`BlockScore`, em `blocos`; 1400/2600 de prova). Cada termo é dividido
     pelo da referência: o mesmo proxy treinado nos 200 volumes de treino do Wu (o `dataset_74` não é régua: o usuário o considera ruim).
     `Dataset/dataset_wu_treino` (volumes 0–199) e `Dataset/dataset_wu_val` (200–219) são só o `DataBase.csv` filtrado do `dataset_wu`,
     reescrito pelo notebook. O SPACE tem 25 genes (camadas, dobra e mergulho abertos em 27/11); os fixos são os padrões do gerador.
     O IoU e a AUC nas validações (`3 - Predict` com `FULL_DATASET = True`) ficam só para comparar. `Synthetic/files/rodar.sh` roda o
     notebook solto (papermill) até o `DEADLINE` e para sem fechar a campanha: rodar de novo continua da `memory`.
     Os notebooks do `Model/` rodam em cópias (`runNotebook`) que leem `Synthetic/files/info.json` e gravam em
     `Synthetic/files/runs/model_N`: o `Task/info.json` e o `Model/Backup` não mudam. Retomada: `files/memory` (o notebook põe
     `Memory.EVERY = 1`) e o cache por `info.json` em `files/runs`. Genoma: cada faixa vira (centro, largura).
   - `Synthetic/Utils/index.py`: `runNotebook` (cópia com trechos trocados e parâmetros do papermill, gravada na pasta do log e
     rodada com o `cwd` na pasta do notebook, para nunca deixar arquivo nas pastas do pipeline; um trecho que sumiu do
     notebook levanta erro, para a cópia nunca gravar no lugar errado), `BlockScore` (prediz o F3_wu inteiro e só a inline
     anotada de cada patch do Marlim), `SyntheticScore` (prediz os tiles de um dataset, o `dataset_wu_val`, e mede contra o rótulo;
     referências das duas guardadas em `files/blocks/`) e `getRuns`. A conta do clDice (esqueleto 2D por seção, tolerância,
     orçamento igual) mora só em `Model/Metrics/ClDice`, usada por essas duas classes.
2. **`Dataset/<nome>/Format.ipynb`.** Lê `original/*/images|masks`, escalona e grava `images/`, `masks/` e
   o `DataBase.csv` (`id`, estatísticas, `shape`, `img_path`/`mask_path` absolutos, `scaling`). Também reescreve
   `Task/info.json` com o nome do dataset. O que se repete entre os `Format` mora no `Dataset/index.py`: `getFiles`,
   `setFolder`, `showTile`, `getDataInfo` (a tabela da célula logo antes do fim do `Format`: formato, shape, mínimo, máximo,
   média e desvio do `original/` e do `images/`), `TilesBuilder` e a `Normalization`, escolhida pelo `scaling` do `info.json`:
   `'normalize'` (o padrão sem a chave: **min-max de cada volume** para `[0,1]`, decisão de 06/10/2026), `'percentile'`
   (corte no p01/p99 do conjunto inteiro, reescalado para `[0,1]`), `'standardize'` (média 0 e desvio 1 por volume, o
   passo do artigo da ResACEUnet) ou `null` (volume cru). A escolha fica na coluna `scaling` do `DataBase.csv`
   (`Normalization.read`: sem ela é o min-max, célula vazia é `null`), e o `1 - Model` e o `3 - Predict`
   param com erro se ela não bater com a rodada. O `marlim_opt` é `'normalize'` desde 17/11/2026: 495 tiles do
   `SyntheticGenerator` em dois regimes, 220 na configuração do `dataset_74` (com o `gain`/`clip` que dão o contraste e
   a saturação do bloco) e 275 calibrados no Marlim, sem rótulo nas falhas menores; o `README.md` da pasta tem a receita
   e os números (a versão anterior, calma, está em `Synthetic/files/experiments/marlim_opt_calma`). O `dataset_marlim`
   (275 tiles, `'normalize'`) é só o regime calibrado no Marlim (a configuração C5), gravado pelo `2 - Calibration`. O `dataset_74_wu` não tem `images/` próprios: junta os do `dataset_74` e do `dataset_wu` e roda,
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
   batch (a agenda da ResACEUnet), `constant` (ou outro nome) = `lr` fixo, como o Adam 1e-4 do FaultSeg3D, `ModelEMA` do
   `Model/EMA/` quando `ema` é true, progresso
   corrente em `Model/progress.json`). Salva `Model/Backup/model_N/` com `info.json`, `model.pth`
   (`{'model', 'optimizer', 'timestamp', 'history'}`), `train.png` e `predictions/`; `N` é o maior `model_N` + 1. No
   `info.json` salvo, `processing` é o `Task/info.json` exatamente como veio (copiar e colar reproduz a rodada),
   `division` guarda o split (`val_size`, `test_size`, `n_images`) e `model` os argumentos da rede: o
   `model.img_size` vem do `shape` do `DataBase.csv` ou, com `crop` no `augmentations`, da janela do recorte. O
   `3 - Predict` lê o split do `division` e não tem valor próprio.
5. **`Model/2 - Compare.ipynb`** junta todo `Backup/*/info.json` numa tabela e compara variações (média±std entre
   trials); **`Model/3 - Predict.ipynb`** reavalia os modelos de um `BASE_PATH` no dataset em que cada um treinou (ou
   num `DATASET` comum, sem os tiles vistos no treino; com `FULL_DATASET = True`, esse dataset inteiro em vez do split de teste), somando os voxels de todos os tiles de teste antes de dividir —
   a mesma conta do `test_iou` — e grava as métricas no `test.<dataset>` do `info.json`. O protocolo da Tabela 2 da
   ResACEUnet (recortes 96³) saiu dele em 08/10/2026 e só aparece nos relatórios de `Documents/`. A matriz de confusão e
   a curva ROC (opcional, `ROC_CURVE`; histograma de log-odds por voxel, acumulado na mesma passada da rede) moram em
   `Model/Metrics/ConfusionMatrix` e `Model/Metrics/ROC`. Com `SAVE_PATH`, cada modelo grava `matrix.png` e `roc.png` na
   própria pasta (`_<dataset>` no nome quando testado fora do dataset do treino), e a `auc` entra no `test.<dataset>` do
   `info.json`.
6. **`Blocks/Marlim/Analysis.ipynb`** — o bloco real com interpretação. Predição e análise ficam num notebook só.
   - **Código comum:** os notebooks de `Blocks/` importam do `Blocks/index.py` o `TileGrid` (grade e montagem pelos
     núcleos), a `SlidingWindow`, o `BlockPredictor`, o `FaultStickExtractor`, a `BlockSection` (painéis; a camada
     `expert` é a anotação), o `getBlockInfo` (shape, tiles e min/máx/média/desvio de uma amostra), o `modelSummary` e o
     `saveReport`. Cada notebook tem só a configuração e o fluxo; o `FaultComparer` do Marlim estende a `BlockSection`.
   - **Volume e tiles:** `getVolume(grid, tiles, cache, context)` monta o bloco pelos núcleos e grava em
     `files/cache/volume*.npy` (no Marlim, `context=32` guarda os 128 inlines reais dos tiles); `writeTiles` corta um
     volume em tiles no formato dos blocos (núcleo 64, borda 32 espelhada) e reproduz os `.dat` do F3 com diferença zero.
   - **3D:** a seção "VISUALIZAÇÃO 3D" de cada notebook grava `<pasta do bloco>/view3d.html` pelo `saveViewer`, a partir
     do `predicted.npy` (por isso `SAVE_PRED = True` nos três). O `Viewer3D` usa o template `Blocks/viewer3d.html` (Three.js 0.128
     do jsdelivr) em resolução cheia: a probabilidade em uint8 embutida, e a sísmica em partes `.js` compartilhadas em
     `Blocks/<bloco>/files/cache/view3d/`, lidas pelo caminho relativo (abrir o HTML dentro do repositório). Mostra a
     fatia de tempo e as paredes inline/crossline como a figura 7 do Wu, limiar ajustável, sísmica ou fundo preto, anotação na
     inline anotada do Marlim, ficha da rede (`viewerInfo`) e, com várias camadas, comparação lado a lado com orçamento igual opcional
     (`#a=<índice ou grupo__modelo>&b=...&thr=&budget=1&mode=mask` no link).
   - **Configuração:** `BASE_PATH` (a pasta dos `model_N/`: `../../Model/Backup` ou `../../Documents/Backups/<campanha>`),
     `MODELS`, `PATCH_IDS`, `SAVE_TILES`, `SAVE_PRED`, `SKIP_DONE`, `OVERLAP`, `WINDOW` e os botões do extrator
     (`STICK_TOL 0.80`, `STICK_MIN 140`, `STICK_GAP 59`, calibrados no patch 1200).
   - **Entrada:** `files/patches/<id>/tiles/*.dat`, float32 cru em `[0,1]` (p01/p99 do bloco), eixos
     `(inline, z, xline)`, shape no `patch_metadata.json` da mesma pasta. Os 128 inlines de cada `.dat` são reais; o
     `original_shape` de 64 é só o miolo.
   - **Predição:** roda a `SlidingWindow` **na janela em que a rede foi treinada**, com peso de Hanning na emenda, e
     grava `predict.json` em `<BASE_PATH>/<modelo>/marlim/patch_<id>`, o `predicted.npy` com `SAVE_PRED` (padrão `True`)
     e as `masks/` (um `.dat` por tile) só com `SAVE_TILES` (padrão `False`). Com `SKIP_DONE`, um `predicted.npy` pronto
     dispensa a predição e os `.dat` inteiros de `masks/` são relidos. Só prediz modelo com `scaling` `'normalize'` ou
     `'percentile'`.
   - **Análise:** extrai sticks (`FaultStickExtractor`) e compara com o especialista (`FaultComparer`, contra
     `files/patches/<id>/<id>_interpretado.png`). A inline anotada não é a central; o índice dela está em
     `files/cache/inline_index.json`.
   - **Saída:** figuras e `sticks_report.csv` em `<BASE_PATH>/outputs/marlim`.
   - A anotação é parcial (só as falhas principais): a precisão é limite inferior e o F1 premia marcar pouco.
7. **`Blocks/Iracema/Analysis.ipynb`** — o mesmo notebook para o Iracema, que não tem interpretação: só figuras e o
   resumo da inline. Os tiles ficam soltos em `files/patches/`, com um `files/patch_metadata.json` (2751 × 2000 × 1754,
   38 528 tiles; só os 896 da inline 1375 estão em disco). Com `FULL_PREDICT = False` ele prediz só os tiles que cobrem
   a `INLINE`; `SAVE_TILES` e `SAVE_PRED` (padrão `False`) gravam as `masks/` e o `predicted.npy` (~38,6 GB).
   - **`Blocks/F3/Analysis.ipynb`** é a mesma cópia para o F3 (Mar do Norte, Holanda), também sem interpretação: 651 × 462 ×
     951 em 1320 tiles soltos em `files/patches/` (mesmo corte e mesma escala p01/p99 em `[0,1]`; o SEG-Y original fica em
     `~/Projects/falhas_f3`). Prediz o bloco inteiro por padrão (`FULL_PREDICT`) e grava em `<BASE_PATH>/<modelo>/f3` e
     `<BASE_PATH>/outputs/f3`. As `masks/` (~11 GB, um `.dat` por tile) só saem com `SAVE_TILES = True` e o
     `predicted.npy` (1,1 GB, padrão `True`, o 3D lê dele) com `SAVE_PRED`. F3 e Iracema têm o mesmo código e
     diferem só nas constantes da configuração (`BLOCK`, `BASE_PATH`, `INLINE`, ...).
   - **`Blocks/F3_wu/Analysis.ipynb`** é o recorte do F3 que o Wu usa no artigo (o `gxl.dat`: inline 0:512, z 334:462,
     xline 0:384), 512 × 128 × 384 em 96 tiles gerados pelo próprio notebook a partir dos tiles do F3 (`writeTiles`). Além
     do fluxo do F3, desenha a fatia de tempo do artigo (`TIME_SLICE = 99`, z 433); `STICK_MIN 60` porque a seção tem 128 de altura.
8. **`Synthetic/Nature/`** — framework mono-objetivo de otimização usado pelo `Synthetic/Analysis.ipynb`.
    - No notebook, com a pasta `Synthetic/` como cwd: `from Nature.index import NatureSelector`.
    - `NatureSelector(nome, params, memory)` escolhe entre `genetic` (CMA-ES; aceita `mean` para a primeira nuvem nascer
      num ponto conhecido), `pso`, `de`, `lshade` (população ≥ 8; o orçamento é população × gerações avaliações) e
      `lsrtde`, e delega `update()`/`portrait()`/`info()`; `n_gaussian > 1` não combina com `memory`.
    - `Problem` traduz `{variável: {'type', 'bounds'}}` em genoma; um objetivo `nan` vira penalidade.
    - `Memory` persiste `state.npz`/`best.json`/`history.json`, a cada `Memory.EVERY` gerações (padrão 10).
9. **`Documents/Backups/<campanha>/model_N/`** (`Dissertacao`, `Marcia`, `Multiscale`) guarda modelos de campanhas
    anteriores no formato de `Model/Backup`, com as figuras dos blocos em `outputs/`; os notebooks dos blocos trocam a
    pasta pelo `BASE_PATH`. O relatório geral é o `Documents/REPORT.html`; o 3D dos blocos, a pasta `Validation/` (gerada sob pedido).
10. **`Documents/Reference/`** (fora do git) é a validação contra os dois artigos, rodando o código dos autores:
    - `projeto_wu/code` (o `faultSeg` do Wu, ambiente `~/envs/wu-tf`, sempre por `~/envs/wu-tf/bin/wupython`: TF 2.21 +
      `tf_keras`) e `projeto_zu/code` (a ResACEUnet, ambiente `~/envs/zu`); os scripts acrescentados (`train_ref.py`,
      `predict_blocks.py`, `evaluate_val.py`) e os desvios de compatibilidade estão no `README.md` de cada `results/`.
    - `projeto_*/results/<modelo>/` tem o formato de uma pasta de modelos (`info.json`, `marlim/patch_*/`, `f3/`, `f3_wu/`
      com `predicted.npy` gravado pelo código dos autores), então os notebooks de `Blocks/` só analisam.
    - Na raiz ficam só os dois projetos; os auxiliares moram em `aux/`. `aux/rodar_blocos.py --base <pasta> --models ...`
      roda os três notebooks de `Blocks/` para qualquer pasta de modelos (cópia temporária com `BASE_PATH`/`MODELS`/`PATCH_IDS`
      trocados; saída em `aux/logs/`); as filas da GPU são `.sh` em `aux/` (log em `aux/logs/fila.log`).
    - `aux/controle/` treina pelo `Model/1 - Model.ipynb` sem tocar no `Task/info.json` nem no `Model/Backup` (`treinar.py`);
      `aux/experimentos.py` testa pré (contraste casado, z-score do bloco) e pós-processamento (suavização) por bloco;
      `metricas_blocos.py` mede a continuidade (pedaços por 10 mil voxels, no limiar 0,5 e no orçamento igual ao fseg-70) e o
      Marlim contra o especialista (`aux/logs/metricas_blocos.csv`).
    - `aux/mapeamento.py` mede, no F3_wu e no F3, quanto das falhas do modelo do artigo do Wu (`wu_artigo_2018`) e do fseg-70 cada modelo acha
      (orçamento igual, tolerância de 3 amostras; `aux/logs/mapeamento.csv`); `aux/sintetico.py` dá o IoU sintético no 0,5 e no melhor limiar, a AUC
      e o IoU nas validações do Wu (200–219) que cada modelo não viu (`aux/logs/sintetico.csv`); `aux/metricas_sinteticas.py` (CPU) mede no teste
      sintético as candidatas a métrica de escolha (AP, F1 com tolerância, clDice, continuidade) para a correlação com os blocos no README; `aux/ablacao.sh` + `ablacao_pos.sh` é a ablação
      perda × otimizador (UNet do Wu/`dataset_wu` e MACNN/`dataset_74`), treinada em `aux/controle/model_3..6`.
    - Relatórios: `aux/readme_task.py` refaz o `Documents/REPORT.html`, um arquivo só e sem dependência (só os modelos do `Model/Backup`
      com o Wu e o Zu como régua; textos em `aux/readme_task_textos.json`), sem 3D. O 3D dos blocos é a pasta `Validation/` da raiz
      (no `.gitignore`), gerada só sob pedido do usuário por `aux/validation.py` para os modelos do `MODELS` no topo dele: `index.html`,
      `src/index.js` (o visualizador, cuja fonte mora em `aux/validation/` e é copiada a cada rodada), `blocks/<bloco>/` (a sísmica) e
      `results/<modelo>/` (`info.json` reduzido com a legenda e as predições); zipada, abre em qualquer computador sem internet. O
      `Validation/README.md` explica o formato e como gerar para outros modelos. A nota geral de escolha é o clDice com tolerância de 3 amostras no
      orçamento igual (`aux/cldice_blocos.py`: F3_wu contra o modelo do artigo do Wu, Marlim contra o especialista na inline anotada); no 3D
      o esqueleto é a união dos esqueletos 2D das seções inline e crossline, porque o `skeletonize` 3D do scikit-image reduz uma superfície
      a linhas. `aux/teste_metricas.py` é o teste que a escolheu (defeitos de propósito e concordância entre os patches do Marlim); `aux/readme_projetos.py` gera o `README.html`
      de `projeto_wu` e `projeto_zu` (pipeline, rede, arquivos do `code/`, como rodar); `aux/esquemas.py` desenha as redes e os
      fluxos em SVG. O relatório antigo ficou em `aux/README_anterior_23-11.html`.
    - Disco: cada modelo grava ~5 GB de `predicted.npy` nos três blocos (Marlim 4 × 0,9, F3 1,1, F3_wu 0,1); 35 modelos encheram o
      disco em 26/11. Confira o espaço antes de rodar muitos modelos.

## Convenções que atravessam o repositório

- **Módulo é pasta com `index.py`** exportando o tipo principal: `from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Synthetic.Nature.index import NatureSelector`. Notebook fora da pasta ajusta
  o path (`sys.path.append('../../Model')` nos `Blocks/`; o `Synthetic/Utils/index.py` põe a raiz e o `Model/`).
  Importar módulo nunca executa trabalho.
- **Eixos:** os datasets são gravados em `(x, z, y)` — o `saveTile` transpõe `(0, 2, 1)` na saída do gerador, e
  `Synthetic/utils.formatAxis` faz o mesmo para visualizar. Os tiles reais (os `.dat` dos blocos e as regiões) são
  `(inline, z, xline)`, o mesmo arranjo.
- **Nova rede:** arquivo em `Model/Network/types/X.py`, import e um `if` em `ModelNetwork.get()` com uma linha
  MAIÚSCULA citando a origem; o nome usado ali é o que vai em `Task/info.json`. Hoje: `unet_3d`, `dbrnet`,
  `segresnet`, `resaceunet_grva`, `resaceunet_wu` (a `ResACEUNet2` atual do repositório dos autores),
  `resaceunet_zu` (a do artigo, 3 estágios, do primeiro commit), `resaceunet_grva_2` (26/11: a híbrida, esqueleto e entrada em z da
  `unet_wu_z`, pares residuais e os blocos ACE do Zu nos níveis 1/4 e 1/8, presa à janela do treino; 14,9 M com nf 16), `macnn`, `fault_seg_net`, `nru_net`, `fault_edge_former`,
  `nra_sanet`, `unet_wu` (a UNet do FaultSeg3D, sem normalização, igual ao `fseg-70` do Wu) e `unet_wu_z` (a mesma com a
  entrada `[0, 1]` do percentil levada de volta à escala z). A `dbrnet` é a antiga `unet3d_v2`: um `info.json` com o nome
  velho só carrega trocando o nome.
  O `ensemble` (`types/Ensemble.py`) não treina: é a média das probabilidades de várias redes prontas (`members` no
  `model` do `info.json`), elevada a `power` (o ponto de operação) e com espelhos laterais opcionais (`flips`). Roda nos
  notebooks dos blocos como qualquer `model_N`; quem monta é o `checks/make_ensemble.py` da campanha.
- **Nova loss:** classe em `Model/Losses/index.py` e entrada em `Losses.options` (`cross_entropy`, `dice_focal`,
  `dice_ce`, `focal`, `smooth_dice`, `compound`, `tversky`, `dice_focal_cldice`, `bce_balanced` — a BCE balanceada do
  FaultSeg3D). Toda loss força `float32` fora do autocast. No MONAI 1.5.2 o termo
  focal do `DiceFocalLoss` é sempre sigmoide, mesmo no caso multiclasse — considere isso antes de comparar campanhas
  `dice_focal`.
- **Aumentação:** só pelo `augmentations` do `info.json` (`null`/ausente = tiles do `Format` como estão, bit a bit igual
  ao pipeline sem ela). `Model/Transforms/index.py` reproduz o `build_data.py` da ResACEUnet (as chaves `contrast` (gamma),
  `rotate90`, `flip`, `rotate`, `smooth`, `noise` e `crop` (recorte falha/fundo); chave fora de `Transforms.OPTIONS`
  levanta erro) mais o `zoom` isotrópico (fator ≥ 1, para cobrir o período de 17–25 px
  das falhas do Marlim) e o `gain` (o desvio do tile multiplicado por um fator sorteado: a rede sem normalização passa a
  aceitar o contraste menor com que os blocos chegam pelo p01/p99), sorteado no `DataLoader` com semente `(42 + trial, época, índice)`,
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
