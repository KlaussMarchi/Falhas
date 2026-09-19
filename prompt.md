# MISSÃO — Metodologia confiável para calibrar o gerador sintético e detectar as falhas do bloco de Marlim

> Você tem terminal e permissão de escrita neste repositório (`/home/grva-mint/Projects/Falhas`, branch `main`).
> Leia este documento inteiro antes de tocar em qualquer arquivo. Ele contém todo o contexto necessário: caminhos,
> versões, números já medidos, custos de execução e as regras do projeto. Não há nada de solução aqui — a solução é
> sua, e é ela o produto deste trabalho.

---

## 1. PAPEL

Você é **engenheiro sênior de geofísica computacional e aprendizado de máquina**, com especialidade em:

- sísmica de reflexão 3D (atributos de coerência/semblance, tensor de estrutura, wavelet de Ricker, refletividade,
  dobramento, falhamento, migração e zonas de baixa razão sinal-ruído);
- segmentação volumétrica com redes 3D em PyTorch e treino com dado **sintético** transferido para dado **de campo**
  (a linha FaultSeg3D de Wu et al., 2019 em diante);
- **métricas de comparação entre distribuições de amostras** e seus vieses com amostra pequena;
- **otimização caixa-preta** (CMA-ES, DE, PSO) com orçamento escasso e função-objetivo ruidosa e cara;
- projeto experimental: separar ganho real de ruído de medição.

Tom: direto, quantitativo, em português. **Toda afirmação sua vira um número medido neste repositório ou não é feita.**
Você não entrega opinião: entrega medição, tabela e decisão justificada.

---

## 2. O PROBLEMA, SEM AMBIGUIDADE

### 2.1 O pedido original do usuário (verbatim)

> anexei aqui as fotos dos resultados ao tentar gerar dados sismicos volumetricos sintéticos com falhas de forma a
> fazer os modelos de redes neurais treinarem com esses dados e fazer uma devida predição no bloco de marlim
> detectando de forma correta das falhas e comparando com a notação do especialista. O algoritmo otimizador dos
> parametros do gerador sintetico para tentar replicar a restrutura de Marlim está em Marlim/regions e até que lembra
> um pouco mas ainda há certa divergencia na estrutura e comportamento das falhas e na hora de fazer a predição do
> bloco de marlim ainda percebe-se uma grande divergencia no que deveria detectar (imagens anexadas), as falhas não
> estão continuas e precisam ainda de ajustes pois o modelo precisa treinar de forma a entender e preditar o bloco de
> marlim devidamente. O dataset_74 tenta replicar o dataset_wu que é um repositorio geral para falhas e os modelos com
> eles podem ser vistos na pasta Marcia/ e é o ponto de referencia mas veja o que falta na pasta
> Marlim/Regions/Synthetic/ para assimilar o maximo possivel e fazer o modelo entender o que é pra detectar no bloco de
> marlim, veja se a metrica de similaridade de imagens que to usando de fato é a melhor estrategia para otimizar esses
> parametros e se tudo esta sendo feito da melhor maneira possivel e é a melhor aposta que poderia ter para resolver
> esse problema e conseguir de fato ter a melhor predição possivel com base no que temos na literatura. tudo que for
> mudado deve estar organizado com codigo limpo conforme CODE_STYLE.md e obedecendo a estrutura que ja tenho dos
> codigos mas garanta que as mudanças objetivamente serão melhores e irão de fato, uma vez que eu rodar o
> Synthetic/Analysis.ipynb e otimizar os parametros e rodar o Dataset/Marlim_regions/Format.ipynb e treinar o modelo
> ele vai entender de fato ("ahhh agora sim eu tenho dados sinteticos perfeitos para o bloco de marlim") e de fato
> fazer a predição devida, mas preciso de uma metrica boa e objetiva e replicavel para otimizar os parametros e criar
> os dados sinteticos, explorar e explotar bem e ser perfeitamente capaz de dar conta do recado (...) teste, explore,
> simule e faça o que for preciso para conseguirmos uma metodologia de respeito e confiabilidade.

### 2.2 Desambiguação dos caminhos citados

O usuário cita nomes aproximados. Os caminhos reais são:

| o que ele escreveu | caminho real neste repositório |
|---|---|
| `Synthetic/Analysis.ipynb` | **`Marlim/Regions/Synthetic/Analysis.ipynb`** (é o otimizador; `Synthetic/` na raiz tem `Generate.ipynb` e `Adjustments.ipynb`, não `Analysis.ipynb`) |
| `Dataset/Marlim_regions/Format.ipynb` | **`Dataset/dataset_regions/Format.ipynb`** |
| `Marlim/regions` | **`Marlim/Regions/`** (com `Marlim/` e `Synthetic/` dentro) |
| "as fotos anexadas" | **não chegaram nesta sessão**. A evidência equivalente está em disco — ver §5.6 |

### 2.3 O problema reformulado

O objetivo final **não é** a nota de similaridade nem o IoU sintético: é **a máscara de falha predita no bloco real de
Marlim concordar com a interpretação do especialista**, com traços contínuos e no lugar certo. Hoje ela não concorda
(§5.5) e as falhas saem picotadas.

A cadeia inteira é: **parâmetros do gerador → tiles sintéticos → dataset → rede treinada → predição no Marlim →
comparação stick-a-stick com o especialista.** O que se otimiza hoje é o **primeiro elo** (semelhança de imagem por
região), e o que se cobra é o **último**. Não existe, hoje, no repositório, nenhuma medição ligando um elo ao outro.

Você deve entregar uma **metodologia** — objetiva, reprodutível e defensável perante a literatura — que faça a cadeia
inteira funcionar, e provar com dados que ela é melhor do que a atual. Isso inclui julgar, com evidência, se a
estratégia atual de otimização é a certa, corrigi-la ou substituí-la, e deixar o repositório em estado tal que a
sequência que o usuário executa (§7) produza de fato um ganho mensurável na predição em Marlim.

---

## 3. AMBIENTE

```bash
conda activate torch-gpu            # ou: conda run -n torch-gpu python ...
```

- O python do `base` **não tem torch**. Nunca rode nada deste projeto sem ativar o `torch-gpu`.
- Versões medidas agora: `torch 2.7.1+cu118` (CUDA disponível: True), `monai 1.5.2`, `numpy 2.2.6`, `scipy 1.16.2`,
  `opencv 4.12.0`, `scikit-learn 1.7.2`, `pandas 2.3.3`. Também disponíveis: `torchmetrics`, `albumentations`,
  `papermill`, `scikit-image`, `psutil`, `tqdm`.
- **GPU: NVIDIA Quadro P6000** (Pascal, 24 GB, sem tensor cores — por isso `use_amp = False` no treino).
- CPU: o notebook de busca usa `WORKERS = 6` processos em `SCHED_IDLE` com trava térmica `MAX_TEMP = 88 °C`
  (`waitCool()`), porque acima de ~12 workers a máquina chegou a 92 °C e a vazão satura na banda de memória.
- Não há build, lint nem suíte de testes. **A verificação é rodar o notebook de cima para baixo e comparar números.**
- `.gitignore` exclui `*.npy`, `*.pth`, `*.dat`, `*.png`, `Articles/`: datasets, pesos e figuras são locais.

Executar um notebook isolado:

```bash
conda run -n torch-gpu papermill "<notebook>" /tmp/out.ipynb -k python3 --cwd "<pasta do notebook>"
```

**Todo caminho dentro de um notebook é relativo à pasta dele** — `cwd` errado quebra tudo.

Campanha de treinos (caminho normal): editar `Task/task.json` e rodar `cd Task && conda run -n torch-gpu python index.py`.
Para cada rodada ele grava a linha em `Task/info.json`, executa `Dataset/<dataset>/Format.ipynb` (pulado quando o
`DataBase.csv` já existe e `img_size` é `null`) e depois `Model/Analysis.ipynb`, com saída em `Task/logs/<nome>_out.ipynb`.

---

## 4. O PIPELINE — O QUE CADA ARQUIVO FAZ

Os estágios trocam **arquivos**, nunca variáveis.

### 4.1 `Synthetic/index.py` — classe `SyntheticGenerator` (279 linhas, leia inteiro)

Gera um volume 128³ com a máscara das falhas, num cubo com margem 64 cortado no fim.
Cadeia: `genReflectivity()` → `applyFolding()` → `applyShearing()` → `applyFaulting()` → `applyWavelet()` (Ricker) →
`applyNoise()` → `crop()` → z-score.

- `get()` devolve `(imagem z-scorada float32, máscara uint8)`; `set(options)` sobrescreve atributos; cada atributo é
  uma faixa `(low, high)` sorteada por tile.
- `dataset(options, n_jobs)` gera em multiprocesso e grava `<directory>/<output>/{images,masks}/img_XXXX.npy`, com
  índice **global** entre regiões. `options = {'directory', 'seed', 'regions': {nome: {'n_images','output','params'}}}`;
  chave desconhecida levanta erro.
- `applyGain(image, jitter)` aplica `gain × jitter` e satura em `±clip`. `getJitter(n, seed)` sorteia o contraste
  log-normal do lote **normalizado para mediana 1**.
- `saveTile` transpõe `(0, 2, 1)` na saída: os datasets são gravados em `(x, z, y)`.
- Atributos e padrões atuais (a configuração calibrada do `dataset_74`): `layerRange (26,233)`,
  `layerThickness (1,4)`, `foldCount (15,48)`, `foldSigma (17,57)`, `foldAspect 1.0`, `foldAmplitude (-35,5)`,
  `foldDamping 1.35`, `foldBaseShift (-0.75,4.45)`, `shearOffset (-8.68,3.3)`, `shearGradient (-0.1,0.02)`,
  `faultCount (5,10)`, `faultThrow (15,32)`, `faultDipAngle (55,81)`, `faultRoughness 3.54`,
  `faultRoughSigma 8.55`, `faultDecaySigma (49,59)`, `faultZoneWidth 0.99`, `faultThreshold 0.77`,
  `faultCurveProb 0.07`, `faultCurveMax 8.44`, `waveletFreq (72,99)`, `waveletDuration 0.1`, `waveletDt 0.0012`,
  `noiseLevel (0.015,0.618)`, `noiseSigma (1.0,1.0,0.5)`, `gain None`, `gainJitter 0.0`, `clip None`.
- Em `applyFault`, o azimute é `uniform(0, 2π)` e o mergulho vem de `faultDipAngle`; o rejeito (`getThrow`) é
  gaussiano em torno do centro ou rampa ao longo do mergulho, metade das vezes cada; `getBend` faz a falha lístrica
  com probabilidade `faultCurveProb`.

### 4.2 `Marlim/Regions/Marlim/Analysis.ipynb` — as três regiões reais

Separa tiles reais 128³ dos quatro patches (`1200`, `1300`, `1400`, `2600`) em `calm` / `faulted` / `dead` **por
conteúdo** (não por faixa fixa de z), grava em `Marlim/Regions/Marlim/files/<região>/*.npy` no eixo
`(inline, z, xline)` e na escala [0,1] dos `.dat`, com índice em `files/DataBase.csv`.

Regras: `calm` = inteiro no intervalo refletivo, nenhuma falha anotada a menos de `FAULT_GAP=64` px e mais contínuo
que 95% dos tiles com falha; `faulted` = intervalo refletivo cruzado por ≥ 128 px de falha anotada; `dead` = inteiro
na zona morta, sem envelope de refletor e sem falha anotada por perto. Dois tiles da mesma região dividem no máximo
25% do volume.

Contagem atual: **calm 149, dead 123, faulted 188** (460 tiles).
σ mediano de volume por região: **calm 0.1921, faulted 0.1457, dead 0.0161** (a zona morta é ~12× mais fraca).
`discontinuity` mediana: calm 0.0022, faulted 0.0396, dead 0.1277. `faultLength` mediana no `faulted`: 151 px.

### 4.3 `Marlim/Regions/Synthetic/Analysis.ipynb` — **o otimizador (o coração do problema)**

12 MB, 33 células. É autossuficiente: não lê arquivo de configuração; a única entrada são os tiles reais de
`../Marlim/files`. Estrutura das células:

| célula | conteúdo |
|---|---|
| 0 | imports; `sys.path.append('../../..')`; `SyntheticGenerator`, `NatureSelector`, `Memory`, `getFiles` |
| 2 | `REGIONS`, `CLIP=0.45`, `SEED=20260915`, `N_IMAGES={'calm':12,'dead':12,'faulted':8}`, `WORKERS=6`, `MAX_TEMP=88` |
| 3 | carrega os tiles reais e o `SIGMA` mediano por região |
| 5 | `waitCool()`, `setWorker()`, `getPool()` (pool por `fork`, um por chamada) |
| 6 | **`class ImageSimilarity`** — a métrica inteira (11 KB) |
| 7 | prova visual: `similarity.plot` entre pares de regiões reais |
| 9 | **matriz de calibração** real × real |
| 11 | **`REFERENCE`** (23 genes × 3 regiões), `getParams(region, genome)`, `getTile(params, seed, jitter)` |
| 13 | `BOUNDS` (23 variáveis), `FIXED`, `SPREAD=0.3`, `POPULATION=16`, `GENERATIONS=100`, `PATIENCE=100`, `EXTEND=True`, `Memory.EVERY=1` |
| 14 | `getBox`, `decode`, `getTasks`, `getTileFeatures`, `getMemory`, `getGenerations`, **`objective(X, region)`** |
| 16/18/20 | uma busca CMA-ES por região (`NatureSelector('genetic', …, memory=getMemory(região))`) + gráficos |
| 22 | **MÉTRICAS FINAIS** — reavalia busca × `REFERENCE` em **sementes novas** (`CHECK_SEED = SEED+1000`, `N_CHECK=36`) e escolhe o vencedor por região → `REGIONS_OPTIMIZED` |
| 25 | `SyntheticGenerator().dataset(options, n_jobs=WORKERS)` grava `Dataset/dataset_regions/original/<região>/{images,masks}` + `synthetic.json`, e remede a nota dos arquivos gravados |
| 27–31 | `showRegion()` — comparação visual real × sintético por região |

**A métrica (`ImageSimilarity`), em resumo do código:**

- cada tile 128³ vira um vetor de **42 colunas** a partir de **8 seções 2D** (4 inline + 4 crossline, nas frações
  0.125/0.375/0.625/0.875). A média das seções é o valor do tile, **menos** os atributos laterais
  (`lag1/2/4/8/16`, `Px0..Px3`), que saem um por orientação (sufixo `X` = xline lateral, `I` = inline lateral);
- 5 grupos e pesos: `amplitude` 0.15 (`logStd, kurt, satFrac, q95n`), `espectro` 0.20 (`Pz0..Pz5, logPeakZ, Px*X, Px*I`),
  `estrutura` 0.20 (`cohMean, coh25/50/75, dip10/50/90` pelo tensor de estrutura), `continuidade` 0.15
  (`lag*X, lag*I, zcr`), `falhas` 0.30 (`discFrac, lineDens, lineDip, lineLen, discSharp`, todos derivados da
  **semblance com direção de mergulho**, sem usar rótulo);
- cada atributo é comparado pelo **coeficiente de energia normalizado** de Rizzo & Székely (2016):
  `H = (2A − B − C) / (2A + ε)`, `sim = 100·(1 − H)`, com `A` a distância média entre os dois lados e `B`/`C` as
  dispersões internas; `ε = RESOLUTION(0.25) × max(IQR da região, FLOOR(0.05) × IQR do bloco)`;
- agregação: média **aritmética** dentro do grupo, média **geométrica ponderada** entre grupos;
- `SEMBLANCE=0.8` define voxel descontínuo, `LINE_MIN=24` px e `LINE_DIP=(35,88)°` definem lineamento tipo falha;
- **não há grupo de rótulo**: os tiles reais em arquivo não têm máscara, então a comparação atual **não olha a
  máscara sintética** em nenhum momento.

**A busca:**

- CMA-ES (`NatureSelector('genetic', …)`), `backend='vector'` (a população inteira vira um lote num pool só);
- **23 variáveis normalizadas em [0,1]** dentro de uma caixa que é a vizinhança da `REFERENCE`
  (`SPREAD=0.3` da faixa global, mínimo ±1 nas inteiras). As variáveis: `layerLow/Span`, `thickLow/Span`,
  `foldLow/Span`, `foldSigmaLow/Span`, `foldAspect`, `foldAmpLow/Span`, `foldDamping`, `shearLow/Span`,
  `faultLow/Span`, `freqLow/Span`, `duration`, `noiseLow/Span`, `noiseSmooth`, `noiseInline`, `gainJitter`;
- **fora da busca (padrão da classe):** todo o bloco de falha (`faultThrow`, `faultDipAngle`, `faultRoughness`,
  `faultRoughSigma`, `faultDecaySigma`, `faultZoneWidth`, `faultThreshold`, `faultCurveProb`, `faultCurveMax`),
  `foldBaseShift`, `shearOffset`, `waveletDt`;
- `FIXED = {'dead': {'faultLow': 0, 'faultSpan': 1}}` — zona morta sem rótulo de falha, por decisão de conteúdo;
- **ganho fora do genoma**: `gain = 2 × CLIP × σ_real(região)`;
- **sementes fixas na busca** (`SEED`), **sementes novas na decisão** (`CHECK_SEED`);
- memória retomável em `files/memory/<região>_<md5(BOX, N_IMAGES, SEED, GROUPS)>/` — `state.npz`, `best.json`,
  `history.json`. Rodar a célula de novo **estende** a campanha.

### 4.4 `Marlim/Regions/Synthetic/README.md` — **cuidado: está defasado**

484 linhas explicando a função de similaridade, a calibração, as degradações controladas e a bibliografia
(Rizzo & Székely 2016; Xu et al. 2018; Bińkowski et al. 2018; Lopez-Paz & Oquab 2016; Quesada et al. 2025;
Wu et al. 2019; Marfurt et al. 1998; Bahorich & Farmer 1995; Hale 2013; Van Vliet & Verbeek 1995;
Fehmers & Höcker 2003; Gretton et al. 2012; Sejdinovic et al. 2013; Heusel et al. 2017; Haralick et al. 1973;
Portilla & Simoncelli 2000).

**Ele descreve uma versão anterior do notebook.** Divergências já confirmadas: fala em 42 features e **6 grupos**
(inclui um grupo `rotulo` com `maskFrac`/`visibility`/`maskDip`/`maskLen` que **não existe mais** no código),
em `SEARCH.update(região)`/`Generator.REFERENCE`/`regions.json` (nomes que não existem mais), em 24 variáveis,
em 18 tiles por avaliação e em `Dataset/marlim_nature` (dataset apagado). Leia-o como **registro do raciocínio e das
medições históricas**, não como descrição do código atual — e trate a divergência como parte do problema a resolver.

### 4.5 `Dataset/dataset_regions/Format.ipynb`

Lê `original/*/images|masks`, normaliza para [0,1] e grava `images/`, `masks/` e `DataBase.csv`
(`id`, estatísticas, `shape`, `img_path`/`mask_path` absolutos). Também reescreve `Task/info.json` com o nome do
dataset.

A normalização **não** usa o percentil da amostra: usa os trilhos **declarados** no `synthetic.json`
(`(CLIP,) = {clip de cada região}`; `p01, p99 = -CLIP, +CLIP`), porque com ganho por região a saturação fica abaixo de
1% e o percentil cairia *dentro* dos trilhos e mudaria com o número de tiles de cada região. Estado atual: **430
tiles** (50 `calm`, 50 `dead`, **330 `faulted`**).

### 4.6 `Task/info.json` + `Model/Analysis.ipynb` — o treino

`Task/info.json` é a configuração única da rodada (`network`, `dataset`, `img_size`, `lr`, `loss`, `batch_size`,
`scheduler`, `dropout`, `num_filters`), lida como `OPTIONS`.

`Model/Analysis.ipynb`: split `train_test_split(random_state=42)` com ~4,5% val e ~4,5% teste; `CustomDataset` +
`Compose` de `Model/Transforms/index.py`; `Trainer` com clip de gradiente `max_norm=1.0`, `ReduceLROnPlateau`
(factor 0.5, patience 10) ou `CosineAnnealingWarmRestarts`, `EarlyStopping(patience=15, mode='max')` sobre `val_iou`,
`use_amp=False`, 100 épocas, progresso em `Model/progress.json`. Salva `Model/Backup/model_N/` com `info.json`,
`model.pth` (`{'model','optimizer','timestamp','history'}`), `train.png` e `predictions/`. `N` = maior existente + 1.
O `img_size` gravado vem do `shape` do `DataBase.csv`, não do `Task/info.json`.

Redes em `Model/Network/types/` e um `if` em `ModelNetwork.get()`: `standard`, `unet3d_v2`, `segresnet`,
`resaceunet`, `resaceunet_grva`, `macnn`, `fault_seg_net`. Losses em `Model/Losses/index.py`: `cross_entropy`,
`dice_focal`, `focal`, `smooth_dice`, `compound`. Toda loss força `float32` fora do autocast.
O `Trainer.criterion` soma um termo próprio da rede quando ela expõe `compose()`.

### 4.7 `Marlim/1 - Predict.ipynb` — predição no bloco real

Lê `Dataset/marlim/patch_<id>/*.dat` (float32 cru, shape no `patch_metadata.json`), roda `SlidingWindow` **na janela em
que a rede foi treinada** (peso de Hanning na emenda, `OVERLAP` configurável, padrão 0.25) e grava as máscaras
(probabilidade sigmoide, um `.dat` por tile de entrada) em `Model/Backup/model_N/marlim/patch_<id>/masks` + `predict.json`.
Botões no topo: `BASE_PATH` (`'../Model/Backup'` ou `'../Marcia'`), `PATCH_IDS`, `MODELS`, `OVERLAP`, `WINDOW`, `SKIP_DONE`.

Os `.dat` do Marlim **já estão em [0,1]** (p01/p99 do próprio slab) — não há pré-processamento nenhum na predição.
Medido agora: tiles de água são a constante ~0.5026 (σ = 0.0000); tiles com sinal vão de 0.0 a 1.0 com σ ≈ 0.155.

### 4.8 `Marlim/2 - Analysis.ipynb` — a comparação com o especialista

Remonta o volume predito a partir dos `.dat` (descarta a borda de overlap), resolve **qual fatia do slab é a inline
anotada** (não é a central: 1200→16, 1300→52, 1400→24, 2600→8, resolvido por correlação e cacheado em
`Marlim/files/cache/inline_index.json`), extrai sticks com `FaultStickExtractor` e compara com
`Marlim/files/patches/<id>/<id>_interpretado.png` (2240×1601, alinhado 1:1 pixel-voxel com a inline anotada) pelo
`FaultComparer`.

Métricas de `FaultComparer.metrics` (casamento 12 px / 20°, `hit=0.5`):
`recall` = fração do comprimento anotado coberta pelos sticks preditos; `precisao` = fração do comprimento predito
coberta pela anotação; `deteccao` = fração dos sticks do especialista cobertos em ≥ 50%; `f1`, `sticks`, `falhas_gt`,
`len_p50`, `len_total`, `len_total_gt`. Botões: `STICK_TOL=0.50`, `STICK_MIN=140` px, `STICK_GAP=59` px,
`dipRange=(43,76)°`. Saída: figuras `predicted_{seismic,mask,sticks}.png` por patch e o CSV
`Marlim/files/comparisons/sticks_report_<base>.csv`.

### 4.9 `Nature/` — framework de otimização

`NatureSelector(nome, params, memory)` escolhe entre `genetic` (CMA-ES), `pso`, `de`, `lshade`, `lsrtde` e delega
`update()`/`portrait()`/`info()`; `Problem` traduz `{variável: {'type','bounds'}}` em genoma; `Memory` persiste
`state.npz`/`best.json`/`history.json`. Módulos em `Nature/Models/*/index.py` e `Nature/Processing/*/index.py`.
**Quirk conhecido (não corrigido):** a retomada não é bit a bit igual à corrida contínua — `decompose(True)` no resume
faz `eigen = nfe` e a primeira geração depois dele pula a decomposição. Continua sendo CMA-ES válido.

### 4.10 `Marcia/` — o ponto de referência

`Marcia/model_N/` guarda modelos antigos/externos no mesmo formato de `Model/Backup`, **com as predições no Marlim já
feitas** (10 modelos com pasta `marlim/`). São a régua histórica que o usuário cita. Datasets usados lá:
`dataset_74`, `dataset_wu`, `dataset_74_wu`, `dataset_74_mar`, `dataset_wu_mar`. Ambos os notebooks do Marlim aceitam
`BASE_PATH = '../Marcia'`.

### 4.11 Datasets existentes

`Dataset/`: `dataset_74` (220 tiles, `synthetic.json` com a configuração genérica), `dataset_74_2000`, `dataset_74_mar`,
`dataset_74_wu`, `dataset_74_wu_mar`, `dataset_wu` (220 tiles, o dado público de Wu et al.), `dataset_wu_mar`,
`dataset_regions` (430 tiles, o novo), `marlim_opt`, e `marlim/` com os quatro patches reais em `.dat`.

### 4.12 Literatura local

`Articles/` (gitignored, mas presente em disco): Fault-Seg-Net; "A hybrid network for three-dimensional seismic";
"Automatic fault detection on seismic"; FaultEdgeFormer; "Algorithm for Intelligent Recognition Low-Grade Seismic
Faults Using Codec Target Edges"; "Fault Detection on Seismic Structural Images Using a Nested Residual U-Net";
ResACEUnet; `s11831-026-10571-1.pdf`. Também `Synthetic/files/fault_seg.pdf`.

---

## 5. EVIDÊNCIA MEDIDA — O ESTADO ATUAL, EM NÚMEROS

Tudo abaixo foi extraído agora do repositório (saídas gravadas nos notebooks, CSVs e `info.json`). São **fatos**, não
interpretações.

### 5.1 Calibração da métrica (célula 9, real × real)

Cada coluna é uma amostra de `N_IMAGES` tiles reais de uma região comparada com os tiles reais da região da linha
(na diagonal a amostra sai da referência antes da conta):

```
real \ amostra        calm        dead     faulted
calm            99.0 ± 0.6  50.8 ± 3.0  79.1 ± 6.0
dead            52.2 ± 1.2  98.7 ± 1.4  73.8 ± 6.4
faulted         81.4 ± 1.4  73.4 ± 2.8  97.3 ± 1.9
```

A diagonal é o teto atingível; o fora da diagonal é o quanto a métrica separa regiões diferentes.

### 5.2 O resultado da última busca (célula 22) — **o dado mais importante desta seção**

```
  região   candidato     nota  nota na busca   sigma  sigma real  rótulo  amplitude  espectro  estrutura  continuidade   falhas
0   calm       busca  97.7841        98.6410  0.1843      0.1921  0.0000    97.3948   95.4524    99.7034       97.8442  98.2566
1   calm  referência  96.2579            NaN  0.1889      0.1921  0.0000    99.3736   94.6814    97.3050       93.3423  96.5710
2   dead       busca  91.6895        93.6815  0.0161      0.0161  0.0000    92.5061   89.0861    97.5058       88.0279  91.1542
3   dead  referência  87.9274            NaN  0.0161      0.0161  0.0000    92.8627   83.1919    96.0449       90.2404  82.6209
4 faulted       busca  85.1750        97.1105  0.1454      0.1457  0.0207    97.7632   89.7761    91.7533       81.4019  74.7218
5 faulted  referência  95.0598            NaN  0.1457      0.1457  0.0251    94.9550   93.6701    97.3959       98.2471  92.9627
```

Leia a linha 4: no `faulted` a busca marcou **97.11 nas sementes dela e 85.18 em sementes novas** — 12 pontos de
diferença. O vencedor escolhido no `faulted` foi a **referência**, não a busca. No `calm` o custo foi 0.9 ponto e no
`dead` 2.0.

Custo das buscas gravadas: `calm` 1600 avaliações em **5 h 55 min** (13.3 s/aval), `dead` 1600 em **6 h 13 min**,
`faulted` interrompido em **544/1600 (3 h 22 min)** — as saídas do notebook vêm de execuções diferentes, então os
números do `tqdm` e da tabela final não são da mesma corrida. Memórias em disco:
`calm_4c819c62` 99.16, `calm_210eb417` 98.82, `dead_3e9b0d4f` 96.57, `dead_6a58b72e` 93.37,
`faulted_85e96199` 97.74, `faulted_26a6ccbd` 97.11.

### 5.3 O dataset gerado (célula 25)

```
         tiles     nota   sigma  sigma real  rótulo
calm      50.0  96.7933  0.1920      0.1921  0.0000
dead      50.0  93.1701  0.0161      0.0161  0.0000
faulted  330.0  94.2164  0.1453      0.1457  0.0278
```

O σ bate o real nas três regiões. **100 dos 430 tiles (23%) não têm nenhum voxel de falha rotulado**
(`calm` e `dead` com `faultCount` em (0,1)).

### 5.4 Treino

**Nenhum modelo foi treinado no `dataset_regions` ainda.** Os 22 modelos em `Model/Backup` usam `dataset_wu` (21) e
`dataset_74` (1). `Task/info.json` e `Task/task.json` apontam hoje para `dataset_74`.

IoU de teste dos modelos mais recentes: `model_19` resaceunet_grva/wu **0.7791**; `model_20` macnn/wu 0.7398;
`model_21` unet3d_v2/wu 0.7395; `model_22` macnn/`dataset_74` 0.7199; melhor histórico `model_16` unet3d_v2/wu 64³
0.7916 (**mas esse tem vazamento de split — ver §6**).

### 5.5 A predição no Marlim — `Marlim/files/comparisons/sticks_report_backup.csv`

```
modelo,rede,janela,iou_teste,overlap,patch,fatia,recall,deteccao,precisao,f1,sticks,falhas_gt,len_p50,len_total,len_total_gt
model_22,macnn,128³,0.7199,0.25,1200,16,0.588,0.545,0.283,0.382,63,33,223.1,17600,7454
model_19,resaceunet_grva,128³,0.7791,0.25,1200,16,0.396,0.273,0.182,0.250,64,33,198.5,17177,7454
model_21,unet3d_v2,128³,0.7395,0.25,1200,16,0.301,0.212,0.424,0.352,19,33,217.0,5294,7454
model_20,macnn,128³,0.7398,0.25,1200,16,0.273,0.212,0.479,0.348,18,33,205.5,4168,7454
model_16,unet3d_v2,64³,0.7916,0.25,1200,16,0.170,0.152,0.713,0.275,8,33,187.0,1662,7454
model_17,unet3d_v2,32³,0.7863,0.25,1200,16,0.087,0.061,0.428,0.145,7,33,185.0,1430,7454
model_8,unet3d_v2,128³,0.7631,,1200,16,0.000,0.000,0.000,0.000,0,33,0.0,0,7454
model_16,unet3d_v2,64³,0.7916,0.25,1300,52,0.134,0.083,0.419,0.203,9,24,161.0,1600,5345
model_17,unet3d_v2,32³,0.7863,0.25,1300,52,0.120,0.083,0.599,0.200,6,24,162.5,1002,5345
model_16,unet3d_v2,64³,0.7916,0.25,1400,24,0.186,0.222,0.454,0.264,11,18,186.0,2081,5605
model_17,unet3d_v2,32³,0.7863,0.25,1400,24,0.074,0.111,0.457,0.127,4,18,195.0,756,5605
model_16,unet3d_v2,64³,0.7916,0.25,2600,8,0.097,0.000,0.684,0.170,10,30,158.6,1687,13087
model_17,unet3d_v2,32³,0.7863,0.25,2600,8,0.048,0.033,0.696,0.091,5,30,160.0,843,13087
```

O melhor F1 de todos é **0.382**. O `iou_teste` **não ordena** o recall real: `model_16` tem o melhor IoU sintético
(0.7916) e um dos piores recalls (0.170). O `model_22`, com o **pior** IoU da lista (0.7199), tem o melhor recall
(0.588) — e desenha 17 600 px de stick contra 7 454 px do especialista, ou seja, ganha recall desenhando 2,4× mais.

### 5.6 "As fotos anexadas" — onde está a mesma evidência, em disco

As imagens não chegaram na sessão. O equivalente exato está gravado (abra-as, elas são a prova visual do problema):

- `Model/Backup/model_{8,19,20,21,22}/marlim/patch_1200/predicted_seismic.png` — sísmica com sticks sobrepostos;
- `.../predicted_mask.png` — a máscara predita na inline anotada;
- `.../predicted_sticks.png` — sticks preditos (vermelho) × especialista (azul), com recall/detecção/precisão no título;
- `Marcia/model_*/marlim/patch_*/predicted_*.png` — o mesmo para os 10 modelos de referência;
- `Marlim/files/patches/<id>/<id>_interpretado.png` e `<id>.png` — a anotação e a sísmica;
- as figuras dentro de `Marlim/Regions/Synthetic/Analysis.ipynb` (células 7, 11, 27, 29, 31) — real × sintético por região;
- `Model/Backup/model_*/train.png` — curvas de treino.

Você **deve** olhar essas figuras antes de concluir qualquer coisa sobre "a natureza das predições do jeito que tá".

---

## 6. LIÇÕES JÁ MEDIDAS NESTE PROJETO — NÃO REPETIR O TRABALHO

Cada item abaixo custou horas de medição. Trate como dado, confira por amostragem se for reusar, e **não gaste
orçamento refazendo**:

1. **Piso de coerência do gerador.** O pipeline monta refletividade 1D e convolve em z, então o campo é localmente
   planar. Varrendo `foldSigma` (3–30) e `foldAmplitude` (até ±80), a coerência do tensor de estrutura nunca desceu de
   0.90; a zona morta real de Marlim tem 0.65. Falha densa com superfície corrugada (8–18 e 20–34 falhas/tile) **piorou**
   a textura. A alavanca que funcionou foi `noiseSigma` + `noiseLevel`.
2. **Mergulho aparente raso.** O azimute uniforme `(0, 2π)` faz o mergulho visto na seção ser `atan(tan δ · |sin φ|)`;
   ~13% das falhas cruzam a seção quase deitadas. O especialista não anota traço abaixo de ~39°.
3. **Botões de forma da falha não mudam o contraste da falha.** `faultRoughness` 8/14, `faultRoughSigma` 3,
   `faultDecaySigma` (15,30), `faultZoneWidth` 0.5, `faultThreshold` 1.5: menos de 0.6 ponto de nota cada, e a razão
   descontinuidade-sob-rótulo/fora ficou entre 16 e 18 (real 2.3). Quem controla essa razão é o **fundo**, não o corte.
4. **Lição de 2026-08-17:** calibrar a falha **só** pela estatística da máscara (fração, comprimento, mergulho) acertou
   os números pelo motivo errado — superfícies fragmentadas, rejeito com piso zero, rótulo bonito sobre imagem lisa.
5. **Warm start é obrigatório.** O CMA-ES do `Nature` sorteia a média inicial uniformemente na caixa (`born()`) e não
   tem API para ponto inicial. Com a caixa larga e ~26 variáveis, 32 avaliações chegaram a 72% no `calm`, contra 84%
   do genoma calibrado à mão.
6. **`gainJitter` precisa ser normalizado pela mediana** antes de resolver o ganho: com 18 tiles a mediana do sorteio
   chegou a 1.17 e o ganho absorvia o viés daquele lote (o dataset sairia 13–17% mais fraco que o real).
7. **Wavelet:** ela depende de `freq × dt` e o kernel de `duration/dt`. Valores calibrados com `dt=0.002` só valem com
   `dt=0.0012` multiplicando a frequência por 1.667 e a duração por 0.6.
8. **Trilhos do Format:** com ganho por região a saturação fica abaixo de 1%, então o percentil da amostra cairia
   dentro dos trilhos e mudaria com a contagem de tiles. Por isso o `Format` do `dataset_regions` usa o `clip`
   declarado no `synthetic.json`.
9. **Augmentação:** como o `Format` já grava em [0,1] e val/teste **não** passam por transform, usar `Normalize`/`Clip`
   só no treino dessincroniza as distribuições e derruba o IoU. Mexa em `Model/Transforms/index.py`
   (o `Model/Augmentor/index.py` é cópia antiga).
10. **`DiceFocalLoss` do MONAI 1.5.2:** o termo focal é sempre sigmoide, mesmo no caso multiclasse.
11. **Onde mora o erro no `dataset_wu`** (medido sobre `model_1`, 22 volumes de teste): detecção já resolvida (99,51%
    dos voxels de falha do GT a ≤3 voxels de alguma predição); **92,3% de todo o erro é de borda**; falha genuinamente
    perdida = 5,9% dos FN, alucinação = 9,2% dos FP; o threshold 0.5 já é ótimo e a curva é chata (0.771 em 0.2 a
    0.764 em 0.8). O IoU ali é proxy de precisão sub-voxel: o `model_1` posiciona a superfície com ~0,40 voxel RMS.
12. **Ruído entre rodadas idênticas ≈ 0,03 IoU** (model_1 0.7784, model_15 0.7706, Marcia/model_2 0.7491, mesma
    config). **Diferença menor que isso não é resultado.** lr é chato de 1e-3 a 1e-2 e dropout de 0.01 a 0.3.
13. **`model_16` e `model_17` têm vazamento de split** — o `TilesBuilder` corta os volumes em tiles numa pasta única e
    o `train_test_split` é por tile, sem agrupar por volume de origem. Não use o IoU deles como referência.
14. **Janela pequena não transfere.** Média de 12 modelos 128³ nos 4 patches: recall 0.387 / precisão 0.312 / 53 sticks;
    `model_16`/`model_17` (64³/32³): recall 0.090 / precisão 0.590 / 6 sticks. A máscara de janela pequena sai
    **fragmentada** (806 blobs, mediana 8 px) e o extrator descarta quase tudo.
15. **Extrator de sticks (v6):** a arquitetura atual (votação Hough → traçado com lacunas → PCA → matching pursuit)
    venceu, por medição em 10 pares (modelo, patch), quatro alternativas: fragmentos do esqueleto + ligação colinear;
    resposta orientada + ligação; RANSAC sobre fragmentos; traçar sobre a resposta orientada. Tetos de cobertura da
    anotação pela própria predição dilatada 12 px: 1200 0.815, 1400 0.774, 2600/1300 ~0.75 — o extrator entrega
    65–70% desse teto.
16. **Geometria da anotação do especialista** (4 patches): 33/24/18/30 sticks; comprimento mediano 220–440 px;
    mergulho p5/p50/p95 = 40/65/77°; ~80% da família mergulha para a direita; ele anota **só as falhas principais**,
    então a densidade anotada é **piso, não alvo**.
17. **Fração de rótulo dos datasets que transferem** (seção central): `dataset_74` 7.3%, `dataset_wu` 7.6%,
    Marlim `faulted` pelo especialista 3.6%.
18. **Os 128 inlines de cada `.dat` são reais** (o `original_shape` 64 é só o miolo); a água é a constante 0.50263; a
    xline 0 é traço morto em todos os patches (no 2600, também 0–6 e 2238–2239).

---

## 7. O CRITÉRIO DE ACEITAÇÃO

A sequência que o usuário vai executar, na ordem, é:

```text
1. Marlim/Regions/Synthetic/Analysis.ipynb    (otimiza os parâmetros por região e grava Dataset/dataset_regions)
2. Dataset/dataset_regions/Format.ipynb       (normaliza, monta DataBase.csv, aponta Task/info.json)
3. cd Task && python index.py                 (treina, salvando em Model/Backup/model_N)
4. Marlim/1 - Predict.ipynb                   (prediz nos patches 1200/1300/1400/2600)
5. Marlim/2 - Analysis.ipynb                  (sticks + comparação com o especialista + CSV)
```

O trabalho só está pronto quando:

1. **A sequência roda de cima para baixo, sem edição manual entre etapas**, no `torch-gpu`, com os caminhos relativos
   corretos, e cada notebook termina com prova visível do que produziu.
2. **Existe uma métrica objetiva e reprodutível** guiando a otimização dos parâmetros do gerador, cuja validade você
   **demonstrou** — não apenas descreveu. Demonstrar significa: dizer o que ela mede, medir o teto e o piso dela,
   medir o ruído dela, medir a monotonicidade dela sob degradações controladas, e mostrar que ela ordena corretamente
   casos cuja ordem é conhecida de antemão.
3. **Você mediu a relação entre o que a métrica otimiza e o que o usuário cobra** (§5.5). Se a relação for fraca,
   isso é um resultado a reportar e a atacar, não um detalhe.
4. **A busca explora e explota de forma defensável**: o gap entre a nota na busca e a nota fora dela (hoje até 12
   pontos, §5.2) é quantificado e tratado; o orçamento de avaliações é justificado; a campanha é retomável e monótona
   (rodar de novo nunca piora).
5. **Há um antes-e-depois numérico** de ponta a ponta: o estado atual (§5) contra o estado após as suas mudanças, nas
   mesmas condições, com o ruído de medição declarado (§6.12). Qualquer ganho reivindicado precisa ser maior que o
   ruído.
6. **Nada quebrou:** os contratos de arquivo (eixos `(x, z, y)` nos datasets, `(inline, z, xline)` nos tiles de
   região, escala [0,1], `DataBase.csv`, `synthetic.json`, `Task/info.json`, `Model/Backup/model_N/`,
   `marlim/patch_<id>/masks/*.dat` + `predict.json`) continuam válidos, e os datasets e modelos antigos continuam
   legíveis pelos mesmos notebooks.
7. **A documentação bate com o código.** O `Marlim/Regions/Synthetic/README.md` descreve o que existe, com as
   medições que o sustentam e as fontes verificadas.

---

## 8. COMO TRABALHAR

### 8.1 Raciocínio explícito antes de agir (obrigatório)

Antes de **qualquer** edição, escreva — em português, no seu texto de resposta — nesta ordem:

1. **Mapa:** o que cada peça da cadeia faz e onde ela pode estar quebrando, com o arquivo e a linha.
2. **Hipóteses:** as causas candidatas da divergência entre sintético e Marlim, e da descontinuidade dos traços
   preditos, **ordenadas pelo efeito esperado**, cada uma com a medição que a confirma ou a mata.
3. **Plano de medição:** qual experimento decide cada hipótese, quanto custa em minutos, e qual resultado numérico
   faria você mudar de ideia. Declare o critério de falseamento **antes** de rodar.
4. **Decisão:** o que você vai mudar, por quê, e o que vai deixar como está (com o motivo).

Depois de agir, mostre o resultado medido de cada experimento — inclusive os que refutaram a sua hipótese. Um
experimento que deu errado e foi reportado vale mais do que uma mudança sem medição.

### 8.2 O loop de auto-verificação (obrigatório, sem limite de tempo)

Para **cada** mudança, sem exceção:

```text
  projetar → simular mentalmente → implementar → executar → medir → comparar com o baseline
      ↑                                                                      │
      └──────────────── se houver qualquer erro, regressão ou dúvida ────────┘
```

O loop só termina quando o resultado é **exatamente** o esperado e você consegue explicar o número. Se algo divergir
um epsilon do que você previu, volte e entenda por quê antes de seguir — divergência não explicada é bug não achado.

> **You must think exhaustively, mentally simulate, self-critique, and iteratively refine over and over again—taking as
> much time as needed, even hours—until you are absolutely certain the solution is perfect and leaves zero doubt.**

Em português, para não restar dúvida: **pense exaustivamente, simule mentalmente, critique o seu próprio trabalho e
refine iterativamente, repetidas vezes, levando o tempo que for necessário — horas, se preciso — até ter certeza
absoluta de que a solução está perfeita e não deixa nenhuma dúvida.** Não há limite de tempo. Não entregue por
cansaço; entregue por convergência.

### 8.3 Regras de medição

- **Equivalência bit a bit primeiro.** Ao reescrever ou otimizar algo que já funciona, prove que a saída é idêntica
  (`max|Δ|`) antes de mudar o comportamento. O projeto já usa esse padrão (`max|Δ| = 1.2e-07`, `Δ = 0`).
- **Semente fixa em tudo que é aleatório**, e a semente declarada no resultado.
- **Nunca digite um número à mão.** Todo número do seu relatório sai de código que você rodou e que pode ser rerodado.
- **Amostra pequena engana.** Declare o desvio da sua medição; compare contra ele antes de chamar algo de ganho.
- **Cuidado com o custo.** Veja §8.4 antes de disparar qualquer coisa longa.

### 8.4 Orçamento real das execuções (medido)

| operação | custo medido |
|---|---|
| um tile 128³ do gerador | 6–25 s (`calm`/`dead` ~1.1 tile/s com 6 workers; `faulted` ~2.3 s/tile) |
| uma avaliação da métrica (lote + features) | 13–24 s com `WORKERS=6` |
| uma campanha CMA-ES de 1600 avaliações | **~6 h por região** |
| gerar `Dataset/dataset_regions` (430 tiles) | ~14 min |
| um treino completo (100 épocas, batch 2, 128³, `use_amp=False`) | **horas** |
| predição de um patch (910 tiles) no P6000 | 128³ 0.45 s/tile; 64³@0.25 1.30; 32³@0.25 0.75 → **11–20 min/patch** |
| `Marlim/2 - Analysis.ipynb` (remontar + sticks + figuras) | minutos por (modelo, patch) |

Regra do repositório (`CLAUDE.md`): **"Uma rodada custa horas; nunca re-execute um treino para 'verificar' sem pedir."**
Ao mesmo tempo, o usuário pediu explicitamente para você testar, explorar e simular. Resolva essa tensão assim:
tudo que é barato (segundos a minutos) você roda sem perguntar e o resultado entra no relatório; antes de disparar
algo que custe horas, **declare o plano, o custo estimado e o que ele decide**, e siga — mas deixe registrado o que
foi executado e o que não foi, e nunca reporte como medido algo que você não rodou. Rodadas longas em background devem
gravar em `Marlim/Regions/Synthetic/files/run/` (o `/tmp` é apagado no boot) e **nunca** duas buscas em paralelo
escrevendo no mesmo checkpoint.

---

## 9. AS PERGUNTAS QUE VOCÊ PRECISA RESPONDER COM DADOS

Responda cada uma explicitamente no relatório final, com a medição que a sustenta:

1. **A similaridade de imagem é a estratégia certa para otimizar os parâmetros do gerador?** Ela é a melhor aposta
   disponível, dado que o produto final é um detector de falhas e não um gerador de texturas? Qual é a evidência?
2. **O que a métrica atual não enxerga?** Onde ela pode dar nota alta a um lote que treina mal, ou nota baixa a um
   lote que treina bem? Meça, não suponha.
3. **A busca está explorando e explotando bem?** O gap busca × sementes novas (§5.2), a caixa `SPREAD=0.3` em volta da
   `REFERENCE`, as 23 variáveis escolhidas, o que ficou fora do genoma, a população 16, o CMA-ES contra as outras
   quatro opções do `Nature` — cada uma dessas escolhas se sustenta?
4. **O que falta em `Marlim/Regions/Synthetic/` para que a rede aprenda o que deve detectar em Marlim?** A resposta
   pode estar na métrica, na parametrização, na composição do dataset, na cadeia de normalização ou em mais de um
   lugar — descubra qual, medindo.
5. **Por que as falhas preditas saem descontínuas?** Isole a causa: é o dado sintético, a rede, a janela de inferência,
   a emenda, a extração de sticks, ou a própria comparação? Cada candidato tem um experimento que o isola.
6. **A composição atual do `dataset_regions`** (50/50/330, 23% dos tiles sem nenhum rótulo, σ casado por região)
   ajuda ou atrapalha o treino? Qual evidência sustenta a resposta?
7. **O `dataset_74` / `dataset_wu` / `Marcia/` como referência:** o que eles fazem que o `dataset_regions` não faz, e
   vice-versa? O que a comparação entre eles ensina sobre o que transfere?
8. **O que a literatura disponível** (`Articles/`, e o que você buscar) diz sobre treinar com sintético e transferir
   para dado de campo que este projeto ainda não está fazendo? Cite a fonte e diga se ela se aplica aqui, com o
   número que justifica.

Se, ao medir, você concluir que alguma escolha atual está certa e deve ficar, **diga isso com o número** — manter algo
por evidência é uma resposta tão válida quanto trocar.

---

## 11. ESTILO DE CÓDIGO — OBRIGATÓRIO

`CODE_STYLE.md` na raiz (922 linhas) **é a regra**. Leia antes de escrever a primeira linha, com atenção especial à
**§10 (notebooks)** e à **§14 (assinaturas de código gerado a remover)**. O essencial, para você não errar:

- **Módulo é pasta com `index.py`** exportando o tipo de mesmo nome (`from Nature.index import NatureSelector`).
  Importar módulo nunca executa trabalho.
- **Uma linha MAIÚSCULA em português acima de cada classe**, e acima só das funções cujo nome não diz por que existem.
  **Zero docstrings, zero type hints, zero banners, zero narração, zero underscore inicial, zero comentário dirigido
  ao leitor.**
- **Identificadores em inglês e `camelCase`**; comentários, markdown, títulos de gráfico, commits e relatórios **em
  português**. Chaves gravadas em JSON/CSV em `snake_case`; constantes ajustáveis em `UPPER_SNAKE_CASE`.
- **Vocabulário único de verbos:** `update()`, `get()/set()`, `info()`, `print()`, `plot()/showX()`, `process()`,
  `apply()`, `evaluate()`, `check()`, `reset()`, `start()/stop()`, `save()/load()/export()`. Sem sinônimos.
  **Construção só armazena;** o trabalho acontece em `update()`/`start()`/`plot()`.
- **Chamada nunca quebrada em várias linhas.** Uma etapa por linha, atribuições relacionadas alinhadas.
- **Notebook:** imports da biblioteca na primeira célula; célula markdown `# SEÇÃO EM MAIÚSCULAS` como cabeçalho, com
  bullets em português trazendo o fato ou a teoria (LaTeX quando couber) **antes** do código; **uma célula, uma etapa**;
  **toda célula que trabalha termina em prova visível** (o objeto como última expressão, `display(...)`, um gráfico ou
  um `print` curto); células silenciosas só para import e definição; fluxo linear, nunca embrulhado em `main()`.
- **Diferença vira dado:** blocos iguais a menos de valores viram uma peça mais uma tabela.
- **Reusar antes de criar.** Se já existe helper, classe ou padrão no projeto, use — não escreva uma segunda versão.
- **Obedeça a estrutura que já existe.** O usuário foi explícito: as mudanças entram na organização atual dos códigos,
  não numa nova.

---

## 12. FORMATO DA ENTREGA

Edite os arquivos **diretamente** no repositório (não devolva patches para o usuário aplicar). Ao final, produza um
**relatório em português** com, nesta ordem:

1. **Diagnóstico** — a causa raiz de cada problema, com o número que a comprova e o arquivo:linha onde ela vive.
2. **O que mudou** — tabela `arquivo | o que mudou | por quê | como foi verificado`, uma linha por mudança lógica,
   com o hash do commit correspondente.
3. **A metodologia entregue** — o que a métrica mede, por que essa e não outra (com a alternativa que você considerou
   e descartou, e o número que decidiu), como a busca explora e explota, e como alguém reproduz tudo do zero.
4. **Antes × depois** — tabela com as mesmas medições da §5 no estado antigo e no novo, o desvio de cada medição, e a
   declaração explícita de quais diferenças são maiores que o ruído e quais não são.
5. **O que foi executado e o que não foi** — lista honesta, com o custo de cada execução e o motivo de cada omissão.
6. **O que ficou em aberto** — o que você suspeita mas não conseguiu provar, e qual experimento provaria.
7. **Como rodar** — a sequência exata de comandos, do zero, com o tempo esperado de cada etapa.

Atualize também, no repositório: `Marlim/Regions/Synthetic/README.md` (para descrever o que existe, com as medições e
as fontes) e, se o contrato do pipeline mudar, o `CLAUDE.md`.

---

## 13. COMO UMA AFIRMAÇÃO SUA DEVE PARECER

**Inaceitável:**

> Melhorei a métrica adicionando atributos de falha, o que deve deixar o sintético mais parecido com o Marlim e
> melhorar a predição.

**Aceitável:**

> Medição em N sementes novas, `WORKERS=6`, semente base 20260915:
>
> | configuração | nota (± desvio) | grupo falhas | recall no 1200 | precisão no 1200 | len_total / len_gt |
> |---|---|---|---|---|---|
> | baseline (estado atual) | 94.22 ± 1.1 | 74.7 | 0.588 | 0.283 | 2.36 |
> | alteração X | ... | ... | ... | ... | ... |
>
> O ganho de recall é de A pontos, contra um ruído de rodada de 0,03 IoU / B pontos de recall medido em C repetições
> — portanto conta / não conta como resultado. O que refutou a hipótese Y foi: ...

**Inaceitável:**

> Os traços saem descontínuos porque o dataset sintético não tem falhas longas o bastante.

**Aceitável:**

> Contei os blobs conexos da máscara predita na inline anotada do 1200: N blobs, mediana M px, contra K px exigidos
> pelo `STICK_MIN`. Rodando o mesmo modelo com <condição isolada>, os blobs passaram a N' / M'. Isso descarta /
> confirma a hipótese porque ...

---

## 14. PRIMEIRO PASSO

Antes de propor qualquer coisa: leia `CODE_STYLE.md`, `CLAUDE.md`, `Marlim/Regions/Synthetic/README.md`,
`Synthetic/index.py`, as células de `Marlim/Regions/Synthetic/Analysis.ipynb`, `Dataset/dataset_regions/Format.ipynb`,
`Model/Analysis.ipynb`, `Marlim/1 - Predict.ipynb`, `Marlim/2 - Analysis.ipynb` e os módulos de `Nature/` e `Model/`;
**abra as figuras da §5.6**; e só então escreva o mapa e as hipóteses da §8.1.

Trabalhe até a convergência, não até o cansaço. Quando — e só quando — a cadeia inteira estiver medida, reprodutível
e demonstrada melhor que o estado atual, reporte.


Veja se o modo de objetivo do otimizador está bom , se nao vale a pena colocar um agregador multiobjetivo com pesos usando o modelo do wu pra fazer prediçãoe  calcular iou pra ver a natureza das falhas tambem, alguma coisa assim, mas pense muito bem antes de fazer , ao final tem que estar perfeito e testado e garantido que vai funcionar nao importa quanto tempo leve