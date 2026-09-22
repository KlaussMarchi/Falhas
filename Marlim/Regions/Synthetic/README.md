# Calibração do gerador sintético para o bloco de Marlim

Esta pasta escolhe os parâmetros do `SyntheticGenerator` com que o `Dataset/dataset_regions` é gerado.
O `Analysis.ipynb` faz isso com **uma sonda de transferência**: treina uma rede pequena em cada lote
sintético candidato e mede, com a métrica de sticks do projeto, o quanto ela acha das falhas que o
especialista anotou nas inlines de Marlim. A similaridade de imagem entre sintético e real, que era
o objetivo até 17/09/2026, continua no notebook como **diagnóstico**.

Este documento explica por que o objetivo mudou (com as medições que decidiram), o que a sonda mede,
quanto ela vale e o que se sabe sobre ela, como a busca usa a nota, e o que a similaridade ainda diz.

**Como rodar.** O `Analysis.ipynb` roda de cima para baixo. Entram os tiles reais de
`../Marlim/files` (similaridade), os tiles `.dat` de `Dataset/marlim/patch_<id>` e as anotações de
`Marlim/files/patches/<id>/<id>_interpretado.png` (sonda). Sai o `Dataset/dataset_regions/original` +
`synthetic.json`, que o `Format.ipynb` do dataset normaliza. A busca grava o estado a cada geração
em `files/memory/transfer_<hash>/`: interromper e rodar de novo retoma, e rodar uma campanha terminada
só carrega o resultado (`EXTEND = True` estende).

---

## 1. O problema

O que se cobra do dado sintético é **a rede treinada nele achar, no bloco real de Marlim, as falhas
que o especialista anotou** — medido no fim da cadeia pelo `Marlim/2 - Analysis.ipynb`, stick a
stick (recall, detecção, precisão, F1). A cadeia inteira é

    parâmetros do gerador → tiles sintéticos → dataset → rede treinada → predição no Marlim → comparação com o especialista

e até 17/09/2026 o que se otimizava era o **primeiro elo**: a semelhança de imagem entre um lote
sintético e os tiles reais de cada região (`calm`, `faulted`, `dead`). Nenhuma medição ligava esse
elo ao último.

## 2. A similaridade de imagem não serve de objetivo — medido

### 2.1 O caso que decidiu

O `dataset_regions` gerado pela busca por similaridade (50 `calm` + 50 `dead` + 330 `faulted`,
notas de 94–98% contra as regiões reais) treinou o `model_25` (`unet3d_v2`, `smooth_dice`, lr 1e-4,
100 épocas). No sintético ele é o melhor modelo da série (IoU de teste 0.8205). No Marlim:

| modelo | dado | IoU teste | recall | detecção | precisão | F1 (4 patches) |
|---|---|---|---|---|---|---|
| `Marcia/model_1` | `dataset_74` | 0.736 | 0.565 | 0.527 | 0.271 | **0.358** |
| `model_25` | `dataset_regions` (similaridade) | **0.8205** | 0.036 | 0.032 | 0.746 | **0.068** |

No 1200, com a mesma rede, loss e lr, o `model_23` (`dataset_74`) tem recall 0.487 e F1 0.354. A
predição do `model_25` **não tem resposta nem abaixo do limiar**: só 6.0% dos pixels do traço do
especialista passam de 0.05 de probabilidade num raio de 4 px, contra 62.8% do `model_23`, e o
fundo marcado é 0.05% contra 2.1%. Não é calibração — a rede não enxerga as falhas reais.

### 2.2 Similaridade contra resultado, em quatro dados

Nota de similaridade (36 tiles de cada dado contra a região `faulted` real) e o que a rede treinada
naquele dado faz no Marlim (pixel: fração do traço do especialista com predição ≥ 0.5 a até 4 px, no 1200):

| dado | similaridade `faulted` | `unet3d_v2` no 1200: recall de pixel |
|---|---|---|
| `dataset_regions` | **94.4** | **0.056** |
| `marlim_opt` (misturado ao `74`/`wu`) | 88.4 | 0.38 / 0.32 |
| `dataset_74` | 81.6 | 0.60–0.61 |
| `dataset_wu` | 78.8 | 0.44–0.51 |

**Quanto mais o dado se parece com Marlim pela régua, pior a rede no Marlim.** O mesmo aparece nas
misturas antigas da `Marcia/`: juntar `marlim_opt` (o gerador anterior calibrado para Marlim) ao
`dataset_74` ou ao `dataset_wu` baixa o recall de pixel de 0.60 para 0.38 e de 0.51 para 0.32.

### 2.3 A alternativa testada: IoU de um modelo treinado no Wu sobre o sintético

A ideia era medir a "natureza das falhas" do lote pelo IoU que uma rede treinada no `dataset_wu`
consegue nele. Medido com duas redes (`model_21`, `Marcia/model_2`), 20–24 tiles de cada dado:

| dado | IoU `model_21` | IoU `Marcia/model_2` | F1 3D no Marlim |
|---|---|---|---|
| `dataset_wu` (teste) | 0.761 | 0.766 | 0.35 |
| `dataset_74` | 0.620 | 0.622 | 0.35–0.37 |
| `dataset_regions/faulted` | 0.609 | 0.605 | **0.07** |
| `marlim_opt` | 0.187 | 0.200 | — |

O IoU pelo modelo Wu dá ao `dataset_regions` a mesma nota do `dataset_74` — ele **aprovaria o dado
que fracassou**. As falhas rotuladas do `regions` são tão visíveis quanto as do `74`; o defeito dele
não está no rótulo das falhas. O sinal que separa está no sentido inverso: o `model_25` (treinado no
`regions`) marca IoU 0.86 no próprio domínio, 0.49 no `dataset_74` e 0.25 no `dataset_wu`, enquanto o
`model_23` (treinado no `74`) marca 0.71 no `regions` e 0.68 no `wu`. **O que prevê a transferência
é treinar no candidato e testar fora dele**, não testar o candidato com uma rede de fora.

O mesmo vale para o IoU **no próprio sintético**. Nas 15 configurações medidas com a sonda (seção 4), o IoU dela nos
tiles sintéticos que ficaram fora do treino não tem correlação com o F1 no Marlim (Spearman −0.21, p = 0.46): a
estratigrafia do `74` sobre o ruído da similaridade dá IoU 0.617 — o mesmo do `dataset_wu` — e F1 0.009. É o que o
`model_25` já mostrava em 3D (IoU de teste 0.82, o maior da série; F1 0.068 no Marlim).

### 2.4 Por quê

Pela cota de Ben-David et al. (2010), o erro no domínio real de uma rede treinada no sintético é
limitado por $\epsilon_S + d(\mathcal{D}_S, \mathcal{D}_T) + \lambda$: o erro no sintético, a distância
entre as imagens e $\lambda$, o erro da melhor rede nos dois domínios ao mesmo tempo. A similaridade
só ataca $d$. $\lambda$ depende de o rótulo sintético marcar o que no real se chama falha, e a régua
não vê o rótulo. A busca por similaridade encheu o lote de textura parecida com a de Marlim (ruído
liso e correlacionado, contraste baixo, tiles inteiros sem falha) — e a rede aprendeu que textura
parecida com Marlim **não é falha**.

É o mesmo diagnóstico de *Learning to Simulate* (Ruiz et al., 2019) e *Meta-Sim* (Kar et al., 2019):
os parâmetros de um simulador devem ser escolhidos pelo desempenho, num conjunto real rotulado, do
modelo treinado nos dados dele — não por imitar a distribuição das imagens reais.

## 3. A sonda de transferência

### 3.1 O que ela mede

Para um lote sintético (`N_TILES` = 44 tiles de um genoma):

1. a `ProbeNet` — U-Net 3D de quatro níveis (8, 16, 32, 64 filtros, 0.37 M parâmetros), `GroupNorm`,
   LeakyReLU e pool que só reduz a inline a partir do segundo nível, o mesmo desenho da `Unet3D_V2` —
   treina `STEPS` = 2000 passos em blocos de 32 inlines × 128 × 128 sorteados de 40 tiles (lote de 2,
   AdamW, OneCycle até 2e-3, perda BCE + dice, espelhamento em xline e inline);
2. ela prediz os blocos **reais** de 32 inlines × 1601 × 2240 em volta da inline anotada de cada
   patch, remontados dos `.dat` como o `FaultComparer` remonta o slab, em janelas de 128 com passo 64
   e peso Hann na emenda (o `SlidingWindow` do `1 - Predict.ipynb`), e fica a inline central;
3. o `FaultComparer` do projeto — **o mesmo** que monta a tabela final (`Marlim/FaultComparer/index.py`)
   — extrai os sticks e mede recall, detecção, precisão e F1 contra o especialista;
4. a nota é o **F1 médio nos `PATCHES`** (1200, 1300, 2600); o 1400 (`HOLDOUT`) nunca entra na busca;
5. os 4 tiles que sobram medem o IoU da sonda no próprio sintético (`iou`): diz se a tarefa é
   aprendível, não se transfere.

O contexto de 32 inlines não é detalhe. Uma primeira versão 2D (seções soltas, U-Net 2D) ordenava os
datasets pelo recall como a rede completa, mas disparava em 43% do fundo filtrado contra 13% do
`model_23`, e pelos sticks punha o `dataset_wu` à frente do `dataset_74` com folga — a rede 3D rejeita
textura pela continuidade entre inlines, e a 2D não tem como.

Custo medido no P6000: 2000 passos em ~290 s, predição dos 4 patches em ~52 s e sticks em ~20 s —
**~6 min por avaliação**, com os tiles do próximo genoma sendo gerados na CPU enquanto isso.

### 3.2 Validação contra a rede completa

A sonda só serve de objetivo se ordenar os dados como a `Unet3D_V2` completa (100 épocas) ordena no
Marlim. Mesma sonda, 44 tiles de cada dado, semente 0:

| dado | F1 da sonda (4 patches) | F1 da sonda (1200) | F1 da rede completa (1200) |
|---|---|---|---|
| `dataset_regions` (similaridade) | **0.063** | 0.084 | **0.072** (`model_25`) |
| `marlim_opt` | 0.221 | 0.250 | — |
| `dataset_74` | 0.277 | 0.290 | 0.354 (`model_23`), 0.373 (`Marcia/model_1`) |
| `dataset_74` + `marlim_opt` | 0.291 | 0.260 | 0.388 (`Marcia/model_5`) |
| `dataset_wu` | 0.318 | 0.307 | 0.352 (`model_21`), 0.347 (`Marcia/model_2`) |

A sonda **separa com folga o dado que fracassa dos que funcionam** — o `dataset_regions` cai para o mesmo
nível na sonda (0.06–0.08) e na rede completa (0.07), oito desvios abaixo do `dataset_74`, antes de
gastar as 16 h de treino que o `model_25` custou. Entre os dados que funcionam, a rede completa fica em 0.35–0.39 e a sonda em
0.26–0.32; essa faixa é da ordem do ruído dos dois lados, e nenhum dos dois ordena os bons entre si
com confiança. A sonda subestima o F1 dos dados bons (é uma rede de 0.37 M parâmetros com 2000
passos), mas a ordem dos casos separáveis é a mesma, e o recall × precisão também: o `dataset_74`
treina uma rede que desenha mais e acerta mais (recall 0.36, precisão 0.23 na sonda; 0.49 e 0.28 na
completa), o `dataset_wu` uma mais contida (0.30 e 0.37; 0.30 e 0.42).

A célula `VALIDAÇÃO DA SONDA` do `Analysis.ipynb` refaz a checagem a cada execução, com sementes fixas (reproduz
bit a bit). Na execução de 19/09/2026:

| dado | nota (F1 nos `PATCHES`) | F1 da sonda (1200) | F1 da rede completa (1200) |
|---|---|---|---|
| `dataset_74` | 0.346 | 0.300 | 0.354 |
| `dataset_wu` | 0.291 | 0.287 | 0.352 |
| `SIMILAR` (o dado do `model_25`, regerado das opções dele) | **0.105** | 0.179 | **0.072** |
| `REFERENCE`, três sementes | 0.329 / 0.253 / 0.327 | | |

### 3.3 Ruído, piso e teto

**Ruído.** Seis avaliações do `dataset_74` com sorteios diferentes: F1 dos `PATCHES` 0.285 ± 0.026. Com os
mesmos tiles e só a semente do treino mudando o desvio já é 0.031 — o ruído é o treino, não o lote. O
patch de teste, sozinho, oscila mais (±0.039). Por métrica, nas mesmas avaliações:

| métrica da sonda | desvio entre sementes | desvio entre dados | razão |
|---|---|---|---|
| **F1** | 0.026 | 0.106 | **4.1** |
| recall | 0.094 | 0.133 | 1.4 |
| detecção | 0.101 | 0.123 | 1.2 |
| recall de pixel (≥ 0.5 a 4 px) | 0.108 | 0.144 | 1.3 |
| AUC de pixel | 0.035 | 0.023 | 0.7 |

O recall sozinho oscila 3.6 vezes mais que o F1: de uma semente para outra a sonda anda ao longo da
curva recall × precisão, e o F1 é quase invariante a esse deslocamento. É a métrica com melhor relação
sinal/ruído que ainda mede o que se cobra (a precisão sozinha tem razão maior, mas premiaria o
`dataset_regions`, que tem a maior precisão de todas — 0.67 — sem detectar nada).

**Piso.** Um lote que não ensina nada dá F1 perto de zero: a estratigrafia e a wavelet do `74` sobre o
ruído da similaridade dão 0.009. **Teto.** Não há teto medido acima dos dados reais do projeto; o melhor
F1 da sonda até aqui é o do `dataset_wu`, 0.318, e o da rede completa 0.407 (`model_24`, Fault-Seg-Net).

**Reprodutibilidade.** A sonda é determinística por semente (cuDNN determinístico, sorteios por
`default_rng(seed)`), e a mesma configuração do gerador, com sementes diferentes, reproduz o
`dataset_74` gravado a 0.004 de F1 (0.281 contra 0.277).

**Tiles sem falha, de novo, sobre a configuração boa.** Trocando 8 dos 40 tiles de treino por 4 `calm`
e 4 `dead` do `dataset_regions` antigo:

| configuração | F1 | recall | precisão |
|---|---|---|---|
| `74` com z-score | 0.281 | 0.357 | 0.248 |
| `74` com z-score + 20% de tiles sem falha | 0.282 | 0.272 | 0.318 |
| `74` com ganho por região | 0.244 | 0.239 | 0.262 |
| `74` com ganho por região + 20% de tiles sem falha | 0.209 | 0.164 | 0.313 |

Os tiles sem falha não melhoram o F1 em nenhum dos dois casos (diferenças dentro do ruído de 0.026) e
tiram 0.08 de recall — trocam detecção por precisão, e o que faltava no Marlim era detecção. O dado
novo não tem tile sem falha.

## 4. O que a sonda mostrou sobre o gerador

Ablações com a sonda (semente 0, 2000 passos, 40 tiles de treino, F1 médio nos 4 patches), partindo
da configuração `faulted` que a busca por similaridade escolheu e trocando **um grupo de parâmetros
por vez** pelo valor do `dataset_74`:

| variante | F1 | recall | precisão | o que mostra |
|---|---|---|---|---|
| `dataset_regions` inteiro (o dado do `model_25`) | 0.063 | 0.033 | 0.733 | reproduz o fracasso do 3D (0.068) |
| `faulted` + estratigrafia e wavelet do `74` | **0.009** | 0.004 | 0.250 | interação forte: sozinha, a wavelet fina derruba tudo |
| `faulted` com z-score no lugar do ganho | 0.109 | 0.060 | 0.747 | normalização não é o problema |
| `faulted` só, sem os tiles `calm` e `dead` | 0.164 | 0.096 | 0.568 | os 23% de tiles sem falha derrubam o recall |
| `faulted` + dobra do `74` | 0.198 | 0.138 | 0.437 | efeito pequeno |
| `faulted` + 5–9 falhas por tile (era 2–3) | 0.264 | 0.256 | 0.278 | mais falha por tile, como pede Wu et al. (2019) |
| `faulted` + ruído do `74` | 0.285 | 0.187 | 0.616 | o ruído liso da busca imitava falha |
| configuração do `74` regerada, z-score | 0.281 | 0.357 | 0.248 | — |
| `dataset_74` (arquivos) | 0.277 | 0.363 | 0.233 | o gerador reproduz o dataset a 0.004 |
| `dataset_wu` (arquivos) | 0.318 | 0.297 | 0.366 | — |

Três conclusões saem daqui e entram no desenho:

- **Todo tile leva falha.** Tile sem falha no lote não ensina a rede a ignorar o fundo de Marlim: ensina
  que o fundo de Marlim não tem falha. É também a recomendação do FaultSeg3D ("images with more
  faults are more effective than those with fewer faults... we add more than five faults within a
  training image", Wu et al., 2019).
- **O ruído liso e correlacionado que a similaridade pedia é o que mais atrapalha** — sozinho, trocá-lo
  sobe o F1 de 0.16 para 0.29. Ele reproduz a textura descontínua de Marlim sem rótulo, e a rede
  aprende a ignorar descontinuidade.
- **Os fatores não somam.** A wavelet fina do `74` é boa com o ruído do `74` (0.28) e péssima com o ruído
  da similaridade (0.009). É por isso que a busca mexe no gerador inteiro de uma vez, em vez de ajustar
  um fator de cada vez.

**Normalização.** A `Unet3D_V2` e a sonda começam com convolução sem bias seguida de `GroupNorm`, então
multiplicar a entrada por uma constante não muda a saída. Casar o σ de cada região real só mexe em
**quanto satura**: com o ganho do `faulted` (0.131) e jitter 0.52, um tile de jitter alto satura já em
|z| > 1.2 e ceifa as ondículas. O dado novo usa ganho 1 sobre o tile z-scorado e trilho `RAIL` = 2.42
(o p01/p99 que o `Format.ipynb` do `dataset_74` mediu), o que satura ~1.5% dos voxels.

## 5. A busca

CMA-ES (`NatureSelector('genetic')`, Hansen & Ostermeier 2001 com IPOP) sobre **25 variáveis do
gerador inteiro** — estratigrafia, dobra, mergulho regional, número, rejeito e mergulho das falhas,
wavelet e nível e grão do ruído —, normalizadas em [0, 1] numa caixa em volta da `REFERENCE`
(meia-largura `SPREAD` = 0.25 da faixa global, no mínimo ±1 nas inteiras). Fora do genoma, no padrão
da classe: rugosidade e alcance do rejeito, espessura e limiar do rótulo e a falha lístrica — os
botões de forma que já tinham sido medidos sem efeito sobre o contraste da falha.

Cinco decisões, cada uma contra um defeito medido da busca anterior:

- **A primeira nuvem nasce na `REFERENCE`.** O CMA-ES do `Nature` sorteava a média inicial na caixa;
  com avaliação de minutos isso joga fora gerações inteiras. O `CMAES` ganhou o argumento `mean`
  (só a primeira nuvem; os reinícios IPOP continuam sorteando), conferido bit a bit contra a versão
  anterior quando não é passado. A `REFERENCE` é a configuração do `dataset_74`, a de melhor
  transferência medida em 3D.
- **Sementes comuns dentro da geração, sementes novas entre gerações.** O CMA-ES só usa a ordem dos
  indivíduos dentro da geração; todos recebem as mesmas sementes de tiles e de treino (números
  aleatórios comuns), o que tira da comparação o ruído que não depende do genoma. A semente vem do
  hash da população, então cada geração sorteia de novo e a retomada repete a mesma. A busca antiga
  usava sempre as mesmas sementes, e decorava o lote: 97.1 na busca contra 85.2 fora dela no `faulted`.
- **A decisão é refeita fora da busca.** A melhor nota de uma busca ruidosa é otimista por construção
  — a maldição do otimizador (Smith & Winkler, 2006). As `MÉTRICAS FINAIS` reavaliam a `REFERENCE`, o
  melhor indivíduo e a média final da nuvem nas mesmas `CHECK_SEEDS`, que a busca nunca usou.
- **A `REFERENCE` só perde por mais que o erro.** Fica o candidato de maior nota média, desde que o
  ganho pareado sobre a `REFERENCE` passe do erro padrão desse ganho; senão fica a `REFERENCE`. Rodar
  de novo nunca produz um dado pior que o de referência.
- **Um patch fica de fora.** O 1400 não entra em nenhuma nota da busca; ele é o teste de que o gerador
  não se ajustou às anotações dos outros três.

## 6. A campanha de 19-20/09/2026

64 avaliações (8 indivíduos × 8 gerações, 8 h 06 min), caixa de ±25% em volta da configuração do `dataset_74`.
Melhor nota de cada geração: 0.350, 0.338, 0.313, 0.324, 0.349, 0.338, 0.337 — **plana dentro do ruído** (σ 0.03) em volta
do nível da referência (0.305 ± 0.020 em 4 sementes). O passo do CMA-ES caiu de 0.25 para 0.19, ou seja, a nuvem se
fechou sem achar direção de ganho.

Decisão final, em 4 sementes que a busca nunca viu, com os três candidatos nas mesmas sementes:

| candidato | nota | holdout (1400) | recall | precisão | ganho sobre a referência | erro do ganho |
|---|---|---|---|---|---|---|
| **referência** (config do `dataset_74`) | **0.3050** | 0.2175 | 0.418 | 0.263 | — | — |
| melhor indivíduo da busca | 0.2857 | 0.2360 | 0.404 | 0.232 | −0.019 | 0.007 |
| média final da nuvem | 0.3174 | 0.2682 | 0.426 | 0.264 | +0.012 | 0.025 |

O melhor indivíduo marcou **0.3527 dentro da busca e 0.2857 fora dela** — a maldição do otimizador medida em casa: a
vantagem dele era ruído, e ele fica **abaixo** da referência em sementes novas. A média da nuvem ficou 0.012 acima da
referência, menos que o erro padrão do ganho (0.025), então a regra manteve a referência. **O dado gravado é a
configuração do `dataset_74`, com 220 tiles, todos com falha, z-scorados e saturados em ±2.42.**

Isso é um resultado, não um fracasso da busca: em 64 avaliações a vizinhança da melhor configuração conhecida não tem
ganho acima do ruído da sonda, e a metodologia **impediu** que um ganho aparente de 0.05 virasse um dataset pior.
Quem quiser continuar tem `EXTEND = True` (a campanha retoma do checkpoint) e o caminho medido para baixar o ruído:
avaliar cada genoma em duas sementes custa o dobro e divide o desvio por 1.4.

Verificação no dado gravado (sonda treinada nos 220 tiles, 4 patches):

| patch | recall | detecção | precisão | F1 |
|---|---|---|---|---|
| 1200 | 0.462 | 0.455 | 0.221 | 0.299 |
| 1300 | 0.568 | 0.500 | 0.167 | 0.258 |
| 2600 | 0.477 | 0.567 | 0.321 | 0.384 |
| **1400 (fora da busca)** | 0.424 | 0.444 | 0.173 | 0.246 |

A similaridade do dado gravado contra as regiões reais é 77.6 (`faulted`), 61.6 (`calm`) e 61.2 (`dead`) — abaixo dos
94.4 do dado que a busca por similaridade gerava, que é exatamente o ponto.

### 6.1 O dado novo na rede completa

`Dataset/dataset_regions/Format.ipynb` → `Task/index.py` → `Marlim/1 - Predict.ipynb` → `Marlim/2 - Analysis.ipynb`,
com a mesma rede, loss e lr do `model_23` e do `model_25` (`unet3d_v2`, `smooth_dice`, lr 1e-4, 100 épocas, 220 tiles,
8 h 13 min de treino). Média dos quatro patches, contra a anotação do especialista:

| modelo | dado | IoU no sintético | recall | detecção | precisão | F1 |
|---|---|---|---|---|---|---|
| **`model_26`** | **`dataset_regions` novo** | 0.704 | **0.520** | **0.515** | 0.236 | 0.318 |
| `model_23` | `dataset_74` | 0.714 | 0.488 | 0.471 | 0.263 | 0.334 |
| `model_25` | `dataset_regions` antigo (similaridade) | 0.821 | 0.036 | 0.032 | 0.747 | 0.068 |

Por pixel, a cobertura do traço anotado (predição ≥ 0.5 a até 4 px) vai de **0.073 para 0.584** e o fundo marcado de
0.008 para 0.173; o trecho coberto mediano vai de 9.5 px para 15.9 px. Contra o `dataset_74`, que era o melhor dado
conhecido, o novo empata dentro do ruído (recall +0.03, F1 −0.016, com o espalhamento entre redes no mesmo dado em
0.35–0.41 de F1). O IoU no sintético anda ao contrário do resultado mais uma vez: 0.821 no dado que não transfere,
0.704 no que transfere.

A sonda acertou a ordem dos patches para este dado: F1 previsto 0.384 (2600) > 0.299 (1200) > 0.246 (1400) > 0.258
(1300); medido na rede completa 0.415 > 0.344 > 0.273 > 0.239.

## 7. A similaridade como diagnóstico

A classe `ImageSimilarity` continua no notebook, com a mesma régua de 33 atributos, o mesmo
coeficiente de energia e a mesma calibração real × real — ela diz **o quanto um lote se parece com
cada região**, o que continua útil para ler um dado (é ela que mostra, por exemplo, que o sintético
novo é mais contrastado e mais limpo que o `faulted` real). O que mudou é o papel: ela não guia mais
a busca, pelas medições da seção 2.

A estatística é a de Rizzo & Székely (2016): para cada atributo,

$$H = \frac{2A - B - C}{2A + \varepsilon}, \qquad \mathrm{sim} = 100\,(1 - H),$$

com $A$ a distância média entre os dois lados, $B$ e $C$ as dispersões internas e
$\varepsilon = 0.25 \times \max(\mathrm{IQR}_{\text{região}}, 0.05 \times \mathrm{IQR}_{\text{bloco}})$;
média aritmética dentro de cada grupo (`amplitude` 0.15, `espectro` 0.20, `estrutura` 0.20,
`continuidade` 0.15, `falhas` 0.30) e média geométrica ponderada entre grupos. Calibração real × real
(`Analysis.ipynb`, célula de calibração): diagonal 97–99, regiões vizinhas 73–81, opostas 51–52.

## 8. Limites conhecidos

- **A anotação é parcial.** O especialista anota só as falhas principais, então a precisão é limite
  inferior, e o F1 favorece um detector que marque só o que ele marcaria. É o que se quer
  reproduzir, mas uma falha secundária verdadeira achada pela rede conta como erro.
- **Quatro seções.** A nota vem de três inlines anotadas (a quarta é o teste). O gerador tem 25
  variáveis globais e as três seções têm 87 falhas anotadas, então o risco de sobreajuste é pequeno,
  mas não é zero — é para isso que o 1400 fica de fora.
- **A sonda não é a rede.** Ela ordena os dados como a `Unet3D_V2` completa nos casos medidos, mas
  subestima o F1 dos dados bons (0.28 contra 0.36 no `dataset_74`) e não resolve diferenças menores
  que o ruído dela; diferença pequena entre dois dados bons só se decide treinando a rede completa.
- **Mergulho aparente.** O azimute das falhas continua uniforme em `applyFault`, então parte das
  falhas cruza a inline quase deitada; o especialista não anota abaixo de ~40°.
- **O traço predito sai picado porque o dado real é assim.** Ao longo do traço do especialista no 1200, onde a rede
  cobre, a descontinuidade da sísmica (1 − semblance com mergulho, máximo em 4 px) é 0.203; onde não cobre, 0.082, com a
  mesma amplitude local. As bordas das lacunas caem nas emendas dos tiles na taxa do acaso (0.130 contra 0.150), então
  não é artefato de remontagem. É o `FaultStickExtractor`, que costura lacunas de até 59 px, que fecha o traço.
- **A busca não resolve diferenças pequenas.** Com ruído de 0.03 por avaliação, distinguir dois dados que diferem 0.02
  exigiria repetir cada avaliação; a campanha de 64 avaliações mede isso e para na referência quando não há ganho.

## 9. Fontes

Verificadas durante este trabalho:

- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. & Vaughan, J. W. (2010). *A theory
  of learning from different domains*. **Machine Learning** 79, 151–175 — a cota do erro no alvo.
- Ruiz, N., Schulter, S. & Chandraker, M. (2019). *Learning to simulate*. ICLR 2019, arXiv:1810.02513 —
  ajustar os parâmetros de um simulador pelo desempenho do modelo treinado nele, "rather than
  mimicking the real data distribution".
- Kar, A. et al. (2019). *Meta-Sim: learning to generate synthetic datasets*. ICCV 2019,
  arXiv:1904.11621 — distância de distribuição mais um meta-objetivo de desempenho num conjunto
  real rotulado.
- Esteban, C., Hyland, S. L. & Rätsch, G. (2017). *Real-valued (medical) time series generation with
  recurrent conditional GANs*. arXiv:1706.02633 — a avaliação TSTR (treinar no sintético, testar no real).
- Smith, J. E. & Winkler, R. L. (2006). *The optimizer's curse: skepticism and postdecision surprise in
  decision analysis*. **Management Science** 52(3), 311–322 — por que a melhor nota da busca é otimista.
- Tobin, J. et al. (2017). *Domain randomization for transferring deep neural networks from
  simulation to the real world*. IROS 2017, 23–30.
- Wu, X., Liang, L., Shi, Y. & Fomel, S. (2019). *FaultSeg3D*. **Geophysics** 84(3), IM35–IM45 —
  mais de cinco falhas por imagem de treino (p. IM37).
- Wu, X., Geng, Z., Shi, Y., Pham, N., Fomel, S. & Caumon, G. (2020). *Building realistic structure
  models to train convolutional neural networks for seismic structural interpretation*.
  **Geophysics** 85(4), WA27–WA39.
- Di, X. et al. (2026). *FaultEdgeFormer*. **Journal of Geophysics and Engineering** 23(4), 1285–1311
  (em `Articles/`) — a faixa de frequência do sintético deve cobrir a do real; rejeito 0–20 amostras,
  mergulho 45°–90°, ruído de 5 a 25 dB.
- Cunha, A., Pochet, A., Lopes, H. & Gattass, M. (2020). *Seismic fault detection in real data using
  transfer learning from a convolutional neural network pre-trained with synthetic seismic data*.
  **Computers & Geosciences** 135, 104344 — ajuste fino com poucas seções reais (não usado aqui; ver o relatório).
- Rizzo, M. L. & Székely, G. J. (2016). *Energy distance*. **WIREs Computational Statistics** 8(1), 27–38.
- Hansen, N. & Ostermeier, A. (2001). *Completely derandomized self-adaptation in evolution strategies*.
  **Evolutionary Computation** 9(2), 159–195.
