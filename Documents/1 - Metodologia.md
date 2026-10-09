# Metodologia geral — segmentação de falhas geológicas em sísmica 3D

> Documento de referência do repositório `Falhas`, escrito em 07/10/2026 a partir do código e dos registros do projeto.
> Cobre o pipeline inteiro, do gerador sintético à comparação com o especialista no bloco de Marlim. O detalhamento
> matemático do treino (arquitetura, logits, funções de perda, retropropagação, otimizador, agendas) está em
> [`2 - Treinamento.md`](2%20-%20Treinamento.md).

## Sumário

1. [Objetivo e ideia central](#1-objetivo-e-ideia-central)
2. [Visão geral do pipeline](#2-visão-geral-do-pipeline)
3. [Ambiente, execução e convenções](#3-ambiente-execução-e-convenções)
4. [Dado sintético: o `SyntheticGenerator`](#4-dado-sintético-o-syntheticgenerator)
5. [Os datasets](#5-os-datasets)
6. [Formatação: o `Format.ipynb` e o `DataBase.csv`](#6-formatação-o-formatipynb-e-o-databasecsv)
7. [Configuração da rodada e automação (`Task/`)](#7-configuração-da-rodada-e-automação-task)
8. [Treinamento (visão geral)](#8-treinamento-visão-geral)
9. [Avaliação no sintético](#9-avaliação-no-sintético)
10. [Dado real: o bloco de Marlim](#10-dado-real-o-bloco-de-marlim)
11. [Inferência no Marlim](#11-inferência-no-marlim)
12. [Análise no Marlim: sticks e comparação com o especialista](#12-análise-no-marlim-sticks-e-comparação-com-o-especialista)
13. [Calibração do gerador pelo bloco real](#13-calibração-do-gerador-pelo-bloco-real)
14. [O framework de otimização `Nature`](#14-o-framework-de-otimização-nature)
15. [Histórico de decisões e evidências medidas](#15-histórico-de-decisões-e-evidências-medidas)
16. [Reprodutibilidade](#16-reprodutibilidade)
17. [Mapa de arquivos](#17-mapa-de-arquivos)
18. [Referências](#18-referências)

---

## 1. Objetivo e ideia central

- **Objetivo:** mapear automaticamente falhas geológicas em volumes sísmicos 3D, produzindo para cada voxel a
  probabilidade de ele pertencer a um plano de falha (segmentação binária voxel a voxel).
- **O problema:** o dado real de interesse — o bloco de **Marlim** (Bacia de Campos) — não tem rótulo voxel a voxel. O
  que existe é a interpretação 2D de um especialista em uma inline de cada patch, e essa interpretação é **parcial**
  (ele marca só as falhas principais).
- **A estratégia é *sim-to-real*:**
  1. gerar volumes sísmicos sintéticos com uma física simplificada (refletividade em camadas, dobramento,
     cisalhamento, falhamento, convolução com wavelet, ruído), em que o rótulo é exato por construção;
  2. treinar redes neurais 3D de segmentação nesses volumes;
  3. medir o ajuste no próprio sintético (IoU em volumes de teste nunca vistos);
  4. aplicar as redes ao bloco real e comparar com o especialista por uma métrica geométrica de **sticks** (segmentos
     de reta que representam o traço de cada falha na seção).
- **Duas réguas, que não andam juntas:** o IoU sintético mede quão bem a rede aprendeu o domínio de treino; recall e
  detecção de sticks no Marlim medem a transferência. Boa parte do trabalho do projeto foi descobrir que melhorar a
  primeira não garante a segunda (seção 15) e construir as ferramentas para medir a segunda.

---

## 2. Visão geral do pipeline

```
 ┌─────────────────────────┐   tiles crus 128³ em (x, z, y)
 │ Synthetic/index.py      │ ─────────────────────────────────► Dataset/<nome>/original/{images,masks}/
 │ SyntheticGenerator      │                                     (ou dado externo: FaultSeg3D, Zenodo)
 └─────────────────────────┘                                                  │
                                                                              ▼
                                                     Dataset/<nome>/Format.ipynb
                                                     normaliza cada volume → images/, masks/, DataBase.csv
                                                     (e reescreve Task/info.json com o nome do dataset)
                                                                              │
 Task/task.json ──► Task/index.py (papermill, uma rodada por item, n_trials)  │
                                                                              ▼
                                                     Model/1 - Model.ipynb (treino)
                                                     → Model/Backup/model_N/{info.json, model.pth, train.png, predictions/}
                                                                              │
                    ┌─────────────────────────────┬───────────────────────────┴──────────────┐
                    ▼                             ▼                                          ▼
         Model/2 - Compare.ipynb       Model/3 - Predict.ipynb                 Marlim/1 - Predict.ipynb
         tabela de todos os backups    reavalia no teste do modelo             janela deslizante no bloco real
                                       (tile inteiro e protocolo do artigo)    → model_N/marlim/patch_<id>/masks/*.dat
                                                                                                 │
                                                                                                 ▼
                                                                              Marlim/2 - Analysis.ipynb
                                                                              remonta o slab, extrai sticks, compara
                                                                              com o especialista → figuras + CSV

 Ramo de calibração do gerador (pesquisa):
   Marlim/files/patches/<id>/tiles/*.dat + anotação ─► Marlim/Regions/Marlim/Analysis.ipynb ─► tiles reais calm / faulted / dead
                                                                                                    │
                                             Marlim/Regions/Synthetic/Analysis.ipynb ◄──────────────┘
                                   (ImageSimilarity + Calibration + CMA-ES do Nature)
                                                         │
                                   SyntheticGenerator.dataset() ─► Dataset/dataset_regions/original/faulted
```

**Princípios que valem para o pipeline inteiro:**

- Os estágios trocam **arquivos**, nunca variáveis: cada notebook lê o que o anterior gravou. Isso permite rodar
  qualquer estágio isolado e repetir só o que mudou.
- Todo caminho dentro de um notebook é **relativo à pasta dele**; rodar com o diretório de trabalho errado quebra tudo.
- Uma rodada é descrita inteira por um `Task/info.json`; o `info.json` salvo no backup guarda essa configuração
  exatamente como veio (`processing`), então copiar e colar reproduz a rodada.
- O backup nunca é sobrescrito: cada treino cria `Model/Backup/model_N` com `N` = maior existente + 1.

---

## 3. Ambiente, execução e convenções

### 3.1 Ambiente

| item | valor |
|---|---|
| Ambiente conda | `torch-gpu` (o `base` não tem torch) |
| Bibliotecas | torch 2.7.1+cu118, MONAI 1.5.2, torchmetrics, albumentations, papermill, OpenCV, scikit-image, SciPy, psutil |
| GPU | NVIDIA Quadro P6000, 24 GB (arquitetura Pascal, sem tensor cores; FP64 a 1/32 do FP32) |
| Kernel dos notebooks | `python3`, que resolve para o do `torch-gpu` quando o processo roda de dentro dele |

- Não há build, lint nem suíte de testes: a verificação é rodar o notebook de cima para baixo e comparar os números.
- Uma rodada de treino custa horas; não se re-executa treino só para "verificar".
- O `.gitignore` exclui `*.npy`, `*.pth`, `*.dat`, imagens e `*.zip`: datasets, pesos e figuras são locais. Um clone
  limpo precisa regerar os dados (gerador + `Format`).

### 3.2 Convenções de código

- **Módulo é pasta com `index.py`** exportando o tipo principal (`from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Nature.index import NatureSelector`). Importar módulo nunca executa
  trabalho. Notebook fora da pasta ajusta o `sys.path`.
- **Nova rede:** arquivo em `Model/Network/types/X.py`, import e um `if` em `ModelNetwork.get()` com uma linha
  MAIÚSCULA citando a origem; o nome usado ali é o que vai no `info.json`.
- **Nova perda:** classe em `Model/Losses/index.py` e entrada em `Losses.options`; toda perda força `float32` fora do
  autocast.
- **Aumentação:** só pela chave `augmentations` do `info.json` (`Model/Transforms/index.py`).
- **Idioma:** identificadores em inglês e `camelCase`; comentários, markdown, títulos e relatórios em português.
- **Notebooks:** uma etapa por célula, terminando numa prova visível (print, tabela ou figura); a explicação fica no
  markdown da seção, em tópicos.
- **Pool por `fork` depois de OpenCV:** notebook que usa cv2 no processo principal e depois cria pool por `fork`
  chama `cv2.setNumThreads(1)` na primeira célula; sem isso os filhos travam em `futex` e o `pool.map` espera para
  sempre (medido em 19/09/2026).

### 3.3 Convenção de eixos

| dado | ordem dos eixos | observação |
|---|---|---|
| Gerador (interno) | `(x, y, z)` | z é a vertical (tempo/profundidade) |
| Datasets gravados | **`(x, z, y)`** | o `saveTile` transpõe `(0, 2, 1)`; `Synthetic/utils.formatAxis` faz o mesmo para visualizar |
| FaultSeg3D / Zenodo | `(inline, xline, tempo)` | o `Format` transpõe `(0, 2, 1)` para `(x, z, y)` |
| Tiles reais (regiões e `.dat` do Marlim) | `(inline, z, xline)` | mesma convenção `(x, z, y)` |

- Nas redes, `(D, H, W)` do PyTorch = `(x, z, y)`: o primeiro eixo espacial é a inline, o segundo é o tempo.
- Nas aumentações, giros e flips ficam no plano horizontal `[0, 2]`, para nunca virar o eixo do tempo.

---

## 4. Dado sintético: o `SyntheticGenerator`

`Synthetic/index.py` define a classe `SyntheticGenerator`, que monta um volume sísmico sintético e a máscara binária
das falhas. Cada atributo que é uma faixa `(low, high)` é sorteado de novo **a cada tile**, o que dá variedade ao
dataset. Os valores padrão da classe são a configuração do `dataset_74` (a que melhor transferiu para o Marlim).

### 4.1 Volume de trabalho

- O volume é montado num cubo com **margem de 64 voxels** em cada face: para a saída 128³, o cubo interno é
  $n_x = n_y = n_z = 128 + 2\cdot 64 = 256$.
- A margem absorve as bordas do dobramento e do rejeito das falhas (o deslocamento traz material "de fora"); no fim ela
  é cortada (`crop`).
- Toda a conta roda em **torch, float64, na GPU** (cai para CPU sem CUDA). A implementação reproduz as operações do
  SciPy: um filtro 1D ao longo de um eixo vira o produto por uma matriz obtida aplicando o filtro do SciPy à identidade
  (mesma borda), a interpolação cúbica é o spline 1D do `map_coordinates` no modo `nearest` e a trilinear é o
  `grid_sample` com `padding_mode='border'`.

### 4.2 Etapas

`get()` encadeia: refletividade → dobra → cisalhamento → falhamento → wavelet → ruído → corte → z-score.

**1. Refletividade 1D (`genReflectivity`).** Uma coluna $r(z)$ de comprimento $n_z$, inicialmente zero. Para
$N_c \sim \mathcal{U}\{26,\dots,232\}$ camadas (`layerRange`): sorteia a posição $z_0 \sim \mathcal{U}\{0,\dots,n_z-1\}$,
a espessura $e \sim \mathcal{U}\{1,2,3\}$ (`layerThickness`) e escreve $r(z_0:z_0+e) = a$, $a\sim\mathcal{U}(-1,1)$.

**2. Dobramento (`applyFolding`).** Um campo de deslocamento vertical $s(x,y)$ soma $N_f \sim \mathcal{U}\{15..47\}$
gaussianas 2D anisotrópicas e giradas (`foldCount`):

$$
s(x,y)=\sum_{k=1}^{N_f} A_k \exp\!\left(-\frac{u_k^2}{2\sigma_{x,k}^2}-\frac{v_k^2}{2\sigma_{y,k}^2}\right),\qquad
\begin{bmatrix}u_k\\v_k\end{bmatrix}=R(\theta_k)\begin{bmatrix}\alpha(x-x_k)\\ y-y_k\end{bmatrix}
$$

com centros $x_k, y_k$ sorteados em $[-0{,}3n, 1{,}3n]$, larguras $\sigma \in [17,57]$ (`foldSigma`), amplitude
$A_k \in [-35, 5]$ (`foldAmplitude`), ângulo $\theta_k \in [0,\pi)$ e alongamento $\alpha$ (`foldAspect`, 1). O
deslocamento cresce linearmente com a profundidade (`foldDamping` $\beta = 1{,}35$) e soma um deslocamento base
$a_0 \in [-0{,}75; 4{,}45]$ (`foldBaseShift`):

$$z' = z + a_0 + s(x,y)\,\beta\,\frac{z}{n_z-1},\qquad V(x,y,z) = r(z') \;\text{(spline cúbico em } z\text{)}.$$

**3. Cisalhamento (`applyShearing`).** Um mergulho regional planar: $z'' = z + e_0 + f\,x + g\,y$, com
$e_0 \in [-8{,}68; 3{,}3]$ (`shearOffset`) e $f, g \in [-0{,}1; 0{,}02]$ (`shearGradient`); o volume é reamostrado em
$z''$ pelo mesmo spline cúbico 1D.

**4. Falhamento (`applyFaulting`).** $N \sim \mathcal{U}\{5,\dots,9\}$ falhas (`faultCount = (5, 10)`, teto
exclusivo), aplicadas em sequência; cada uma (`applyFault`):

- **Geometria do plano:** ponto $\mathbf{p}_0$ uniforme em $[0{,}15; 0{,}85]$ do cubo, mergulho
  $\delta \in [55°, 81°]$ (`faultDipAngle`), azimute $\varphi \in [0, 2\pi)$. A normal é
  $\mathbf{n} = (\sin\delta\cos\varphi,\ \sin\delta\sin\varphi,\ \pm\cos\delta)$; o vetor de direção (*strike*)
  $\mathbf{s}$ é horizontal e perpendicular a $\mathbf{n}$; o vetor de mergulho é $\mathbf{d} = \mathbf{n}\times\mathbf{s}$.
- **Distâncias:** para cada voxel $\mathbf{q}$, $d_n = \mathbf{n}\cdot(\mathbf{q}-\mathbf{p}_0)$ (distância ao plano),
  $d_d$ e $d_s$ (ao longo do mergulho e da direção).
- **Rugosidade:** $d_n \leftarrow d_n + 3{,}54\cdot G_{8{,}55} * \xi$, um campo normal padrão $\xi$ suavizado por
  gaussiana de $\sigma = 8{,}55$ (`faultRoughness`, `faultRoughSigma`).
- **Falha lístrica** (probabilidade 0,07, `faultCurveProb`): o plano se curva com o quadrado da distância ao longo do
  mergulho, $d_n \leftarrow d_n - c\,(d_d / (\max n / 1{,}5))^2$, $|c| \in [4{,}22; 8{,}44]$ (`faultCurveMax`).
- **Rejeito** $t(\mathbf{q})$, com máximo $T \in [15, 32]$ voxels (`faultThrow`), metade das vezes de cada forma:
  - gaussiano em torno do centro: $t = T\exp\!\big(-(d_s^2+d_d^2)/(2\sigma_t^2)\big)$, $\sigma_t \in [49, 59]$
    (`faultDecaySigma`);
  - rampa ao longo do mergulho: $t = T\cdot\mathrm{clip}\big(\pm d_d/\lVert n\rVert + 0{,}5,\ 0,\ 1\big)$.
- **Deslocamento:** o bloco alto ($d_n > 0$) é reamostrado em $\mathbf{q} + t(\mathbf{q})\,\mathbf{d}$ (trilinear,
  borda presa); o bloco baixo fica. As máscaras das falhas anteriores são deslocadas junto (vizinho mais próximo, zero
  fora do cubo), então uma falha nova corta e desloca as antigas.
- **Rótulo:** a falha nova acrescenta à máscara os voxels com $|d_n| \le 0{,}99$ (`faultZoneWidth`, meia-espessura) e
  $|t| > 0{,}77$ (`faultThreshold`, para não rotular onde o rejeito some). A espessura do rótulo fica em torno de 2
  voxels.

**5. Wavelet (`applyWavelet`).** Convolução em $z$ com uma Ricker de frequência $f \in [72, 99]$ (`waveletFreq`),
amostrada a $\Delta t = 0{,}0012$ (`waveletDt`) em $t \in [-0{,}1; 0{,}1)$ (`waveletDuration`):

$$w(t) = \big(1 - 2\pi^2 f^2 t^2\big)\,e^{-\pi^2 f^2 t^2}.$$

O pico do espectro em ciclos por amostra é $f\,\Delta t \approx 0{,}086$–$0{,}119$, ou seja, um **período vertical de
8,4 a 11,6 voxels**. As duas variáveis são redundantes: só o produto $f\Delta t$ importa.

**6. Ruído (`applyNoise`).** Um campo normal suavizado por gaussiana anisotrópica de $\sigma = (1, 1, 0{,}5)$ em
$(x, y, z)$ (`noiseSigma`), normalizado para desvio 1 e escalado por $\eta\cdot\mathrm{std}(V)$ com
$\eta \in [0{,}015; 0{,}618]$ (`noiseLevel`). Sinal + ruído passam por uma suavização lateral final de
$\sigma = (0{,}5; 0{,}5; 0)$.

**7. Corte e escala.** Corta a margem (256³ → 128³) e faz z-score do tile: $I = (V - \mu)/(\sigma + 10^{-8})$.
Opcionalmente (usado pela calibração), `gain`, `gainJitter` e `clip` reescalam depois do z-score:
$I \leftarrow \mathrm{clip}(I\cdot g\cdot j, \pm c)$, com $j$ log-normal de mediana 1 entre os tiles de um lote
(`getJitter`), para o ganho não depender do tamanho do lote.

### 4.3 Geração de um dataset (`dataset`)

```python
options = {'directory': '...', 'seed': 123, 'regions': {'faulted': {'n_images': 220, 'output': 'original/faulted', 'params': {...}}}}
SyntheticGenerator().dataset(options)
```

- Chave desconhecida em `options`, numa região ou nos `params` levanta erro; duas regiões não podem gravar na mesma
  pasta. A pasta de cada região é apagada e recriada.
- Cada tile $i$ usa a semente própria `seed + offset + i`: um tile qualquer é reproduzível isolado.
- Saída: `<directory>/<output>/{images,masks}/img_XXXX.npy`, já transposta para `(x, z, y)`; imagem `float32`
  (z-scorada, ou com ganho e saturação), máscara `uint8`. O índice é global entre regiões.
- `fast = True` sorteia os dois campos de ruído (rugosidade e ruído da imagem) direto na placa em float32: mesma
  distribuição, ~6× mais rápido, mas a semente dá outro tile. A busca da calibração usa `fast`; o dataset final sai
  no padrão, que repete tile a tile o gerador histórico.
- Custo medido: 2–7 s por tile na GPU (contra 15–70 s da versão em CPU).

### 4.4 Limites conhecidos do gerador (medidos)

- **Coerência:** como a refletividade é 1D convoluída em $z$, o campo é sempre localmente planar; dobramento não
  baixa a coerência abaixo de ~0,90. A alavanca é o ruído (`noiseSigma` + `noiseLevel`).
- **Mergulho aparente:** com azimute uniforme, o mergulho visto numa seção é $\arctan(\tan\delta\,|\sin\varphi|)$;
  ~13% das falhas cruzam a seção quase deitadas, abaixo do que o especialista anota (~40°).
- **Escala:** o período vertical sintético (8–11 px) é bem menor que o das falhas do Marlim (17–25 px na faixa onde
  está a maior parte do comprimento anotado) — ver seção 15.

---

## 5. Os datasets

Todos são 220 volumes 128³ (exceto o misto), com ~7% dos voxels rotulados como falha.

| dataset | origem | conteúdo | observações |
|---|---|---|---|
| `dataset_74` | `SyntheticGenerator` com `Dataset/dataset_74/synthetic.json` | 220 tiles, 5–9 falhas por tile | a configuração que vira o padrão da classe; o melhor dado conhecido para transferir ao Marlim |
| `dataset_wu` | FaultSeg3D (Wu et al., 2019) | 220 `.dat` float32 128³ (`original/`), já padronizados | o *benchmark* público; ~7% de falha, ~2–3 planos por volume, mergulho mediano ~76° |
| `dataset_zu` | `200-20.zip` dos autores da ResACEUnet (Zenodo 20339874) | 220 `.npy` | **é o FaultSeg3D bit a bit**, só renumerado (o `Compare.ipynb` e o `README.md` da pasta provam e trazem o mapa); hoje formatado com `scaling: 'standardize'` |
| `dataset_74_wu` | misto | 110 primeiros do `dataset_74` + 220 do `dataset_wu` = 330 | aponta para os arquivos das fontes; roda o `Format` da fonte que estiver sem `DataBase.csv` ou em outro `scaling` |
| `dataset_regions` | calibração por região (`Marlim/Regions/Synthetic`) | 220 tiles `faulted`, crus e saturados em ±2,42 | `synthetic.json` guarda as opções e a semente; hoje sem `DataBase.csv` |
| `marlim_opt` | `MarlimSyntheticGenerator` (`Marlim/files/Generator.ipynb`) | 220 tiles | gerador próprio, copiado do bloco real (coluna virtual com perfis por profundidade); legado, formatado com `scaling: 'percentile'` (o ganho global do gerador pede o percentil do conjunto) |

Estatísticas dos `DataBase.csv` atuais (min-max por volume): média ~0,50 em todos, desvio médio 0,126 (`dataset_74`),
0,100 (`dataset_wu`), 0,108 (`dataset_74_wu`); `dataset_zu` padronizado (média 0, desvio 1).

---

## 6. Formatação: o `Format.ipynb` e o `DataBase.csv`

Cada dataset tem o seu `Dataset/<nome>/Format.ipynb`, que roda com o diretório de trabalho na própria pasta.

1. Lê o `Task/info.json`, troca o `dataset` pelo nome da pasta e **regrava o `Task/info.json`** (é assim que o treino
   seguinte sabe qual dataset usar).
2. Lê `original/images` e `original/masks` (ou `original/*/images` quando há regiões), corrigindo os eixos quando o
   dado vem de fora (`formatAxis`, transposição `(0, 2, 1)`). O `dataset_zu` baixa o zip do Zenodo se faltar volume e
   confere o md5.
3. **Escalona cada volume** pela chave `scaling` do `info.json`, com a `Normalization` do `Dataset/index.py`:
   - `'normalize'` (padrão sem a chave, decisão de 06/10/2026): **min-max do próprio volume** para $[0,1]$,
     $I' = (I - \min I)/(\max I - \min I)$;
   - `'percentile'`: corte no p01/p99 **do conjunto inteiro**, reescalado para $[0,1]$,
     $I' = (\mathrm{clip}(I, p_{01}, p_{99}) - p_{01})/(p_{99} - p_{01})$ (o `marlim_opt`, cujo gerador tem ganho global);
   - `'standardize'`: **padronização** do volume, $I' = (I-\mu)/\sigma$ (o passo do artigo da ResACEUnet);
   - `null`: o volume cru.
4. Grava `images/` e `masks/` e o **`DataBase.csv`**, uma linha por volume: `id`, `img_min`, `img_max`, `img_mean`,
   `img_std`, `msk_min`, `msk_max`, `shape`, `img_path`, `mask_path` (absolutos) e `scaling`.
5. Com `img_size: null` (o caso normal) termina em `sys.exit` — no papermill isso aparece como erro, mas tudo já foi
   gravado. Com `img_size` preenchido, o `TilesBuilder` corta os volumes em blocos sem sobreposição
   (`tiles/{images,masks}`, imagem com borda por reflexão, máscara com zero) e o `DataBase.csv` aponta para os blocos.

**Trava de normalização:** o `1 - Model`, o `0 - Model_CV` e o `3 - Predict` param com erro se a coluna `scaling`
do `DataBase.csv` não bater com a da rodada (sem a coluna, vale min-max); o `Task/index.py` refaz o `Format` quando ela
difere. O `dataset_74_wu` não tem `images/` próprios (junta os do `dataset_74` e do `dataset_wu`) e roda, por
papermill, o `Format` da fonte que não estiver no `scaling` da rodada; no percentil, cada fonte fica com o p01/p99 dela.
O `getFiles`, o `setFolder`, o `showTile` e o `TilesBuilder` também vêm do `Dataset/index.py`.

> **Armadilha medida:** com `img_size` preenchido, o split do treino é **por bloco**, não por volume — blocos do mesmo
> volume caem em treino e teste. Os modelos de 64³ e 32³ que pareciam bater o 128³ no IoU sintético tinham esse
> vazamento (e foram os piores no Marlim). Por isso o caminho normal é `img_size: null` e, para treinar em janela
> menor, o recorte dinâmico do `Transforms` (seção 8.4).

---

## 7. Configuração da rodada e automação (`Task/`)

### 7.1 `Task/info.json`

É a configuração única de uma rodada, lida como `OPTIONS` no treino:

| chave | tipo (padrão) | efeito |
|---|---|---|
| `network` | texto | nome no `ModelNetwork.get()`: `unet_3d`, `dbrnet`, `segresnet`, `resaceunet_grva`, `resaceunet_wu`, `resaceunet_zu`, `macnn`, `fault_seg_net`, `nru_net`, `fault_edge_former` |
| `dataset` | texto | pasta em `Dataset/` |
| `img_size` | `null` ou lista | `null` = volume inteiro; lista = o `Format` corta em blocos (ver armadilha acima) |
| `scaling` | texto (`'normalize'`) | `'normalize'` (min-max por volume), `'percentile'` (p01/p99 do conjunto), `'standardize'` (média 0, desvio 1 por volume) ou `null` (cru) |
| `lr` | float | taxa de aprendizado do AdamW (o pico, na agenda `cosine`) |
| `loss` | texto | `cross_entropy`, `dice_focal`, `dice_ce`, `focal`, `smooth_dice`, `compound`, `tversky` |
| `batch_size` | int | volumes (ou recortes) por passo |
| `scheduler` | `plateau` ou `cosine` | agenda do lr (outro valor quebra o `Trainer`) |
| `dropout` | float | taxa base de dropout (cada rede distribui pelos níveis) |
| `num_filters` | int | largura base da rede |
| `ema` | bool | média móvel exponencial dos pesos na avaliação |
| `amp` | bool (`false`) | precisão mista (float16 no autocast + `GradScaler`) |
| `epochs` | int (100) | teto de épocas |
| `patience` | int (15) | paciência do early stopping sobre `val_iou` |
| `n_trials` | int (1) | repetições da mesma rodada com sementes 42, 43, ... |
| `augmentations` | dict ou `null` | receita de aumentação (seção 8.4) |
| `trial` | int | gravado pelo `Task/index.py`; a semente é `42 + trial` |

### 7.2 `Task/task.json` e `Task/index.py`

- `task.json` é uma **lista de rodadas**, cada uma no formato do `info.json`.
- `cd Task && python index.py` percorre a lista. Para cada rodada:
  1. grava a rodada no `Task/info.json`;
  2. executa `Dataset/<dataset>/Format.ipynb` — **pulado** quando o `DataBase.csv` já existe, `img_size` é `null` e a
     coluna `scaling` dele bate com a da rodada;
  3. executa `Model/1 - Model.ipynb` uma vez por trial (`n_trials`), gravando `trial` no `info.json` antes de cada
     execução.
- A execução é por **papermill** (kernel `python3`, `cwd` = pasta do notebook); a saída executada vai para
  `Task/logs/<nome>_out.ipynb`. Uma exceção num notebook é impressa e a campanha segue para a próxima rodada.
- O progresso da época corrente fica em `Model/progress.json` (a saída das células não chega ao terminal do
  `index.py`).

### 7.3 Campanha atual (`task.json` de 07/10/2026)

1. `resaceunet_zu` (nf 16) no `dataset_zu` padronizado, `dice_ce`, lote 8, `cosine` com pico 1e-3, AMP, 200 épocas,
   paciência 20, receita completa de aumentação dos autores com recorte 96³ — a reprodução do artigo da ResACEUnet.
2. `dbrnet` (nf 32) e 3. `resaceunet_zu` (nf 16) no `dataset_74_wu`, 128³, min-max, sem aumentação, mesmo treino
   (`dice_focal`, lr 1e-3, `plateau`, lote 2, dropout 0,1, 100 épocas).
4–7. `dbrnet` no `dataset_wu` com quatro perdas (`dice_focal`, `cross_entropy`, `smooth_dice`, `focal`), lr 1e-4,
   EMA, 3 trials cada — para média ± desvio por perda.

---

## 8. Treinamento (visão geral)

O detalhamento matemático está em `2 - Treinamento.md`. Em resumo, o `Model/1 - Model.ipynb`:

### 8.1 Semente e dados

- `seed_everything(42 + trial)`: semente de Python, NumPy, torch e CUDA; cuDNN determinístico e sem *benchmark*
  (exceção: a `macnn`, cuja convolução dilatada não tem algoritmo determinístico que caiba na memória).
- Lê o `DataBase.csv` do dataset (com a trava de normalização) e o `shape` dos volumes.
- **Split fixo** com `random_state = 42`, independente do trial: ~4,5% para teste e ~4,5% para validação — 220
  volumes dão **200 / 10 / 10** (330 dão 300 / 15 / 15). O split vai para o `division` do `info.json` salvo.

### 8.2 Rede, perda, otimizador e agenda

- `ModelNetwork(network, img_size, classes=1, channels=1, lr, dropout, num_filters)` monta a rede na GPU, o otimizador
  **AdamW** (`weight_decay = 1e-4`) e a métrica `BinaryJaccardIndex` (limiar 0,5).
- A rede devolve **logits** (sem sigmoide); a sigmoide vive na perda e na inferência.
- `Losses(loss)` escolhe a perda; toda perda é calculada em float32 fora do autocast.
- Agendas: `plateau` (`ReduceLROnPlateau` no `val_loss`, fator 0,5, paciência 10, por época) ou `cosine` (aquecimento
  linear de 1e-6 ao `lr` em 10 épocas e cosseno até 1e-7, a cada passo — a agenda da ResACEUnet).

### 8.3 Laço de treino

- Por passo: forward (no autocast se `amp`), perda, `backward` pelo `GradScaler`, *clip* da norma global do gradiente
  em 1,0, passo do AdamW, passo da agenda `cosine`, atualização do EMA, acúmulo do IoU.
- Por época: validação (com os pesos do EMA, se ligado), passo do `plateau`, histórico em `progress.json` e **early
  stopping** sobre `val_iou` (`min_delta` 1e-4, paciência `patience`).
- No fim, o melhor estado volta para o modelo vivo, que é o que se testa e salva.

### 8.4 Aumentação (`Transforms`)

- Só pelo `augmentations` do `info.json`; `null`/ausente = tiles exatamente como o `Format` gravou.
- Reproduz o `build_data.py` da ResACEUnet: `contrast` (gamma), `rotate90`, `flip`, `rotate` (rotação livre),
  `smooth` (suavização gaussiana), `noise` (ruído gaussiano), `crop` (recorte centrado em falha ou fundo) — mais o
  `zoom` isotrópico do projeto (fator ≥ 1, para levar o período sintético à escala das falhas do Marlim).
- Sorteio por tile no `DataLoader` com semente `(42 + trial, época, índice)`: o mesmo treino repete a mesma aumentação
  com qualquer número de workers, sem tocar no gerador global.
- Aplicada só no treino, na ordem do JSON; `n_aug` (padrão 1) = quantas variações de cada tile entram por época.
- Com `crop`, a rede é montada na **janela do recorte** e valida/testa o tile inteiro por janela deslizante; o
  `model.img_size` salvo é a janela.

### 8.5 O que é salvo (`Model/Backup/model_N/`)

| arquivo | conteúdo |
|---|---|
| `info.json` | `trainer` (caminho, perda, agenda, épocas, lote, ema, trial, `val_iou` do melhor estado, `test_iou`), `processing` (o `Task/info.json` como veio), `division` (`val_size`, `test_size`, `n_images`), `model` (argumentos do `ModelNetwork`, com o `img_size` da janela), `iou` |
| `model.pth` | `{'model': state_dict, 'optimizer': state_dict, 'timestamp', 'history'}` |
| `train.png` | curvas de perda e IoU de treino e validação |
| `predictions/` | até 20 figuras de teste: acerto em verde, falso alarme em vermelho, falha perdida em azul |

### 8.6 Validação cruzada (`Model/0 - Model_CV.ipynb`)

- Separa o mesmo teste do `1 - Model` (mesmo `train_test_split` com `random_state = 42`) e divide o resto em 5 dobras
  (`KFold`, `shuffle`, `random_state = 42`); os tiles de teste nunca entram em dobra nenhuma.
- Cada dobra treina uma rede nova com a mesma semente; a de maior `val_iou` é a que se testa e salva. O `info.json`
  guarda `k_fold`, `fold`, `folds_iou` e o histórico de todas as dobras.

---

## 9. Avaliação no sintético

### 9.1 A métrica: IoU binário

$$\mathrm{IoU} = \frac{TP}{TP + FP + FN}$$

com a predição $\hat y = \mathbb{1}[\sigma(z) > 0{,}5]$ e as contagens **somadas sobre todos os voxels de todos os
volumes** do conjunto (micro-média), não a média por volume. O IoU aqui é severo: dilatar o rótulo perfeito em 1 voxel
já derruba o IoU para ~0,41; ele funciona como medida de precisão de posicionamento sub-voxel da superfície (seção 15).

### 9.2 `Model/2 - Compare.ipynb`

- Junta todos os `Backup/*/info.json` numa tabela (cada seção do JSON vira colunas).
- `VariationAnalysis(df, variations, metric, mode)`: para cada hiperparâmetro da lista, fixa os outros e compara as
  variações (mínimo, máximo, média ± desvio ou mediana entre trials), em tabela e barras. O `trial` nunca entra nas
  `variations`.

### 9.3 `Model/3 - Predict.ipynb`

- Reavalia um modelo salvo **no teste do treino dele**: lê `processing` e `division` do `info.json`, refaz o
  `train_test_split(test_size, random_state=42)` e para se as contagens não baterem com o `n_images` gravado.
- Opcionalmente avalia em outro dataset (`DATASET`), com o mesmo `test_size` e sem os volumes que o modelo viu.
- Soma TP, FP, FN, TN de todos os voxels e mostra acurácia, precisão, recall, IoU, F1 e a matriz de confusão.
- Modelo treinado com recorte: o tile inteiro vai por janela deslizante (a mesma conta do `test_iou`), e roda também o
  **protocolo da Tabela 2 da ResACEUnet**: 8 recortes 96³ por volume de teste, metade centrada em falha e metade no
  fundo, métricas por recorte e média. No código dos autores precisão e recall estão trocados
  (`compute_prec(outputs, targets)` conta FN no lugar de FP); IoU e F1 não mudam.

---

## 10. Dado real: o bloco de Marlim

### 10.1 Organização

- Bloco completo $3530 \times 1601 \times 2240$ em `(inline, z, xline)`.
- Quatro **patches** (`Marlim/files/patches/{1200,1300,1400,2600}/tiles`, ao lado da interpretação), cada um um slab de inlines em volta da inline
  de mesmo número, cortado em **910 tiles** `.dat` (float32 cru, 128³) com 32 voxels de sobreposição de cada lado
  (passo 64, núcleo central de 64³ por tile): grade `1 × 26 × 35`. O `patch_metadata.json` descreve forma original (`64 × 1601 × 2240`, o núcleo do slab), passo,
  sobreposição e normalização.
- Os `.dat` foram normalizados para $[0,1]$ pelo **p01/p99 do próprio bloco** (`vmin = −1,365`, `vmax = 1,351`); a
  água é uma constante (~0,503). Só os modelos com `scaling` `'normalize'` ou `'percentile'` (dado em $[0,1]$) se
  aplicam ao Marlim; os outros são pulados.
- Os 128 inlines de cada `.dat` são dados reais (vizinhos correlacionam ≥ 0,99); só as bordas em z e xline têm
  preenchimento por reflexão.

### 10.2 A interpretação do especialista

- `Marlim/files/patches/<id>/<id>_interpretado.png` (2240 × 1601) marca as falhas na **inline anotada**, que alinha
  1:1 pixel a voxel com uma fatia do slab — mas **não com a fatia central**. A fatia é achada por correlação do slab com
  `<id>.png` e guardada em `Marlim/files/cache/inline_index.json`: 1200 → 16, 1300 → 52, 1400 → 24, 2600 → 8.
- A anotação é **parcial**: só as falhas principais. Por isso a precisão contra ela é limite inferior, e recall e
  detecção são as métricas que importam.

### 10.3 O que se sabe do bloco (medido)

- **Coluna vertical:** água até z ≈ 250 (fundo do mar entre z 256 e 420, mergulhando para leste); pacote raso forte
  250–600 (desvio ~0,17); intervalo fraco 600–800 (~0,10); banda brilhante dobrada 800–1200 (0,15–0,20); zona morta
  abaixo de ~1200 (0,02–0,04, com *crosshatch* de migração a ±45°).
- **Frequência dominante cai com a profundidade:** período vertical de ~4 px no raso, 7–9 px em z 520–600, 13–16 px em
  z 650–720 e **17–25 px de z 776 a 1400**, onde está a maior parte do comprimento anotado. A wavelet real é
  não-estacionária.
- **Falhas do especialista** (88 traços nos 4 patches): mergulho aparente p5/p50/p95 = 48/66/78°; comprimento
  p25/p50/p95 = 250/348/747 px (83% atravessam um tile inteiro); famílias paralelas em dominó (espaçamento ~20–60 px),
  ~80% mergulhando para a direita; faixa z ~250–1500.

---

## 11. Inferência no Marlim (`Marlim/1 - Predict.ipynb`)

- Configuração no topo: `BASE_PATH` (pasta dos modelos: `../Model/Backup`, `../Documents/Backups/...`), `PATCH_IDS`,
  `MODELS` (vazio = todos), `OVERLAP` (0,25), `WINDOW` (`None` = a janela do treino) e `SKIP_DONE`.
- `MarlimPredictor` carrega cada modelo pelo `info.json` (`ModelNetwork(**model)` + `model.pth`), pula os de
  `scaling` fora de `'normalize'`/`'percentile'` e os patches já preditos.
- **`SlidingWindow`** prediz cada tile 128³ **na janela em que a rede foi treinada**:
  - posições a cada `round(janela·(1 − OVERLAP))` voxels, com a última encostada no fim do tile; janela maior que o
    tile → o tile é estendido por reflexão;
  - peso 3D separável de Hanning, $w(\mathbf{p}) = \max\big(h_D(p_1)\,h_H(p_2)\,h_W(p_3),\ 10^{-3}\big)$, ~1 no
    centro e ~0 na borda da janela, para a emenda não virar degrau;
  - probabilidade final $P = \sum_k w\,\sigma(z_k) \,/\, \sum_k w$ (média ponderada das **probabilidades**);
  - lote de janelas pelo orçamento de $128^3$ voxels por passada na GPU (máximo 256 janelas).
  - Janela igual ao tile → uma passada só (o `OVERLAP` não tem efeito).
- Saída: um `.dat` float32 de probabilidade por tile de entrada em
  `Model/Backup/model_N/marlim/patch_<id>/masks/`, com o mesmo nome, e um `predict.json` (janela, sobreposição,
  janelas por tile).

---

## 12. Análise no Marlim: sticks e comparação com o especialista

`Marlim/2 - Analysis.ipynb`.

### 12.1 Reconstrução do slab

`FaultComparer.reconstruct` monta o slab `64 × 1601 × 2240` a partir dos 910 tiles: de cada tile 128³ fica só o
**núcleo central 64³** (descarta 32 voxels de cada borda, onde a predição tem menos contexto), colado na grade de passo
64; o último tile de cada eixo encosta no fim. Um patch com tile faltando é recusado (sairia zerado). O volume predito
vira `predicted.npy` na pasta do patch.

### 12.2 `FaultStickExtractor`: da probabilidade aos sticks

Cada falha numa seção 2D vira uma reta (stick), a mesma representação para a rede e para o especialista. Os botões
do topo do notebook: `STICK_TOL = 0,80` (permissividade, 0 = só o traço óbvio, 1 = tudo que parece falha),
`STICK_MIN = 140 px` (comprimento mínimo) e `STICK_GAP = 59 px` (maior lacuna costurada dentro de um stick).

1. **Evidência:** a probabilidade suavizada ($\sigma = 1$) e reescalada para $[0,1]$ entre um piso e um teto. Os dois
   sobem pelo percentil quando a predição é densa (orçamento de 2% da seção), para uma predição "teia" não explodir o
   traçado.
2. **Mapas da sísmica:** energia local (desvio gaussiano $\sigma = 25$, normalizado pelo p95: ~0 na água e na zona
   morta, ~1 nos refletores) e ângulo e coerência das camadas pelo tensor de estrutura (Sobel, $\sigma = 12$).
3. **Votação tipo Hough local:** para cada orientação de 34° a 146° em passos de 3°, a evidência é girada
   (`warpAffine`) e somada numa caixa de 5 × 140 px; os máximos locais acima do limiar são as sementes (até 1200).
4. **Supressão de sementes redundantes** por distância **perpendicular** (≤ 9 px) e ao longo da reta (≤ 51 px), com
   ângulo ≤ 9,9° — euclidiana apagaria as famílias em dominó.
5. **Traçado:** a partir da semente, a reta se estende enquanto houver suporte (evidência ≥ limiar no corredor de 5
   px), tolerando lacunas de até `STICK_GAP`; é isso que costura os pedaços.
6. **Refinamento por PCA:** a reta é reajustada ao centro de massa e à direção principal da evidência dentro do
   corredor (rejeita giros bruscos).
7. **Aceitação:** mergulho entre 43° e 76° (p5–p95 do especialista), comprimento ≥ `STICK_MIN`, preenchimento mínimo,
   **energia** (≥ 21% do stick com energia ≥ 0,167: a falha tem de cortar refletores, não a água nem a zona morta) e
   **ângulo com as camadas** acima do limiar (a falha corta as camadas; um stick paralelo a elas é refletor).
8. **Estiramento:** a reta aceita procura continuação além das pontas com lacuna maior (86 px) e corredor mais largo,
   exigindo preenchimento próprio no trecho novo.
9. **Matching pursuit:** o corredor da reta aceita é zerado na evidência residual, e a próxima semente não redesenha a
   mesma falha.
10. **Fusão e limpeza:** retas colineares (ângulo, deslocamento lateral e lacuna pequenos) se fundem pelo PCA das
    pontas; a reta que corre por cima de outra maior sai.

A anotação do especialista entra pelo `fromRaster`: esqueleto, corte nas junções e cada pedaço endireitado (quebrado no
ponto de maior desvio) até caber em 7 px.

### 12.3 Métricas

Para sticks preditos $M$ e do especialista $G$, um ponto de um stick está **coberto** se passa a menos de 12 px de um
stick do outro conjunto com direção a menos de 20°:

$$
\mathrm{recall} = \frac{\sum_{g\in G} \mathrm{cob}(g, M)\,\ell_g}{\sum_g \ell_g},\qquad
\mathrm{precisão} = \frac{\sum_{m\in M} \mathrm{cob}(m, G)\,\ell_m}{\sum_m \ell_m},\qquad
\mathrm{detecção} = \frac{|\{g : \mathrm{cob}(g, M) \ge 0{,}5\}|}{|G|}
$$

e o F1 é a média harmônica de recall e precisão. A tabela traz também o número de sticks, o comprimento mediano e o
comprimento total desenhado (contra o do especialista — sem olhar isso, qualquer mudança "melhora" o recall só
desenhando mais).

### 12.4 Saídas

- Por (modelo, patch): `predicted.npy` e as figuras `predicted_seismic.png`, `predicted_mask.png`,
  `predicted_sticks.png` na pasta do patch.
- `Marlim/files/comparisons/sticks_report_<base>.csv`: o ranking, com o par (modelo, patch) recalculado substituindo o
  antigo.

---

## 13. Calibração do gerador pelo bloco real

Ramo de pesquisa para aproximar o sintético do Marlim. Não faz parte do caminho normal de treino, mas produz datasets
(`dataset_regions`) e explica várias decisões.

### 13.1 Tiles reais por região (`Marlim/Regions/Marlim/Analysis.ipynb`)

- `MarlimBlock` monta o slab de cada patch, acha a água, o fundo do mar (primeira amostra que sai da constante da água),
  os traços mortos e a inline anotada.
- `RegionFinder` mede cada tile candidato 128³ (ocupando os 128 inlines) e separa por **conteúdo**, não por faixa fixa
  de z:

| região | regra | tiles |
|---|---|---|
| `calm` | inteiro no intervalo refletivo, nenhuma falha anotada a menos de 64 px, descontinuidade (semblance com mergulho) ≤ p5 dos tiles com falha | 149 |
| `faulted` | inteiro no intervalo refletivo e cruzado por ≥ 128 px de falha anotada | 188 |
| `dead` | inteiro na zona morta, sem voxel com envelope acima do Otsu, sem falha anotada a 64 px | 123 |

- Seleção gulosa com no máximo 25% de volume compartilhado; `check` relê e confere cada arquivo. Saída em
  `files/<região>/*.npy` e `files/DataBase.csv`.

### 13.2 A régua: `ImageSimilarity` (`Marlim/Regions/Synthetic/Analysis.ipynb`)

A similaridade entre um lote sintético e o `faulted` real é medida **pelos olhos de uma rede de falhas** — a
`Unet3D_V2` (dbrnet) de `Regions/Synthetic/model/`, treinada no `dataset_wu` — porque a régua anterior, de atributos
sísmicos, andava ao contrário do resultado no Marlim:

- **`rede`:** média e desvio de cada canal do gargalo da rede (hook), padronizados pelo `faulted` real e comparados pelo
  **KID** (MMD² não enviesado com núcleo polinomial cúbico); $\mathrm{sim} = 100\,e^{-\mathrm{KID}/\tau}$, com $\tau$ o
  KID entre `faulted` e `dead` reais (~23,7);
- **`falhas`:** a fração do tile que a rede marca como falha, comparada pelo **coeficiente de energia**
  $H = (2A - B - C)/(2A + \varepsilon)$ (Rizzo & Székely); $\mathrm{sim} = 100(1-H)$;
- **`periodo`:** o período vertical pelo centróide do espectro em z, pelo mesmo coeficiente — o eixo a que a rede quase
  não é sensível e o que mais separa o sintético do Marlim.
- A nota é a média geométrica ponderada dos grupos (rede 0,5, falhas 0,25, período 0,25). A mesma passada dá o IoU da
  rede nos rótulos sintéticos.

### 13.3 A busca: `Calibration`

- **Objetivo:** $0{,}9\cdot\mathrm{similaridade} + 0{,}1\cdot\mathrm{largura} - \mathrm{piso}$, em que a *largura* é a
  entropia média dos intervalos (log da largura entre 1% da faixa e a faixa inteira — numa direção que a régua não vê,
  o intervalo abre em vez de derivar) e o *piso* é a média, **tile a tile**, do que falta ao IoU da rede para chegar a
  40%.
- **Genoma:** cada intervalo do gerador vira centro + log da largura, normalizados em $[0,1]$; o centro anda numa caixa
  de ±25% da faixa em volta da `REFERENCE` (a configuração do `dataset_74`).
- **Otimizador:** CMA-ES (`NatureSelector('genetic')`), população 10, 40 gerações, primeira nuvem centrada na
  `REFERENCE`, 40 tiles por avaliação no modo `fast`. Os indivíduos de uma geração dividem as sementes e cada geração
  recebe sementes novas (o CMA-ES só usa a ordem dentro da geração).
- **Escolha final contra a maldição do otimizador:** `evaluate` refaz a decisão em 4 lotes de sementes novas entre a
  `REFERENCE`, o melhor indivíduo e a média da nuvem; a `REFERENCE` só perde se a vantagem passar do erro padrão da
  diferença pareada.
- **Geração:** `SyntheticGenerator.dataset()` grava 220 tiles do vencedor em
  `Dataset/dataset_regions/original/faulted`, crus e saturados em ±2,42, e o `synthetic.json` com as opções e a
  semente.

### 13.4 Gerador próprio de Marlim (`Marlim/files/Generator.ipynb`)

`MarlimSyntheticGenerator`: coluna virtual z 0–1600 em que cada tile sorteia um topo e os perfis de frequência,
amplitude, dobra e *crosshatch* seguem a profundidade (`freqKnots`, `ampKnots`, `hatchKnots`, `foldKnots`, calibrados
pelas medições da seção 10.3), campo de deslocamento único (dobra + mergulho + fundo do mar + falhas por cisalhamento
vertical), tiles de água e de zona morta como negativos duros e *gate* de visibilidade na máscara. Alimentou o
`marlim_opt`; perdeu para o `dataset_74` no Marlim.

---

## 14. O framework de otimização `Nature`

- `NatureSelector(nome, params, memory)` escolhe entre `genetic` (CMA-ES com reinício IPOP; aceita `mean` para a
  primeira nuvem nascer num ponto conhecido), `pso`, `de`, `lshade` e `lsrtde`, e delega `update()`, `portrait()` e
  `info()`.
- `Problem` traduz `{variável: {'type', 'bounds'}}` em genoma numérico (inteiros, contínuas, restrições); `nan`/`inf`
  entram como penalidade.
- `Memory` persiste `state.npz` (estado do otimizador), `best.json` e `history.json`: interromper e rodar de novo retoma
  a campanha; uma campanha terminada só é carregada.
- `backend='vector'` avalia a população inteira de uma vez (usado na calibração, em que o objetivo gera os tiles na GPU).
- `Portrait`/`Plotter` desenham a evolução das métricas e das variáveis.

---

## 15. Histórico de decisões e evidências medidas

Os itens abaixo são medições registradas ao longo do projeto que moldaram a metodologia. Valem como referência; os
números de backups antigos estão em `Documents/Backups`.

### 15.1 Sobre o IoU sintético

- **No `dataset_wu`, 92% de todo o erro é de borda** (medido em 03/09/2026 na `Unet3D_V2`, test IoU 0,778): 99,5% dos
  voxels de falha estão a ≤ 3 voxels de alguma predição; FN a ≤ 2 voxels da predição são 94% dos FN e FP a ≤ 2 voxels do
  rótulo são 91% dos FP. A detecção está resolvida; o jogo é localização sub-voxel.
- **IoU como precisão de posicionamento:** deslocando o rótulo por um campo suave de amplitude RMS $d$ e medindo o IoU
  contra o original, $d = 0{,}40$ voxel dá 0,778 — ou seja, o melhor modelo posiciona a superfície com ~0,4 voxel RMS.
- **Variância entre rodadas idênticas ≈ 0,03 de IoU**: diferença menor que isso não é resultado. Daí `n_trials` e a
  média ± desvio no `2 - Compare`.
- **O limiar 0,5 já é o ótimo** e a curva é chata (0,771 em 0,2 a 0,764 em 0,8): a probabilidade é bimodal.
- **Hiperparâmetros da `dbrnet` no `dataset_wu`** (`dice_focal`, 100 épocas): lr 5e-5 0,734 · 1e-4 0,753 · 5e-4 0,768 ·
  **1e-3 0,778** · 5e-3 0,774 · 1e-2 0,777; dropout chato de 0,01 a 0,3 (0,1 o melhor); `plateau` 0,778 × `cosine`
  0,771; perda `dice_focal` 0,778 > `cross_entropy` 0,758 > `focal` 0,695.

### 15.2 Sobre a transferência ao Marlim

- **O IoU sintético não prevê o Marlim.** Exemplos: os modelos de janela 64³/32³ tinham o melhor IoU (com vazamento) e
  quase o pior recall; o `dataset_regions` calibrado por atributos sísmicos teve similaridade 94 e F1 de sticks 0,07,
  contra 0,36 do `dataset_74`; com `smooth_dice` + lr 1e-3 o IoU sintético subiu e o F1 no Marlim caiu nas três redes
  com os dois pontos.
- **Janela pequena não transfere:** treinada em 64³/32³, a máscara no Marlim sai fragmentada (centenas de blobs de
  mediana 8 px) e o extrator descarta quase tudo.
- **Escala:** o sintético tem período vertical de 8–11 px e o Marlim, onde estão as falhas, 17–25 px. Reamostrando o
  Marlim antes da rede, cada faixa de z só é bem detectada quando o seu período casa com o do treino. Daí o `zoom` do
  `Transforms` (fator 1,5–3 leva o período a ~11–23 px, preservando o mergulho).
- **Normalização:** a troca de p01/p99 para min-max por volume custa só 0,005 de IoU a uma rede treinada com
  percentil; o Marlim continua no p01/p99 do bloco.
- **Melhor referência no Marlim até aqui:** `dbrnet` no `dataset_74` com `dice_focal`, lr 1e-3 e dropout 0,1 — médias
  nos 4 patches de recall 0,565, detecção 0,527 e F1 0,358. Teto medido de cobertura do traço anotado pela própria
  predição dilatada: 0,75–0,82; o extrator entrega ~65–70% desse teto.

### 15.3 Sobre a calibração do gerador

- A régua de 33 atributos sísmicos andava ao contrário do resultado (dava 96,9 ao dado de pior resultado). A régua pela
  rede de falhas ordena os dados como o Marlim.
- Quando o IoU da rede entrou como termo do objetivo, a busca colapsou nas falhas fáceis (5 falhas, rejeito 26–28,
  mergulho até 80°, período 8 px contra 16 no real) e o ganho veio todo do IoU. Daí o IoU virar **piso por tile** e a
  `largura` entrar no objetivo.

### 15.4 Sobre as redes

- **ResACEUnet do artigo × a nossa rodada:** a reprodução com pico de lr 1e-4 deu test 0,692 (val 0,687) contra 0,764
  do artigo; o diagnóstico foi **subajuste** (IoU no próprio treino 0,717, curva ainda subindo quando o cosseno chegou a
  1e-7) e não métrica nem generalização. Os configs do repositório dos autores usam 1e-3 ou 2e-3; a rodada está sendo
  refeita com pico de 1e-3 (rodada 1 do `task.json`).
- **As ResACEUnet dos autores dependem da posição absoluta:** na `resaceunet_wu` em 128³, 22,7 M dos 43,9 M parâmetros
  são embeddings de posição por voxel e projeções lineares sobre as posições (na `resaceunet_zu` em 96³, 9,6 M de
  59,4 M). Treinada sem aumentação, a `resaceunet_wu` memorizou os tiles: espelhar um tile de treino derrubou o IoU de
  0,74 para 0,52. Por isso essas redes só fazem sentido com o protocolo dos autores (recorte + aumentação).
- **Fault-Seg-Net:** o termo de perda do MSR (eq. 4 do artigo) é nocivo aqui (val IoU 0,17 com ele, 0,28 sem), por isso
  `ETA = 1`.
- **FaultEdgeFormer:** aprende muito devagar no protocolo do repositório; a Tversky engorda a máscara e piora o Marlim.
- **NRU:** melhor no sintético no mesmo orçamento, sem ganho no Marlim (o recall a mais vem de desenhar mais).
- **BatchNorm com lote 2** dá estatísticas móveis ruins; as redes do projeto usam GroupNorm.

---

## 16. Reprodutibilidade

- **Dado:** o `synthetic.json` de cada dataset sintético guarda as opções e a semente; cada tile é reproduzível isolado
  (semente `seed + i`). O gerador na GPU repete o gerador histórico com diferença ≤ 1,5e-8 na imagem e máscara
  idêntica.
- **Formatação:** determinística (min-max, percentil do conjunto ou padronização do volume); a coluna `scaling` trava
  o par dado/rodada.
- **Split:** fixo (`random_state = 42`) e gravado no `division`; o `3 - Predict` o refaz e confere.
- **Treino:** semente `42 + trial`, cuDNN determinístico, aumentação com semente própria por (trial, época, índice). Na
  prática os históricos por época se repetem até a terceira casa (algumas operações da CUDA com soma atômica no
  backward não são cobertas).
- **Rodada:** o `processing` do `info.json` salvo é o `Task/info.json` exato; colar no `task.json` refaz a rodada.
- **Do zero (clone limpo):** gerar os dados (gerador ou download), rodar o `Format` de cada dataset, preencher o
  `task.json` e rodar `cd Task && python index.py`; depois `Marlim/1 - Predict` e `Marlim/2 - Analysis`.

---

## 17. Mapa de arquivos

```
Falhas/
├── Synthetic/            index.py (SyntheticGenerator), utils/ (formatAxis, showTile), Analysis.ipynb (visualização)
├── Dataset/
│   ├── dataset_74/       Format.ipynb, synthetic.json, original/, images/, masks/, DataBase.csv
│   ├── dataset_wu/       FaultSeg3D (.dat em original/)
│   ├── dataset_zu/       Zenodo 200-20.zip; Compare.ipynb e README.md provam que é o FaultSeg3D
│   ├── dataset_74_wu/    misto, só Format.ipynb + DataBase.csv
│   ├── dataset_regions/  saída da calibração (original/faulted, synthetic.json)
│   ├── marlim_opt/       gerador próprio de Marlim (legado)
│   └── marlim/           patch_<id>/*.dat + patch_metadata.json (bloco real)
├── Task/                 index.py, task.json, info.json, logs/
├── Model/
│   ├── 0 - Model_CV.ipynb, 1 - Model.ipynb, 2 - Compare.ipynb, 3 - Predict.ipynb
│   ├── Network/          index.py (ModelNetwork) + types/ (uma rede por arquivo)
│   ├── Losses/ EMA/ EarlyStopping/ Transforms/ utils/
│   ├── Backup/           model_N/ (info.json, model.pth, train.png, predictions/, marlim/)
│   └── progress.json
├── Marlim/
│   ├── 1 - Predict.ipynb, 2 - Analysis.ipynb
│   ├── files/            Generator.ipynb, patches/<id>/ (anotações), cache/, comparisons/
│   └── Regions/          Marlim/Analysis.ipynb (tiles reais por região), Synthetic/Analysis.ipynb (calibração)
├── Nature/               index.py (NatureSelector), Models/ (CMA-ES, PSO, DE, L-SHADE, LSRTDE), Processing/
└── Documents/            este documento, 2 - Treinamento.md, Articles/ (PDFs), Backups/ (modelos antigos)
```

---

## 18. Referências

Artigos em `Documents/Articles/` e trabalhos citados no código:

- Wu, X. et al. (2019). *FaultSeg3D: using synthetic data sets to train an end-to-end convolutional neural network for
  3D seismic fault segmentation*. Geophysics — o `dataset_wu` e a receita de dado sintético.
- Zu, S. et al. (2024). *ResACEUnet: an improved transformer Unet model for 3D seismic fault detection* —
  `resaceunet_zu`/`resaceunet_wu`, `dice_ce`, a agenda `cosine` e a receita do `Transforms`.
- Gao, K. et al. (2022). *MACNN*, Geophysics 87(1), N13–N29 — `macnn` e `smooth_dice`.
- Gao, K., Huang, L. & Zheng, Y. (2022). *Fault detection on seismic structural images using a nested residual U-Net*,
  IEEE TGRS 60 — `nru_net`.
- Li et al. (2023). *Fault-Seg-Net*, Computers and Geotechnics 158, 105412 — `fault_seg_net` e `compound`.
- Di et al. (2026). *FaultEdgeFormer*, Journal of Geophysics and Engineering 23(4) — `fault_edge_former` e `tversky`.
- Myronenko, A. (2018). *3D MRI brain tumor segmentation using autoencoder regularization* — `segresnet` (MONAI).
- Bińkowski, M. et al. (2018). *Demystifying MMD GANs* — KID. Rizzo, M. & Székely, G. (2016). *Energy distance*.
- Ben-David, S. et al. (2010). *A theory of learning from different domains*. Smith, J. & Winkler, R. (2006). *The
  optimizer's curse*.
- Hansen, N. & Ostermeier, A. (2001). CMA-ES; Auger, A. & Hansen, N. (2005). IPOP-CMA-ES.
