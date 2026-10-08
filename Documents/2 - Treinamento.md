# Treinamento das redes — documento técnico

> Complemento técnico de [`1 - Metodologia.md`](1%20-%20Metodologia.md), escrito em 07/10/2026 a partir do código de
> `Model/` (`1 - Model.ipynb`, `Network/`, `Losses/`, `Transforms/`, `EMA/`, `EarlyStopping/`). Descreve o que acontece
> dentro de um treino de segmentação de falhas: o tensor que entra, a rede, os logits, a função de perda, a
> retropropagação, o otimizador, as agendas, a regularização, as métricas e a inferência. Os shapes e as contagens de
> parâmetros foram conferidos rodando as redes na CPU.

## Sumário

1. [Formulação do problema](#1-formulação-do-problema)
2. [Do arquivo ao lote](#2-do-arquivo-ao-lote)
3. [Aumentação de dados](#3-aumentação-de-dados)
4. [Blocos de construção das redes](#4-blocos-de-construção-das-redes)
5. [A `dbrnet` (`Unet3D_V2`) camada a camada](#5-a-dbrnet-unet3d_v2-camada-a-camada)
6. [A `resaceunet_zu` camada a camada](#6-a-resaceunet_zu-camada-a-camada)
7. [As outras redes do repositório](#7-as-outras-redes-do-repositório)
8. [Logits, sigmoide e decisão](#8-logits-sigmoide-e-decisão)
9. [Funções de perda](#9-funções-de-perda)
10. [Retropropagação](#10-retropropagação)
11. [Precisão mista (AMP) e escala do gradiente](#11-precisão-mista-amp-e-escala-do-gradiente)
12. [Corte do gradiente e otimizador AdamW](#12-corte-do-gradiente-e-otimizador-adamw)
13. [Agendas da taxa de aprendizado](#13-agendas-da-taxa-de-aprendizado)
14. [Regularização: dropout, decaimento, EMA, early stopping](#14-regularização-dropout-decaimento-ema-early-stopping)
15. [O laço completo de treino](#15-o-laço-completo-de-treino)
16. [Métricas durante o treino](#16-métricas-durante-o-treino)
17. [Inferência](#17-inferência)
18. [Determinismo e sementes](#18-determinismo-e-sementes)
19. [Memória e custo](#19-memória-e-custo)
20. [Como ler um treino](#20-como-ler-um-treino)
21. [Estado atual dos treinos](#21-estado-atual-dos-treinos)

---

## 1. Formulação do problema

- **Tarefa:** segmentação binária voxel a voxel. Cada voxel do volume é *falha* (1) ou *fundo* (0).
- **Entrada:** $X \in \mathbb{R}^{B\times 1\times D\times H\times W}$, com $(D, H, W) = (x, z, y)$ = (inline, tempo,
  crossline). Um canal (a amplitude sísmica), normalizado por volume para $[0,1]$ (min-max) ou para média 0 e desvio 1
  (`normalize = false`). Tamanho típico: $B = 2$ volumes de $128^3$, ou $B = 8$ recortes de $96^3$.
- **Rótulo:** $Y \in \{0,1\}^{B\times 1\times D\times H\times W}$. A classe *falha* é rara: **~7% dos voxels** nos
  datasets do projeto, distribuídos em lâminas finas (~2 voxels de espessura na direção normal ao plano, contínuas em z).
- **Modelo:** uma rede convolucional (ou híbrida com atenção) $f_\theta$ que devolve um mapa de **logits** do mesmo
  tamanho da entrada,

$$Z = f_\theta(X)\in\mathbb{R}^{B\times 1\times D\times H\times W},\qquad P = \sigma(Z) = \frac{1}{1+e^{-Z}},\qquad \hat Y = \mathbb{1}[P > 0{,}5] = \mathbb{1}[Z > 0].$$

- **Aprendizado:** minimizar uma perda $\mathcal{L}(Z, Y)$ média sobre o conjunto de treino, por descida de gradiente
  estocástica com o AdamW, usando o gradiente $\nabla_\theta \mathcal{L}$ calculado por retropropagação.
- **Critério de escolha do modelo:** o IoU binário na validação (10 volumes), com early stopping.

Uma única classe de saída (`classes = 1`) com sigmoide é o caso de todos os treinos do projeto (`MULTICLASS = False`). O
código das perdas e da métrica também aceita multiclasse (softmax + `argmax`), mas esse caminho não é usado.

---

## 2. Do arquivo ao lote

### 2.1 `CustomDataset`

```python
row  = df.iloc[index % len(df)]
img  = np.load(row.img_path).astype(np.float32)        # (128, 128, 128) em (x, z, y)
mask = np.load(row.mask_path).astype(np.float32)
if transforms: img, mask = transforms.apply(img, mask, epoch=self.epoch, index=index)
return torch.tensor(img).unsqueeze(0), torch.tensor(mask, dtype=torch.long).unsqueeze(0)   # (1, D, H, W)
```

- `__len__` = número de volumes × `n_aug`: com `n_aug = k`, cada volume aparece $k$ vezes por época, cada vez com o seu
  índice e portanto o seu sorteio de aumentação.
- A máscara viaja como `long` e vira `float` dentro da perda.

### 2.2 `DataLoader`

- Treino: `shuffle = True` (ordem nova a cada época, sorteada pelo gerador global do torch, que a semente fixa),
  `num_workers = 2`, `pin_memory = True` (cópia assíncrona para a GPU).
- Validação e teste: `shuffle = False`.
- O lote empilha os volumes: $X$ de forma `(B, 1, D, H, W)` em float32.
- **Passos por época** $= \lceil N_\text{treino}\cdot n_\text{aug} / B\rceil$: 200 volumes com lote 2 dão 100 passos; com
  lote 8, 25 passos.
- Os workers são recriados a cada época (`persistent_workers` falso); o `Trainer` incrementa `dataset.epoch` antes de
  cada época, então cada worker nasce com o contador certo.

### 2.3 Split

$$\text{teste} = 0{,}0\overline{45}\cdot N,\qquad \text{validação} = \frac{0{,}0\overline{45}}{1-0{,}0\overline{45}}\cdot(N - \text{teste})$$

via `train_test_split(..., random_state=42)` duas vezes: 220 → 200 / 10 / 10. O split não depende do trial: trials
diferentes treinam nos mesmos volumes, com inicialização, ordem dos lotes e aumentação diferentes.

---

## 3. Aumentação de dados

`Model/Transforms/index.py`, ligada pelo `augmentations` do `info.json`. Cada operação tem uma probabilidade `prob` e
roda na ordem em que aparece no JSON, só no treino. O sorteio usa um gerador próprio,
`np.random.default_rng([42 + trial, época, índice])`, que não toca no gerador global: com ou sem aumentação, a ordem dos
lotes e os pesos iniciais são os mesmos.

| operação | o que faz (eixos em `(x, z, y)`) | receita dos autores da ResACEUnet |
|---|---|---|
| `zoom` | sorteia $f$ log-uniforme em `factor` ($f\ge 1$), recorta um cubo de lado $n/f$ em posição aleatória e o reamostra ao tamanho do tile (bilinear; máscara reamostrada e binarizada em 0,5). Período, rejeito e espaçamento das falhas crescem juntos; o mergulho fica | não existe (adição do projeto, para a escala do Marlim) |
| `contrast` | gamma na faixa do próprio tile: $I' = \left(\frac{I - I_{\min}}{I_{\max}-I_{\min}}\right)^{\gamma}(I_{\max}-I_{\min}) + I_{\min}$, $\gamma \sim \mathcal{U}(0{,}9; 1{,}1)$ | p 0,2 |
| `rotate90` | `np.rot90` de $k\in\{1,2,3\}$ quartos de volta no plano `[0, 2]` (horizontal) | p 0,2 |
| `flip` | espelha os eixos `[0, 2]` juntos | p 0,2 |
| `rotate` | rotação livre em torno do centro, um ângulo $\sim\mathcal{U}(-0{,}25; 0{,}25)$ rad por plano (`[[1, 2], [0, 1]]`), matriz $R = R_{12}R_{01}$, `affine_transform` bilinear com borda zero; máscara bilinear → $\ge 0{,}5$ | p 0,2 |
| `smooth` | suavização gaussiana com $\sigma_a \sim \mathcal{U}(0{,}25; 1{,}5)$ sorteado por eixo; kernel integrado por erf, $k_j = \tfrac12[\mathrm{erf}(\frac{j+1/2}{\sigma\sqrt2}) - \mathrm{erf}(\frac{j-1/2}{\sigma\sqrt2})]$, borda zero | p 0,1 |
| `noise` | $I' = I + s\,\varepsilon$, $\varepsilon\sim\mathcal{N}(0,1)$ por voxel, $s \sim \mathcal{U}(0, 0{,}03)$ | p 0,2 |
| `crop` | recorte do tamanho `size` (96³) centrado num voxel de falha com probabilidade `pos/(pos+neg)` (1:1) ou num voxel de fundo com sinal positivo, empurrado para dentro do tile | sempre, por último |

- Giros e flips ficam no plano horizontal para nunca virar o eixo do tempo (as falhas são quase verticais e o
  empilhamento das camadas é em z).
- A rotação livre nos planos `(z, y)` e `(x, z)` inclina o eixo z em até ±14°: muda o mergulho aparente das falhas e
  das camadas.
- **Com `crop`, a rede é montada na janela de 96³** (`model.img_size = [96, 96, 96]`) e o tile inteiro de validação e
  teste é predito por janela deslizante (seção 17). O recorte é o que permite lote 8 numa placa de 24 GB e o que dá à
  rede variedade de posição — essencial para as redes com embedding de posição absoluta (seção 6).

---

## 4. Blocos de construção das redes

Todas as redes do projeto são variações de codificador–decodificador (U-Net 3D): o codificador reduz a resolução e
aumenta os canais para ganhar contexto; o decodificador volta à resolução plena e usa as **conexões de salto** (*skip
connections*) do codificador para localizar com precisão; uma convolução final $1\times1\times1$ dá o logit por voxel.

### 4.1 Convolução 3D

$$y_o(\mathbf p) = b_o + \sum_{i=1}^{C_\text{in}}\ \sum_{\mathbf k\in\{-r..r\}^3} W_{o,i,\mathbf k}\; x_i(\mathbf p + d\,\mathbf k)$$

- Kernel $k = 2r+1$ (quase sempre 3), dilatação $d$, *padding* $d\cdot r$ para manter o tamanho; parâmetros
  $C_\text{out}C_\text{in}k^3$ (+ $C_\text{out}$ de viés, omitido quando há normalização logo depois).
- **Convolução dilatada** ($d > 1$): o mesmo kernel $3^3$ amostra a cada $d$ voxels e cobre $2d+1$ — campo receptivo
  maior sem perder resolução nem somar parâmetros.
- **Convolução com passo** (*stride* $s$): reduz a resolução por $s$ (a `resaceunet_zu` usa kernel 4, passo 4 no
  *stem*).
- **Convolução transposta**: a adjunta da convolução com passo; sobe a resolução de forma aprendida.
- A mesma conta em todos os voxels (equivariância a translação) é o que permite treinar em 128³ e predizer em qualquer
  tamanho — exceto nas redes com parâmetros por posição.

### 4.2 Normalização

- **GroupNorm** (a das redes do projeto): divide os $C$ canais em $G$ grupos (8, ou 1 se $C$ não for múltiplo de 8) e,
  para cada amostra e grupo, com média $\mu_g$ e variância $\sigma_g^2$ sobre canais do grupo × voxels,

  $$\hat x = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}},\qquad y = \gamma_c\,\hat x + \beta_c.$$

  Não depende do lote: funciona igual com $B = 2$ e é idêntica em treino e avaliação.
- **BatchNorm** usa a média e a variância do *lote* no treino e médias móveis na avaliação. Com $B = 2$ volumes as
  estatísticas são ruidosas e a rede fica pior em `eval()` do que em `train()` por motivo estatístico (medido na MACNN:
  gap de +0,031 com BatchNorm, zero com GroupNorm). Por isso as redes adaptadas de artigos trocaram BN por GN; só a
  `unet_3d` e o `conv51` dos blocos das ResACEUnet dos autores ainda usam BN (lote 8 de recortes na receita deles,
  lote 2 nas rodadas em 128³).
- **InstanceNorm** = GroupNorm com um canal por grupo (os `UnetResBlock` da ResACEUnet). **LayerNorm** normaliza cada
  token sobre os canais (blocos transformer).
- Uma consequência: primeira convolução sem viés + normalização ⇒ a rede é quase **cega a ganho e deslocamento
  globais** da amplitude.

### 4.3 Ativações

| ativação | fórmula | derivada | onde |
|---|---|---|---|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x>0]$ | `unet_3d`, `macnn`, `fault_seg_net`, `nru_net` |
| LeakyReLU | $\max(x, a x)$, $a = 0{,}1$ (`dbrnet`, `resaceunet_grva`) ou $0{,}01$ (blocos MONAI) | $1$ ou $a$ | `dbrnet`, ResACEUnet |
| GELU | $x\,\Phi(x)$ | $\Phi(x) + x\varphi(x)$ | MLP dos transformers |
| ReLU6 | $\min(\max(0,x),6)$ | $\mathbb{1}[0<x<6]$ | `fault_edge_former` |

A LeakyReLU nunca zera o gradiente (evita neurônios "mortos").

### 4.4 Redução e subida de resolução

- **MaxPool** $2^3$ (ou anisotrópico, p. ex. $(1,2,2)$): cada saída é o máximo da janela; no backward o gradiente vai
  só para a posição do máximo.
- **Vizinho mais próximo** (*nearest*): cada voxel vira um bloco de $2^3$ cópias; no backward o gradiente das cópias é
  somado.
- **Trilinear**: interpolação linear nos três eixos; no backward o gradiente é distribuído pelos mesmos pesos.

### 4.5 Conexões de salto e residuais

- **Concatenação** (U-Net): o decodificador recebe `cat([subido, skip])` nos canais — a informação fina do codificador
  chega à saída sem passar pelo gargalo.
- **Soma residual**: $y = x + F(x)$. Como $\partial y/\partial x = I + \partial F/\partial x$, o gradiente sempre tem um
  caminho de identidade e não some nas redes profundas.

### 4.6 Dropout

- `Dropout3d(p)`: no treino zera **canais inteiros** com probabilidade $p$ e multiplica os restantes por $1/(1-p)$; na
  avaliação é a identidade. Em volumes, zerar voxels isolados regularizaria pouco (vizinhos são correlacionados); zerar
  o canal obriga a rede a não depender de um detector só.
- Cada rede distribui o `dropout` do `info.json` pelos níveis (seção 5).

### 4.7 Atenção

Para tokens $T\in\mathbb{R}^{N\times C}$, a atenção de produto escalar é
$\mathrm{softmax}\!\big(QK^\top/\sqrt{d}\big)V$, com $Q, K, V$ projeções lineares de $T$. O custo é $O(N^2)$: em
$24^3 = 13\,824$ tokens já seria proibitivo em 3D, por isso as redes do projeto usam variantes lineares (seção 6),
atenção por janela (`fault_edge_former`) ou atenção só no gargalo (`resaceunet_grva`).

### 4.8 Campo receptivo

O campo receptivo é a região da entrada que influencia um voxel da saída. Na `dbrnet`, contando kernels, poolings e
dilatações, ele chega a **~118 voxels no eixo x** (que só é reduzido nos dois níveis mais profundos) e **passa do tamanho
do tile (128) em z e y** já no gargalo — cada voxel da saída "vê" o tile inteiro na vertical.

---

## 5. A `dbrnet` (`Unet3D_V2`) camada a camada

Origem: U-Net 3D modificada do GRVA. Arquivo `Model/Network/types/Unet3D_V2.py`. Configuração de referência:
`num_filters` $f = 32$, `dropout` $p = 0{,}1$, entrada $128^3$. **40,04 M parâmetros.**

### 5.1 Blocos

- `Conv3DBlock`: `conv3 → GroupNorm(8) → LeakyReLU(0,1)`, duas vezes (convoluções sem viés).
- `EncoderBlock`: `Conv3DBlock` → guarda a saída como *skip* → `MaxPool` → `Dropout3d`.
- `DecoderBlock`: sobe por vizinho mais próximo → `conv3 → GN → LeakyReLU` → concatena o *skip* → `Dropout3d` →
  `Conv3DBlock`.
- `DilatedBottleneck`: duas convoluções $3^3$ dilatadas ($d = 2$ e $d = 4$), cada uma `conv → GN → LeakyReLU`, somadas
  residualmente: $x \leftarrow x + g_2(x)$, $x \leftarrow x + g_4(x)$.
- Entrada com lado não múltiplo de 16 é completada com zeros e o logit é cortado de volta.

### 5.2 Shapes (conferidos)

| estágio | operação | canais | saída $(x, z, y)$ | dropout |
|---|---|---|---|---|
| entrada | — | 1 | 128 × 128 × 128 | |
| `enc1` | 2 × conv3 → *skip* → MaxPool $(1,2,2)$ | 32 | 128 × 128 × 128 → 128 × 64 × 64 | 0,025 |
| `enc2` | idem, MaxPool $(1,2,2)$ | 64 | 128 × 64 × 64 → 128 × 32 × 32 | 0,075 |
| `enc3` | idem, MaxPool $2^3$ | 128 | 128 × 32 × 32 → 64 × 16 × 16 | 0,15 |
| `enc4` | idem, MaxPool $2^3$ | 256 | 64 × 16 × 16 → 32 × 8 × 8 | 0,25 |
| gargalo | 2 × conv3 → `Dropout3d` → dilatadas $d=2$, $d=4$ (residuais) | 512 | 32 × 8 × 8 | 0,3 |
| `dec1` | sobe $2^3$, conv3, + *skip* `enc4` | 256 | 64 × 16 × 16 | 0,1 |
| `dec2` | sobe $2^3$, + *skip* `enc3` | 128 | 128 × 32 × 32 | 0,1 |
| `dec3` | sobe $(1,2,2)$, + *skip* `enc2` | 64 | 128 × 64 × 64 | 0,1 |
| `dec4` | sobe $(1,2,2)$, + *skip* `enc1` | 32 | 128 × 128 × 128 | 0,1 |
| saída | conv $1^3$ | 1 | 128 × 128 × 128 (logits) | |

- O pooling anisotrópico preserva a resolução do eixo x (inline) nos dois primeiros níveis; z e y são reduzidos 16×, x
  só 4×.
- **Onde ficam os parâmetros:** gargalo 24,8 M (**61,9%**: `bot_conv` 10,6 M + `bot_dilated` 14,2 M), `dec1` 8,8 M
  (22,1%), `enc4` 2,7 M; a resolução plena (`enc1` + `dec4`) tem só 0,17 M (0,4%). A capacidade está no nível mais
  grosso, onde a lâmina de falha (2 voxels) já não é resolvível — medida que motivou o rebalanceamento da
  `resaceunet_grva` original.

---

## 6. A `resaceunet_zu` camada a camada

Origem: ResACEUnet de Zu et al. (2024), cópia do primeiro commit do repositório dos autores (a versão de 3 estágios
descrita no artigo). Arquivo `Model/Network/types/ResACEUnet_Zu.py`. Mudanças em relação ao original: `trunc_normal_`
do MONAI no lugar do timm, `rate1`/`rate2` inicializados em 0,5 (no original nascem de memória não inicializada e dão
NaN) e **sem a sigmoide final** (as perdas recebem logits). No seletor, `hidden_size = 32f` e
`dims = (2f, 4f, 32f)`; $f = 16$ reproduz o artigo. **59,44 M parâmetros em 96³** (72,66 M em 128³: o número depende da
janela, porque há parâmetros por posição).

### 6.1 Shapes (conferidos, entrada 96³, $f = 16$)

| estágio | operação | canais | saída |
|---|---|---|---|
| `encoder1` | `UnetResBlock`: conv3 → InstanceNorm → LeakyReLU(0,01) → conv3 → IN, + atalho conv1 → IN, soma, LeakyReLU | 16 | 96³ |
| *stem* | conv $4^3$ passo 4 → GroupNorm | 32 | 24³ (13 824 tokens) |
| estágio 1 | 3 × `TransformerBlock` (projeção espacial 64) | 32 | 24³ |
| descida | conv $2^3$ passo 2 → GN | 64 | 12³ (1 728 tokens) |
| estágio 2 | 3 × `TransformerBlock` (projeção 64) | 64 | 12³ |
| descida | conv $2^3$ passo 2 → GN | 512 | 6³ (216 tokens) |
| estágio 3 | 3 × `TransformerBlock` (projeção 32) | 512 | 6³ |
| `proj_feat` | `view(B, 6, 6, 6, 512)` + `permute` | 512 | 6³ |
| `decoder4` | conv transposta $2^3$ (512 → 64) + soma do estágio 2 + 3 × `TransformerBlock` | 64 | 12³ |
| `decoder3` | conv transposta $2^3$ (64 → 32) + soma do estágio 1 + 3 × `TransformerBlock` | 32 | 24³ |
| `decoder2` | conv transposta $4^3$ passo 4 (32 → 16) + soma do `encoder1` + `UnetResBlock` | 16 | 96³ |
| `out` | conv $1^3$ | 1 | 96³ (logits) |

- Os saltos são por **soma**, não concatenação.
- **`proj_feat` não é a identidade:** aplicado a um tensor já em `(B, C, H, W, D)`, o `view` para `(B, 6, 6, 6, 512)`
  reinterpreta a ordem dos elementos (conferido: a entrada do `decoder4` é exatamente
  `enc3.reshape(1, 6, 6, 6, 512).permute(0, 4, 1, 2, 3)`, diferente de `enc3`). É uma permutação fixa dos 110 592
  valores do gargalo herdada do código dos autores; a rede aprende através dela, mas a vizinhança espacial no gargalo
  não é preservada.

### 6.2 O `TransformerBlock`

Para o volume $S\in\mathbb{R}^{C\times H\times W\times D}$, com $N = HWD$ tokens:

$$
\begin{aligned}
T &= \mathrm{flatten}(S) + E_\text{pos},\qquad E_\text{pos}\in\mathbb{R}^{N\times C}\ \text{(aprendido, nasce zero)}\\
A &= T + \gamma\odot\mathrm{ACE}(\mathrm{LN}(T)),\qquad \gamma\in\mathbb{R}^C\ \text{nasce } 10^{-6}\ \text{(layer scale)}\\
S' &= \mathrm{vol}(A),\qquad \text{saída} = S' + \mathrm{Conv}_{1^3}\big(\mathrm{Dropout3d}_{0,1}(\mathrm{UnetResBlock}_\text{BN}(S'))\big)
\end{aligned}
$$

- O *layer scale* $\gamma = 10^{-6}$ faz o bloco nascer quase como identidade + ramo convolucional: a atenção "liga"
  aos poucos, conforme $\gamma$ cresce.
- O `UnetResBlock` interno (`conv51`) tem duas convoluções $3^3$ de $C\to C$ com BatchNorm. No estágio 3 ($C = 512$)
  cada um custa 14,2 M parâmetros: **os três do estágio 3 somam 42,5 M, 71% da rede**. O estágio 3 inteiro (6³) tem
  47,6 M (80%).

### 6.3 O bloco de atenção ACE

Cada token passa por uma projeção linear sem viés $C \to 4C$ que dá $Q, K, V_\text{CA}, V_\text{SA}$, divididos em
$h = 4$ cabeças de $d = C/h$ canais. Três ramos em paralelo:

- **Atenção de canal** (transposta, como no UNETR++/XCiT): com $\hat Q, \hat K \in \mathbb{R}^{d\times N}$
  normalizados em L2 ao longo dos tokens,
  $$A_\text{CA} = \mathrm{softmax}\big(\tau_1\,\hat Q\hat K^\top\big)\in\mathbb{R}^{d\times d},\qquad X_\text{CA} = A_\text{CA}\,V_\text{CA}.$$
  Custo $O(N d^2)$: mistura canais com pesos que dependem do volume inteiro.
- **Atenção espacial com projeção** (tipo Linformer): uma linear $E: \mathbb{R}^N \to \mathbb{R}^p$ ($p$ = 64, 64, 32)
  comprime o eixo dos tokens de $K$ e de $V_\text{SA}$ (a mesma camada para os dois),
  $$A_\text{SA} = \mathrm{softmax}\big(\tau_2\,\hat Q^\top (K E^\top)\big)\in\mathbb{R}^{N\times p},\qquad X_\text{SA} = A_\text{SA}\,(V_\text{SA}E^\top)^\top.$$
  Custo $O(N p d)$, linear em $N$.
- **Ramo convolucional** (do ACmix): os $4h$ mapas de $Q, K, V$ são misturados por uma convolução $1\times1$ (para 16) e
  por uma convolução 1D de kernel 3 em grupos, ao longo da sequência de tokens.
- **Fusão:** $\mathrm{ACE}(T) = r_1\,[\,W_\text{SA}X_\text{SA}\ \|\ W_\text{CA}X_\text{CA}\,] + r_2\,X_\text{conv}$, com
  $W$ projeções $C\to C/2$ e $r_1, r_2$ aprendidos (nascem 0,5); $\tau_1, \tau_2$ são temperaturas aprendidas por cabeça.

### 6.4 Dependência de posição

$E_\text{pos}$ ($N\times C$ por bloco) e $E$ ($N \to p$) têm tamanho que depende de $N$, ou seja, da janela: **9,6 M dos
59,4 M parâmetros em 96³** são presos a posições absolutas. Consequências:

- a rede só aceita a janela em que foi montada; por isso o recorte 96³ no treino e a janela deslizante na inferência;
- sem variedade de posição (recorte aleatório + giros/flips), a rede tende a memorizar o tile — medido na
  `resaceunet_wu` (mesma família): espelhar um tile de treino derrubou o IoU de 0,74 para 0,52.

---

## 7. As outras redes do repositório

Parâmetros contados na configuração de referência (128³, salvo indicação).

| nome no `info.json` | arquivo | ideia central | params |
|---|---|---|---|
| `unet_3d` | `UNet3D.py` | U-Net 3D clássica de 4 níveis: `(conv3-BN-ReLU)×2`, MaxPool $2^3$, convolução transposta para subir, concatenação | 25,9 M ($f$ 32) |
| `dbrnet` | `Unet3D_V2.py` | seção 5 | 40,0 M ($f$ 32) |
| `segresnet` | MONAI `SegResNet` | codificador residual (GN-ReLU-conv) assimétrico, decodificador leve com subida trilinear | 18,8 M ($f$ 32) |
| `resaceunet_grva` | `ResACEUnet.py` | U-Net residual com unidade ACE (atenção de canal por média+máximo e porta espacial $1\times7\times7$, ordem CBAM, portas abertas na inicialização), *attention gates* nos saltos, DropPath, gargalo com convoluções dilatadas + 2 blocos transformer globais (8 cabeças, embedding de posição interpolável); pooling derivado do `input_shape` | 49,6 M ($f$ 32) |
| `resaceunet_wu` | `ResACEUnet_Wu.py` | a `ResACEUNet2` atual dos autores: 4 estágios ACE, exige cubo múltiplo de 32 | 43,9 M ($f$ 16) |
| `resaceunet_zu` | `ResACEUnet_Zu.py` | seção 6 | 72,7 M (128³) / 59,4 M (96³) |
| `macnn` | `MACNN.py` | U-Net de 3 níveis com bloco de tripla convolução (a do meio dilatada, entrada reconcatenada, soma residual) e atenção espaço-canal **multiescala**: cada *skip* é refinado por mapas construídos a partir dos três codificadores | 11,8 M ($f$ 32) |
| `fault_seg_net` | `FaultSegNet.py` | 5 estágios; módulo multiescala residual (três ramos dilatados com campos receptivos 3/7/9); saltos trocados por uma atenção ao longo de z (tokens = z, features = $x\cdot y$, escala $1/\sqrt d$); termo de perda MSR desligado (`ETA = 1`) | 17,4 M ($f$ 32) |
| `nru_net` | `NRUNet.py` | U-Net de 3 níveis em que cada codificador/decodificador é uma U-Net residual própria; três mapas de saída (um por nível) fundidos por conv $1^3$; *checkpoint* nas unidades de resolução plena | 19,3 M ($f$ 32) |
| `fault_edge_former` | `FaultEdgeFormer.py` | Sobel 3D treinável em 9 direções na entrada; HRNet 3D de 3 fluxos (1/2, 1/4, 1/8) com pares de blocos Swin (janela 7, deslocada) e MLP com conv *depthwise*; saída em 1/2 interpolada | 0,99 M ($f$ 9) |

Detalhes de implementação e decisões medidas de cada uma estão nos comentários dos arquivos.

---

## 8. Logits, sigmoide e decisão

- **Logit** é a saída linear da última convolução, $z\in\mathbb{R}$, interpretada como log-chance:
  $z = \log\frac{p}{1-p}$. $z = 0$ ⇔ $p = 0{,}5$; $z = \pm 4{,}6$ ⇔ $p \approx 0{,}99 / 0{,}01$.
- **Nenhuma rede aplica a sigmoide dentro do `forward`.** A sigmoide fica na perda e na inferência porque
  $\log\sigma(z)$ calculado como `log(sigmoid(z))` estoura para $z$ muito negativo (em float16 já para $|z| \gtrsim 17$),
  enquanto a forma estável
  $$\log\sigma(z) = -\mathrm{softplus}(-z) = -\log(1 + e^{-z}),\qquad \mathrm{BCE}(z, y) = \max(z,0) - zy + \log\!\big(1 + e^{-|z|}\big)$$
  nunca estoura. Por isso também as perdas são calculadas em **float32 fora do autocast**.
- **Combinação perda–ativação:** para a BCE com sigmoide, $\partial\mathcal L/\partial z = p - y$ — o gradiente não
  satura quando a rede está muito errada (ao contrário de um erro quadrático sobre $p$, que seria multiplicado por
  $p(1-p)\to 0$).
- **Inicialização:** a última convolução nasce com pesos e viés pequenos (`kaiming_uniform` padrão do PyTorch, viés
  $\mathcal{U}(\pm 1/\sqrt{C})$), então $z\approx 0$ e $p\approx 0{,}5$ em todo voxel. Valores iniciais esperados da
  perda com 7% de falha: BCE $\ln 2 \approx 0{,}69$; Dice $1 - \frac{0{,}07}{0{,}5 + 0{,}07}\approx 0{,}88$; focal
  ($\gamma = 2$) $0{,}25\ln 2 \approx 0{,}17$. Logo `dice_ce` começa em ~1,57 e `dice_focal` em ~1,05 — útil para
  conferir se um treino começou certo.
- **Decisão:** $\hat y = \mathbb{1}[\sigma(z) > 0{,}5]$, o mesmo limiar no treino, no teste e no Marlim (onde o
  `FaultStickExtractor` trabalha sobre a probabilidade contínua). Medido no `dataset_wu`: a probabilidade sai bimodal e
  o IoU é quase chato entre os limiares 0,2 e 0,8, então não há ganho em ajustar o limiar.

---

## 9. Funções de perda

Notação: $p_i = \sigma(z_i)$, $g_i = y_i \in\{0,1\}$, $N$ voxels. Todas as perdas recebem logits, convertem para
float32 e ajustam o shape do alvo ao do logit. Escolha pelo `loss` do `info.json` (`Model/Losses/index.py`).

### 9.1 `cross_entropy` — BCE com logits

$$\mathcal L_\text{BCE} = -\frac1N\sum_i\big[g_i\log p_i + (1-g_i)\log(1-p_i)\big],\qquad \frac{\partial\mathcal L_\text{BCE}}{\partial z_i} = \frac{p_i - g_i}{N}.$$

Média sobre todos os voxels do lote. Cada voxel pesa igual: com 93% de fundo, a soma dos gradientes do fundo domina no
início, e o caminho mais barato é prever "fundo" em toda parte. Sem `pos_weight`.

### 9.2 Dice suave (o termo Dice do MONAI)

Por amostra $b$ (o `DiceLoss` do MONAI usa `batch = False`, `squared_pred = False`, $s = 10^{-5}$):

$$\mathcal L_D^{(b)} = 1 - \frac{2 I + s}{S_p + S_g + s},\qquad I = \sum_i p_i g_i,\ S_p = \sum_i p_i,\ S_g = \sum_i g_i,$$

e a perda é a média sobre as amostras. Gradiente em relação a uma probabilidade:

$$\frac{\partial\mathcal L_D}{\partial p_j} = -\frac{2 g_j\,(S_p + S_g + s) - (2I + s)}{(S_p + S_g + s)^2},\qquad \frac{\partial\mathcal L_D}{\partial z_j} = \frac{\partial\mathcal L_D}{\partial p_j}\,p_j(1-p_j).$$

- Para voxel de falha ($g_j = 1$) o gradiente é negativo (sobe $p_j$); para fundo ($g_j = 0$) é positivo e
  proporcional à sobreposição atual $2I+s$ (desce $p_j$).
- A razão entre o empurrão por voxel de falha e por voxel de fundo é $\frac{2(S_p+S_g) + s - 2I}{2I + s}$: grande
  quando a sobreposição é pequena. A perda é **normalizada pelo volume da classe**, então não é dominada pelo fundo —
  é por isso que ela lida com o desbalanceamento.
- O gradiente em $p$ é o mesmo para todos os voxels de uma classe; quem diferencia os voxels é o fator $p(1-p)$ da
  sigmoide, que anula o gradiente nos voxels já saturados.
- Num tile **sem falha**, $I = S_g = 0$ e $\mathcal L_D = 1 - s/(S_p + s)\approx 1$ com gradiente $\approx s/S_p^2 \approx
  0$: o tile não ensina nada pelo termo Dice (só pelo outro termo da soma). Nos datasets atuais quase todo tile tem
  falha.

### 9.3 Focal

$$\mathcal L_F = \frac1N\sum_i -\alpha_t\,(1 - p_t)^\gamma\log p_t,\qquad p_t = \begin{cases}p_i & g_i = 1\\ 1 - p_i & g_i = 0\end{cases},\quad \alpha_t = \begin{cases}\alpha & g_i = 1\\ 1-\alpha & g_i = 0\end{cases}$$

com gradiente

$$\frac{\partial \mathcal L_F}{\partial z} = \begin{cases}\alpha\,(1-p)^\gamma\big[\gamma\,p\log p - (1-p)\big] & g = 1\\[2pt] (1-\alpha)\,p^\gamma\big[p - \gamma\,(1-p)\log(1-p)\big] & g = 0\end{cases}\quad(\div N).$$

- O fator $(1 - p_t)^\gamma$ ($\gamma = 2$) apaga os voxels fáceis: um voxel de fundo com $p = 0{,}05$ pesa
  $0{,}05^2 = 0{,}0025$ do que pesaria na BCE. O treino se concentra nos difíceis — na prática, a borda da falha.
- `focal` sozinha usa $\alpha = 0{,}93$ (`Losses.FOCAL_ALPHA`): peso 0,93 na falha e 0,07 no fundo, aproximadamente o
  inverso da frequência das classes. Dentro da `dice_focal` e da `compound` não há $\alpha$.

### 9.4 Perdas compostas

| nome | fórmula | origem |
|---|---|---|
| `dice_ce` | $\mathcal L_D + \mathcal L_\text{BCE}$ (pesos 1:1; no canal único o MONAI usa a BCE com logits) | BCE + Dice da ResACEUnet (eq. 5); o código dos autores não eleva $p$ ao quadrado no Dice (a eq. 7 eleva) — fica o código |
| `dice_focal` | $\mathcal L_D + \mathcal L_F$ ($\gamma = 2$, sem $\alpha$; 1:1) | MONAI `DiceFocalLoss`; no MONAI 1.5.2 o termo focal é sempre sigmoide (irrelevante aqui, que é canal único) |
| `smooth_dice` | $1 - \frac{2\sum p g + 1}{\sum p + \sum g + 1}$ com somas **globais sobre o lote** | MACNN (eq. 7), também a perda da NRU |
| `compound` | $0{,}3\,\mathcal L_\text{smooth\_dice} + 0{,}7\,\mathcal L_F$ ($\gamma = 2$, sem $\alpha$) | Fault-Seg-Net (eq. 11) |
| `tversky` | $1 - \frac{TP + 1}{TP + 0{,}3\,FP + 0{,}7\,FN + 1}$, $TP = \sum pg$, $FP = \sum p(1-g)$, $FN = \sum (1-p)g$, globais no lote | FaultEdgeFormer (eq. 28); pesa mais o falso negativo |

- A soma Dice + BCE/focal junta um termo **global** (Dice: forma e sobreposição do conjunto) com um termo **local**
  (BCE/focal: cada voxel), que dá gradiente bem comportado mesmo onde o Dice é plano (tiles sem falha, início do treino).
- A `fault_seg_net` pode somar um termo próprio (`compose`): $\eta\,\mathcal L + (1-\eta)\,\mathcal L_\text{MSR}$; com
  `ETA = 1` (padrão, porque o termo foi medido nocivo) a perda é só a escolhida.
- O `val_loss` é a mesma perda na validação; é o sinal que a agenda `plateau` acompanha.

---

## 10. Retropropagação

### 10.1 O grafo e a regra da cadeia

O forward compõe camadas, $Z = f_L\circ\cdots\circ f_1(X)$, e o autograd do PyTorch grava cada operação e os tensores
de que a derivada vai precisar. O `loss.backward()` percorre o grafo de trás para a frente, propagando o **gradiente da
perda em relação à saída de cada camada**, $\delta_l = \partial\mathcal L/\partial h_l$:

$$\delta_L = \frac{\partial\mathcal L}{\partial Z},\qquad \delta_{l-1} = J_{f_l}^\top\,\delta_l,\qquad \nabla_{\theta_l}\mathcal L = \Big(\frac{\partial f_l}{\partial\theta_l}\Big)^{\!\top}\delta_l.$$

Nunca se monta o jacobiano: cada operação implementa o produto vetor–jacobiano. O ponto de partida é o mapa
$\partial\mathcal L/\partial Z$ da seção 9 — um tensor do tamanho do volume, um valor por voxel.

### 10.2 Como o gradiente atravessa cada tipo de camada

| camada | forward | backward |
|---|---|---|
| Convolução | $y_o(\mathbf p) = \sum_{i,\mathbf k} W_{o,i,\mathbf k}\,x_i(\mathbf p+\mathbf k)$ | pesos: $\frac{\partial\mathcal L}{\partial W_{o,i,\mathbf k}} = \sum_{b,\mathbf p}\delta_o(\mathbf p)\,x_i(\mathbf p+\mathbf k)$ (correlação entre entrada e gradiente, somada sobre o lote e **todas as posições**); entrada: $\frac{\partial\mathcal L}{\partial x_i(\mathbf q)} = \sum_{o,\mathbf k}W_{o,i,\mathbf k}\,\delta_o(\mathbf q-\mathbf k)$ (convolução transposta); viés: $\sum_{b,\mathbf p}\delta_o(\mathbf p)$ |
| Conv. transposta | adjunta da convolução com passo | convolução com passo do gradiente |
| GroupNorm | $y = \gamma\hat x + \beta$ | $\frac{\partial\mathcal L}{\partial x} = \frac{1}{\sigma}\big(g - \overline{g} - \hat x\,\overline{g\hat x}\big)$ por grupo, com $g = \gamma\,\partial\mathcal L/\partial y$ e médias sobre o grupo; $\partial\mathcal L/\partial\gamma = \sum g\hat x$, $\partial\mathcal L/\partial\beta = \sum \partial\mathcal L/\partial y$ |
| LeakyReLU | $\max(x, ax)$ | multiplica por 1 ou $a$ |
| MaxPool | máximo da janela | o gradiente vai só para a posição do máximo |
| Vizinho mais próximo | copia cada voxel | soma o gradiente das cópias |
| Trilinear | combinação linear | distribui pelos mesmos pesos |
| Concatenação | junta canais | separa o gradiente: a parte do *skip* volta direto ao codificador |
| Soma residual | $x + F(x)$ | o gradiente passa inteiro pela identidade e também por $F$ |
| Dropout3d | máscara de canais × $1/(1-p)$ | a mesma máscara e escala |
| Softmax | $a = \mathrm{softmax}(s)$ | $\partial\mathcal L/\partial s = a\odot\big(\delta - \langle a,\delta\rangle\big)$ |
| Sigmoide (na perda) | $p = \sigma(z)$ | $\times\,p(1-p)$; com a BCE, combinado em $p - y$ |

### 10.3 O que isso significa na prática

- **Cada peso de convolução recebe a soma das contribuições de milhões de voxels**: um lote de 2 volumes de $128^3$ tem
  4,2 M voxels (~290 mil de falha). O gradiente é uma estimativa com muitas amostras, mas fortemente correlacionadas
  (vizinhos), o que explica treinar bem com lote 2.
- **Os saltos encurtam o caminho do gradiente:** o erro de localização na resolução plena chega às primeiras camadas do
  codificador pela concatenação do `dec4`, sem atravessar o gargalo.
- **Onde o gradiente se concentra:** com Dice e focal, nos voxels de falha e nos de fundo que a rede ainda erra — na
  borda das lâminas. É coerente com a medição de que ~92% do erro no `dataset_wu` é de borda.
- **Na `resaceunet_zu`:** o gradiente para os parâmetros da ACE chega multiplicado por $\gamma = 10^{-6}$ no início;
  $\gamma$ recebe $\langle \mathrm{ACE}(\mathrm{LN}(T)),\ \delta\rangle$ e cresce primeiro. A parte convolucional do
  bloco é a que aprende antes.
- **Memória:** o autograd guarda as ativações do forward até o backward. É isso, e não os parâmetros, que limita o
  tamanho do lote: um mapa de 32 canais em $128^3$ ocupa 268 MB por amostra em float32, e a `dbrnet` guarda dezenas de
  mapas desse porte (~15 GiB com lote 2). As redes mais pesadas usam **gradient checkpointing** (`macnn`, `nru_net`,
  `fault_edge_former`): não guardam as ativações de um trecho e o recalculam no backward, trocando tempo por memória com
  resultado idêntico.

---

## 11. Precisão mista (AMP) e escala do gradiente

Com `"amp": true`:

- **Autocast:** dentro de `torch.amp.autocast('cuda')`, convoluções e multiplicações de matriz rodam em float16;
  reduções e normalizações ficam em float32; as perdas são forçadas a float32.
- **`GradScaler`:** o float16 não representa gradientes muito pequenos (abaixo de ~$6\times10^{-8}$ viram zero). A perda
  é multiplicada por um fator $S$ antes do backward, e os gradientes voltam divididos por $S$:

  1. `scaler.scale(loss).backward()` — gradientes de $S\cdot\mathcal L$;
  2. `scaler.unscale_(optimizer)` — divide por $S$ **antes do corte**, para o corte ver a norma verdadeira;
  3. `scaler.step(optimizer)` — se algum gradiente for `inf`/`NaN`, **pula o passo**;
  4. `scaler.update()` — $S$ cai à metade quando houve estouro e dobra após 2000 passos seguidos sem estouro
     ($S_0 = 2^{16}$).
- Com `"amp": false` (padrão), o `GradScaler` é criado desligado e todas as chamadas viram passagem direta: o mesmo
  código serve aos dois modos.
- Na P6000 (Pascal, sem tensor cores) o ganho do AMP é sobretudo de **memória**; medido na rede de similaridade, não
  acelerou a inferência.

---

## 12. Corte do gradiente e otimizador AdamW

### 12.1 Corte da norma global

`clip_grad_norm_(parâmetros, max_norm=1.0)`: com $\lVert g\rVert = \sqrt{\sum_l \lVert g_l\rVert^2}$ sobre todos os
parâmetros,

$$g \leftarrow g\cdot\min\!\Big(1,\ \frac{1}{\lVert g\rVert + 10^{-6}}\Big).$$

A direção não muda; só passos com norma maior que 1 são encurtados. Protege contra picos (um lote com falha muito
diferente, o início do Dice), que de outra forma entrariam com peso grande nos momentos do Adam.

### 12.2 AdamW

`optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` com $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$,
$\epsilon = 10^{-8}$. No passo $t$, com o gradiente já cortado $g_t$ e a taxa $\eta_t$ da agenda:

$$
\begin{aligned}
\theta &\leftarrow \theta - \eta_t\,\lambda\,\theta &&\text{(decaimento desacoplado, } \lambda = 10^{-4})\\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t &&\text{(média do gradiente)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 &&\text{(média do quadrado)}\\
\hat m_t &= m_t/(1-\beta_1^t),\quad \hat v_t = v_t/(1-\beta_2^t) &&\text{(correção do viés inicial)}\\
\theta &\leftarrow \theta - \eta_t\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

- O passo de cada parâmetro tem tamanho da ordem de $\eta_t$, **independente da escala do seu gradiente**: parâmetros
  do gargalo e da resolução plena andam no mesmo ritmo, mesmo com gradientes de ordens de grandeza diferentes.
- O decaimento é **desacoplado** (AdamW): encolhe o peso diretamente, em vez de entrar no gradiente e ser normalizado
  por $\sqrt{v}$. Com $\lambda = 10^{-4}$ e $\eta = 10^{-3}$ o encolhimento é de $10^{-7}$ por passo — regularização
  fraca. Ele vale para todos os parâmetros (pesos, normalizações, embeddings de posição, temperaturas), sem grupos
  separados.
- `zero_grad()` (com `set_to_none`) antes de cada forward: o gradiente não acumula entre passos.

---

## 13. Agendas da taxa de aprendizado

### 13.1 `plateau`

`ReduceLROnPlateau(mode='min', factor=0.5, patience=10)` sobre o **`val_loss`**, um passo por época: se o `val_loss`
não melhora (limiar relativo $10^{-4}$) por 10 épocas, $\eta \leftarrow \eta/2$. Atenção à diferença de sinais: a
agenda olha o `val_loss`, o early stopping olha o `val_iou`.

### 13.2 `cosine` (aquecimento + cosseno, a agenda da ResACEUnet)

Por passo $t$, com $W = 10\times$ passos por época e $T = $ `epochs` $\times$ passos por época:

$$
\eta(t) = \begin{cases}
\eta_0 + (\eta_\text{pico} - \eta_0)\,\dfrac{t}{W}, & t < W\\[8pt]
\eta_1 + (\eta_\text{pico} - \eta_1)\cdot\dfrac12\Big[1 + \cos\Big(\pi\,\dfrac{t - W}{T - W}\Big)\Big], & t \ge W
\end{cases}
\qquad \eta_0 = 10^{-6},\ \eta_1 = 10^{-7},\ \eta_\text{pico} = \texttt{lr}.
$$

- O aquecimento evita que os primeiros passos, com momentos do Adam ainda mal estimados e atenção recém-nascida,
  desmontem a inicialização.
- Exemplo da rodada atual (lote 8, 200 recortes → 25 passos/época, 200 épocas): $W = 250$, $T = 5000$. No fim da época
  7, $t = 175$ e $\eta = 10^{-6} + (10^{-3} - 10^{-6})\cdot 175/250 = 7{,}003\times10^{-4}$ — o valor que o
  `progress.json` registrou.
- O early stopping (paciência 20) pode encerrar antes de o cosseno terminar.

---

## 14. Regularização: dropout, decaimento, EMA, early stopping

### 14.1 EMA dos pesos (`"ema": true`)

Depois de cada passo do otimizador, uma cópia dos pesos acompanha a média móvel exponencial do modelo vivo:

$$\theta_\text{EMA} \leftarrow d_t\,\theta_\text{EMA} + (1 - d_t)\,\theta,\qquad d_t = \min\!\Big(0{,}999,\ \frac{1 + t}{10 + t}\Big).$$

- O decaimento sobe aos poucos (0,18 no passo 1, 0,92 no passo 100, 0,999 a partir de ~9000 passos): no começo a
  média segue o modelo, depois fica lenta (meia-vida de ~690 passos em 0,999).
- Tensores float (pesos e estatísticas de BatchNorm) entram na média; os inteiros são copiados.
- **Validação, early stopping e teste usam os pesos do EMA.** Ao final, o melhor estado do EMA é carregado no modelo
  vivo, que é o que se salva.
- A média de pesos ao longo da trajetória suaviza o ruído dos passos e costuma dar um modelo mais estável que o último
  ponto.

### 14.2 Early stopping (`EarlyStopping`)

- Monitora o `val_iou` (`mode='max'`); só conta como melhora $\text{val\_iou} > \text{melhor} + 10^{-4}$.
- A cada melhora guarda uma cópia profunda do `state_dict` (do EMA, se ligado) e zera o contador; senão soma 1.
- Para quando o contador chega à `patience` (padrão 15). No fim do treino, `restore_best` carrega o melhor estado.
- O `val_iou` gravado no `info.json` é o do melhor estado; o `test_iou` é medido depois da restauração.

### 14.3 Dropout e decaimento

- Seção 4.6 e tabela da seção 5; o decaimento do AdamW na seção 12.2.
- Com `ResACEUnet`, a receita dos autores usa `dropout = 0` e regulariza pela aumentação e pelo recorte.

---

## 15. O laço completo de treino

```python
trainer = Trainer(network, loss, epochs, scheduler, ema, transforms, use_amp, patience)

for epoch in range(1, epochs + 1):
    # ---------------- treino ----------------
    model.train(); iou.reset()
    loader.dataset.epoch += 1                                  # sorteio novo da aumentação
    for imgs, masks in trainLoader:                            # imgs (B,1,D,H,W) float32, masks (B,1,D,H,W) long
        optimizer.zero_grad()
        with autocast(enabled=amp):
            logits = model(imgs)                               # (B,1,D,H,W)
            loss   = criterion(model, logits, masks)           # perda em float32 (+ compose da fault_seg_net)
        scaler.scale(loss).backward()                          # retropropagação
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()                # AdamW (pulado se houve inf/NaN)
        if cosine: scheduler.step()                            # agenda por passo
        if ema:    ema.update(model)
        iou.update(sigmoid(logits) > 0.5, masks)               # IoU acumulado da época
    # --------------- validação ---------------
    evalModel = ema.model if ema else model
    evalModel.eval()
    with no_grad(), autocast(enabled=amp):
        for imgs, masks in valLoader:
            logits = transforms.infer(evalModel, imgs)         # direto, ou janela deslizante com crop
            ...                                                # val_loss médio por lote e val_iou acumulado
    if plateau: scheduler.step(val_loss)
    progress.json ← {epoch, train_loss, val_loss, train_iou, val_iou, lr}
    if early_stopping.ready(evalModel, val_iou): break

early_stopping.restore_best(model)                             # melhor estado → modelo vivo
test_loss, test_iou = trainer.evaluate(testLoader)
save(model.state_dict(), optimizer.state_dict(), history)
```

---

## 16. Métricas durante o treino

- **IoU:** `BinaryJaccardIndex` do torchmetrics, que acumula a matriz de confusão de **todos os voxels de todos os
  lotes** da época e calcula $TP/(TP+FP+FN)$ no fim (micro-média). Um volume com muita falha pesa mais que um com pouca.
- **`train_iou`** é medido com a rede em modo de treino (dropout ligado) e com os pesos mudando ao longo da época: não é
  diretamente comparável ao `val_iou` (o dropout tende a baixá-lo; a aumentação, a mudá-lo de dificuldade).
- **`train_loss`/`val_loss`:** média das perdas por lote (média de médias).
- **Validação com recorte:** o IoU continua sendo do **tile inteiro**, predito por janela deslizante.
- **`test_iou`:** a mesma conta da validação, no teste, depois de restaurar o melhor estado.
- O `3 - Predict` refaz o teste somando TP, FP, FN e TN e mostra também precisão, recall e F1; e, para modelos de
  recorte, o protocolo da Tabela 2 da ResACEUnet (média por recorte 96³).

---

## 17. Inferência

### 17.1 No dataset (validação, teste e `3 - Predict`)

- Sem `crop`: o tile inteiro vai direto na rede.
- Com `crop`: `Transforms.infer` usa o `sliding_window_inference` do MONAI na janela do treino, com sobreposição 0,5,
  pesos **gaussianos** (máximo no centro da janela) e `batch_size` janelas por passada. A combinação é a média
  ponderada dos **logits** das janelas, e a sigmoide vem depois.

### 17.2 No Marlim (`Marlim/1 - Predict.ipynb`)

- A `SlidingWindow` própria anda na janela do treino sobre cada tile 128³, com sobreposição 0,25 e peso de
  **Hanning** separável (mínimo $10^{-3}$); combina as **probabilidades** ($\sigma$ de cada janela):
  $$P(\mathbf p) = \frac{\sum_k w_k(\mathbf p)\,\sigma\big(z_k(\mathbf p)\big)}{\sum_k w_k(\mathbf p)}.$$
- Janela igual ao tile → uma passada só. O resultado é a probabilidade contínua, gravada em float32; a binarização (0,5)
  e a extração de sticks ficam no `2 - Analysis`.
- O `model.img_size` salvo no `info.json` é sempre a janela de treino, e é ele que os dois preditores usam.

---

## 18. Determinismo e sementes

- `seed_everything(42 + trial)`: `random`, `PYTHONHASHSEED`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed`;
  `cudnn.deterministic = True` e `cudnn.benchmark = False`. A semente fixa a inicialização dos pesos, a ordem dos lotes
  e os sorteios de dropout.
- **Exceção:** com `macnn`, `cudnn.deterministic = False` — a convolução 3D dilatada não tem algoritmo determinístico que
  caiba na memória (pediria 6,75 GiB a mais).
- A aumentação usa sementes próprias `(42 + trial, época, índice)`, independentes de quantos workers há.
- O split é fixo (`random_state = 42`) e não depende do trial.
- `torch.use_deterministic_algorithms` não é ligado: algumas operações da CUDA com soma atômica no backward (poolings e
  interpolações 3D) podem variar na última casa. Na prática os históricos por época se repetem até a terceira casa.
- Por isso `n_trials` existe: a variância entre rodadas idênticas (~0,03 de IoU na `dbrnet`) só aparece mudando a
  semente.

---

## 19. Memória e custo

Medidos na Quadro P6000 (24 GB):

| rede | configuração | memória (pico) | tempo por passo |
|---|---|---|---|
| `dbrnet` | $f$ 32, 128³, lote 2, float32 | ~15 GiB | 2,9 s |
| `resaceunet_zu` | $f$ 16, 128³, lote 2, float32 | ~7,5 GiB | 2,5 s |
| `resaceunet_zu` | $f$ 16, recorte 96³, lote 8, AMP | ~7,0 GiB | 2,8 s |
| `resaceunet_wu` | $f$ 16, 128³, lote 2 | 7,4 GiB | 4,6 s |
| `macnn` | $f$ 32, 128³, lote 2, *checkpoint* na atenção | ~17,7 GiB | ~5 s |
| `fault_seg_net` | $f$ 32, 128³, lote 2 | ~22,3 GiB (projetado; dropout desligado para caber) | — |
| `nru_net` | $f$ 32, 128³, lote 2, *checkpoint* | 13,6 GiB | 4,6 s |
| `fault_edge_former` | $f$ 9, 128³, lote 2, *checkpoint* | ~7 GiB | 3,3 s |

- Uma época da `dbrnet` (100 passos) leva ~5 min; 100 épocas, ~8 h.
- O que ocupa memória são as ativações guardadas para o backward (seção 10.3), proporcionais a voxels × canais × lote.
  Os parâmetros (40 M ≈ 160 MB, mais o dobro para os momentos do Adam) são pouco perto disso.

---

## 20. Como ler um treino

- **`train.png`:** perda e IoU de treino e validação por época.
  - `train_iou` ≈ `val_iou` com a validação ainda subindo no fim → **subajuste** (falta passo, lr ou capacidade). Foi o
    diagnóstico da primeira reprodução da ResACEUnet (IoU 0,717 no próprio treino, 0,687 na validação, lr de pico
    1e-4).
  - `train_iou` bem acima e `val_iou` parado ou caindo → **sobreajuste** (memorização; ver seção 6.4).
  - Degrau no `lr` (no `progress.json` e no histórico) = corte do `plateau`; costuma vir seguido de uma melhora.
- **`val_iou` × `test_iou`:** com 10 volumes cada, diferenças de até ~0,03–0,05 entre os dois são ruído de amostra; o
  `val_iou` gravado é otimista porque foi ele que escolheu a época.
- **Perda inicial fora do esperado** (seção 8) indica problema de dado (normalização trocada, máscara vazia) antes de
  qualquer época.
- **Comparações:** diferença menor que ~0,03 de IoU entre configurações não é resultado; use `n_trials` e o
  `summary()`/`VariationAnalysis` do `2 - Compare`. E o IoU sintético não substitui a avaliação no Marlim
  (`1 - Metodologia.md`, seção 15).

---

## 21. Estado atual dos treinos

Em `Model/Backup` (07/10/2026):

| modelo | rede | dataset | treino | val IoU | test IoU |
|---|---|---|---|---|---|
| `model_1` | `resaceunet_zu` $f$ 16 | `dataset_zu` (padronizado) | receita dos autores: recorte 96³ + aumentação, `dice_ce`, lote 8, cosseno com pico 1e-4, AMP, 200 épocas | 0,686 | 0,692 |
| `model_2` | `dbrnet` $f$ 32 | `dataset_wu` (min-max) | 128³, `dice_focal`, lr 1e-3, `plateau`, lote 2, dropout 0,1, sem aumentação | 0,788 | 0,758 |
| `model_3` | `resaceunet_zu` $f$ 16 | `dataset_wu` (min-max) | igual ao `model_2` (mesmo treino, só muda a rede) | 0,688 | 0,656 |

- No momento da escrita, o `Task/index.py` está rodando a rodada 1 do `task.json` (a mesma do `model_1` com pico de lr
  1e-3); o `progress.json` mostra a época corrente.
- Referências históricas (em `Documents/Backups`): `dbrnet` no `dataset_wu` chegou a 0,778 de test IoU em 100 épocas
  com a mesma configuração do `model_2`; o artigo da ResACEUnet reporta 0,764 na própria validação dos autores, que
  também escolheu a época.
