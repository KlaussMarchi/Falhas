# Synthetic

Gera volumes sísmicos sintéticos com o rótulo das falhas, pelo fluxo do FaultSeg3D (Wu et al. 2019), e otimiza as faixas do gerador para que
uma rede treinada só em sintético acerte o dado do Wu e, no fim, os blocos reais (Marlim e F3). A régua da otimização são os próprios dados
do Wu (o FaultSeg3D): o dataset otimizado tem de superá-los com a mesma rede e o mesmo treino.

## Estrutura

```text
Synthetic/
├── Generator/index.py     SyntheticGenerator: o gerador (um tile 128³ + máscara por chamada)
├── Nature/                otimizadores (L-SHADE, CMA-ES, PSO, DE, L-SRTDE) com memória para retomar
├── Utils/index.py         runNotebook, BlockScore e SyntheticScore (os clDice do objetivo), getRuns, quiet
├── Analysis.ipynb         a otimização: gera, formata, treina, mede, repete
├── prompt                 o roteiro do estudo, com os ajustes de 26/11
└── files/                 tudo o que a busca grava (fora do git)
```

## 1. O gerador

Cada tile nasce num cubo de 256³ (128³ mais 64 de margem de cada lado, que absorve dobra e rejeito) calculado na GPU em float64, e é cortado
no fim. Cada parâmetro é uma faixa `(mínimo, máximo)` sorteada de novo a cada tile: é essa variedade que ensina a rede.

| etapa | o que faz | parâmetros (valor do `dataset_74`) |
|---|---|---|
| refletividade | uma coluna 1D de camadas com amplitude aleatória em [-1, 1] | `layerRange` (26, 233) camadas, `layerThickness` (1, 4) voxels |
| dobra | soma de gaussianas 2D que empurram as camadas para cima ou para baixo, mais forte em profundidade | `foldCount` (15, 48), `foldSigma` (17, 57), `foldAmplitude` (-35, 5), `foldDamping` 1,35, `foldBaseShift` |
| cisalhamento | inclina o pacote inteiro num plano: o mergulho regional | `shearOffset`, `shearGradient` (-0,1, 0,02) |
| falhas | cada falha é um plano; o bloco de um lado desliza ao longo do mergulho, e o rótulo é a faixa em volta do plano onde o rejeito aparece | `faultCount` (5, 10), `faultThrow` (15, 32) voxels, `faultDipAngle` (55, 81)°, `faultStrike` (0, 2π), `faultDecaySigma` (49, 59), `faultRoughness`, `faultCurveProb` (lístricas), `faultZoneWidth` 0,99, `faultThreshold` 0,77 |
| wavelet | convolui a refletividade com um Ricker em z: vira sísmica | `waveletFreq` (72, 99) Hz com `waveletDt` 0,0012 s, ou seja, período de 8 a 12 amostras |
| ruído | ruído gaussiano com grão próprio, proporcional ao desvio do sinal | `noiseLevel` (0,015, 0,618), `noiseSigma` |
| corte e z-score | corta a margem e padroniza o tile (média 0, desvio 1) | |

- A falha: o rejeito é máximo no centro do plano e cai como uma gaussiana (`faultDecaySigma`); o plano tem rugosidade e, às vezes, curva em
  profundidade. O rótulo marca os voxels a até `faultZoneWidth` do plano onde o rejeito passa de `faultThreshold`: falha pequena demais
  para se ver não vira rótulo. Falhas posteriores carregam o rótulo das anteriores junto com o bloco que deslocam.
- `faultCount` tem teto exclusivo: `(5, 10)` são de 5 a 9 falhas por tile.
- Há traços feitos para o Marlim (falhas menores sem rótulo, borrão lateral, água, zona morta, hachura, corpos, lentes, arrasto, traços
  mortos de borda), todos desligados por padrão e no `dataset_74`.
- `dataset(options)` grava uma pasta inteira: o tile `i` usa a semente `seed + i`, então o mesmo `seed` e os mesmos parâmetros dão os
  mesmos tiles. Os arquivos saem em `(x, z, y)`, o arranjo de todos os datasets do projeto. O `dataset_74` é `seed` 121958 e 220 tiles.
- `fast = True` sorteia os campos de ruído na GPU (~0,7 s por tile contra 2 a 7 s): mesma distribuição, tiles diferentes. É o modo da busca;
  o dataset final sai com `fast = False`.
- Os padrões da classe são exatamente o `Dataset/dataset_74/synthetic.json`.

## 2. Do gerador ao dataset

O gerador grava `Dataset/<nome>/original/{images,masks}/img_XXXX.npy`; o `Format.ipynb` da pasta escalona (aqui `percentile`: corte no
p01/p99 do conjunto, levado a [0, 1], o mesmo que os blocos reais recebem) e grava `images/`, `masks/` e o `DataBase.csv` que o
`Model/1 - Model.ipynb` lê.

## 3. A otimização (`Analysis.ipynb`)

```text
genoma (25 números) ──decode──> faixas do gerador
        │
        ├─ DatasetMaker:    200 tiles em Dataset/dataset_syntest + Format.ipynb
        ├─ Trainer:         proxy pelo Model/1 - Model.ipynb  ->  files/runs/model_N
        │                   IoU e AUC nas 20 validações do Wu pelo Model/3 - Predict.ipynb (só para comparar)
        ├─ SyntheticScore:  clDice nas 20 validações do Wu contra o rótulo       ┐
        └─ BlockScore:      clDice no F3_wu e no Marlim 1200/1300 (1400/2600 de prova) ┘  objetivo = soma ponderada (WEIGHTS)
        │
L-SHADE: compara os candidatos, gera a próxima geração e repete
```

- **Proxy:** um treino curto e igual para todo candidato, para que a diferença venha do dado: `unet_wu_z`, a receita do Wu (BCE balanceada,
  lr 1e-4 constante), 30 épocas, AMP e EMA. Leva ~15 min; o candidato inteiro, ~20 a 25 min.
- **Sorteios iguais:** todos os candidatos usam a mesma semente dos tiles e do treino. Dois candidatos só diferem pelos parâmetros.
- **Objetivo (27/11):** o clDice com tolerância de 3 amostras no orçamento igual (`Model/Metrics/ClDice`, a métrica geral do
  `Documents/REPORT.html`) em três lugares: as 20 validações do Wu contra o rótulo (`dataset_wu_val`, os volumes 200–219 do `dataset_wu`),
  o F3_wu contra o modelo do artigo do Wu e a inline anotada dos patches 1200 e 1300 do Marlim contra o especialista. Cada um entra com o peso
  do `WEIGHTS` (0,2 para o sintético, âncora com rótulo exato; 0,4 para o F3_wu; 0,4 para o Marlim), dividido pelo da **referência**. Os
  patches 1400 e 2600 ficam de prova, fora do objetivo.
- **Referência: os dados do Wu.** O mesmo proxy treinado nos 200 volumes de treino do Wu (`dataset_wu_treino`, os volumes 0–199), com duas
  sementes: vale 1, e o melhor candidato tem de passar dela por mais de 2σ. Nem ela nem os candidatos veem as 20 validações no treino. Os dois
  recortes do `dataset_wu` são pastas só com o `DataBase.csv`, reescritas pelo notebook a cada execução.
- **Tiles por candidato:** 200, o número da referência; os dois caem no split 180/10/10.
- **Por que não o IoU:** nos 16 modelos do `Model/Backup`, o IoU no sintético andou contra a nota nos blocos (Spearman −0,47). Ele continua
  gravado em `test.dataset_wu_val` (o `3 - Predict` com `FULL_DATASET = True`), só para comparar.
- O F3 inteiro fica de fora: a referência seria o mesmo modelo do Wu, manchado fora do recorte do artigo, e a predição custaria ~11 vezes a
  do F3_wu em cada candidato.
- **Os notebooks do `Model/` rodam em cópias** (`runNotebook`) que leem `files/info.json` e gravam em `files/runs/`: o `Task/info.json` e o
  `Model/Backup` não são tocados.

### O genoma

Cada faixa otimizada vira dois genes, **centro** e **largura**, então nunca sai uma faixa invertida; `limits` corta no intervalo físico.

| parâmetro | centro | largura | padrão do gerador |
|---|---|---|---|
| `layerRange` | 20 a 200 | 5 a 250 | (26, 233) |
| `layerThickness` | 1,5 a 6 | 1 a 6 | (1, 4) |
| `foldCount` | 2 a 50 | 1 a 40 | (15, 48) |
| `foldSigma` | 10 a 80 | 2 a 60 | (17, 57) |
| `foldDamping` | valor: 0 a 3 | | 1,35 |
| `shearGradient` | -0,15 a 0,15 | 0,01 a 0,3 | (-0,1, 0,02) |
| `faultCount` | 2 a 12 | 1 a 8 | (5, 10) |
| `faultThrow` | 6 a 40 | 2 a 30 | (15, 32) |
| `faultDipAngle` | 50 a 85 | 4 a 40 | (55, 81) |
| `waveletFreq` | 40 a 140 | 5 a 60 | (72, 99) |
| `noiseLevel` | 0,05 a 0,6 | 0,02 a 0,8 | (0,015, 0,618) |
| `foldAmplitude` | -40 a 20 | 5 a 60 | (-35, 5) |
| `faultZoneWidth` | valor: 0,5 a 2,0 | | 0,99 |
| `faultThreshold` | valor: 0,3 a 2,0 | | 0,77 |

- O espaço contém os padrões do `SyntheticGenerator` (o notebook decodifica o genoma deles e reproduz os valores).
- **Abertos em 27/11:** as camadas, a dobra e o mergulho regional (as seis primeiras linhas), para a busca não ficar presa a uma configuração
  antiga. São 25 genes; o orçamento de 1 200 avaliações ficou o mesmo e é a alavanca se a busca não convergir.
- **Fixos com certeza:** `waveletDt` e `waveletDuration` (o Ricker só depende de f·dt, e a frequência já varre o período), `foldBaseShift` e
  `shearOffset` (só deslocam o bloco na vertical), `faultStrike` (azimute livre de 0 a 2π) e os traços do Marlim (desligados).
- **Fixos nesta campanha:** a forma das falhas (rugosidade, alcance do rejeito, lístricas), nos padrões do gerador. Qualquer um pode entrar no
  `SPACE` depois.

### Seções do notebook

1. **Configuração**, **dados do Wu** e **parâmetros do gerador**: o proxy, os dois recortes do `dataset_wu`, o espaço de busca e a prova de
   que os padrões do gerador estão dentro dele.
2. **Dataset maker**, **treinador**, **nota do otimizador** e **avaliação de um candidato**: as peças acima, com uma prova em cada uma (um
   tile gerado, a régua dos três clDice com a `model_1` do `Model/Backup`).
3. **Calibração do proxy:** a referência, o proxy nos 200 volumes de treino do Wu com duas sementes. A média dela em cada lugar é o
   denominador do objetivo e o desvio do objetivo entre as duas é o ruído σ de uma avaliação. Diferença menor que 2σ entre candidatos é empate.
4. **Otimizador da natureza:** L-SHADE com população 8 e 150 gerações, ou seja, 1 200 avaliações, ~2,5 a 3 semanas de GPU (o notebook
   imprime a estimativa depois da calibração).
5. **Progresso:** lido do disco, roda a qualquer momento. O objetivo por candidato, o melhor até ali e o Marlim de ajuste contra o de prova.
6. **Confirmação dos melhores:** os `TOP` melhores (3 no teste, 5 na campanha) treinam de novo com outras duas sementes; vale a média do
   objetivo, ao lado do Marlim de prova. Depois, as **imagens do melhor**: um tile dele ao lado de um volume do Wu e as predições da referência
   e do melhor no F3_wu e nos patches 1200 e 2600.
7. **Dataset final:** o melhor confirmado vira `Dataset/dataset_grva` (`fast = False`) e os parâmetros vão para `files/best.json`.

### Retomada e falhas

- O L-SHADE grava o estado em `files/memory` a cada geração, e todo candidato completo volta do disco pelo `info.json`. Depois de um reboot,
  rodar o notebook de cima até o otimizador de novo: o que já foi feito passa direto, e a busca continua de onde parou.
- Antes de cada candidato novo, a busca confere se a placa tem `GPU_FREE` GiB livres (8 por padrão). Com outro treino na GPU, ela espera e
  avisa, em vez de começar e cair por falta de memória. Ainda assim, o certo é não pôr dois jobs de GPU juntos: o `files/rodar.sh <PID>` só
  começa quando o job daquele PID termina.
- Um candidato que falha (notebook com erro, memória da GPU) vira penalidade e vai para `files/failures.json`. Três falhas seguidas param a
  busca, porque aí o problema não é o candidato.
- Mudar o `PROXY`, o número de tiles ou as sementes muda a chave dos candidatos: nada antigo é reaproveitado por engano. Mudar o `SPACE`
  pede outra pasta em `MEMORY`.

## Arquivos

```text
files/
├── runs/model_N/      um candidato: info.json (parâmetros, test.dataset_wu, sintetico, blocos, minutos), model.pth (~18 MB), curvas, matriz
├── memory/            estado do L-SHADE (state.npz, best.json, history.json)
├── blocks/            referências dos clDice (o modelo do artigo do Wu no F3_wu, o especialista no Marlim, os esqueletos dos rótulos do dataset_wu)
├── logs/              a última saída de cada notebook (Format, Model, Predict)
├── info.json          o info.json do candidato em curso
├── failures.json      candidatos que falharam
└── best.json          os parâmetros do dataset final
```

## Como rodar

1. Deixe a GPU livre: a busca é o único job dela (nada de `Task/index.py` junto).
2. Abra o `Analysis.ipynb` e rode de cima para baixo, ou solto: `bash files/rodar.sh <PID a esperar, ou 0>` (papermill; notebook executado e log em
   `files/logs/`, figuras em `files/figures/`). A calibração leva ~1 h na primeira vez; o otimizador é a célula longa e para no `DEADLINE`.
   Rodar de novo continua a mesma campanha da `memory`, com os candidatos prontos voltando do disco.
3. Para acompanhar, rode a seção **Progresso** quando quiser.
4. No fim, a confirmação e o dataset final; depois é treinar pelo `Task` com `"dataset": "dataset_grva"` na receita completa.

## Cuidados

- O `Dataset/dataset_74` só é lido: dele sai apenas o `Format.ipynb`, copiado para as pastas novas. O `dataset_wu` também só é lido.
- Disco: o `Dataset/dataset_syntest` ocupa ~7 GB e é reescrito a cada candidato; cada `files/runs/model_N`, ~20 MB.
- Não dá para garantir que o dataset otimizado supere o do Wu antes de medir: o critério é a confirmação com 3 sementes e, depois, o treino
  completo comparado nos blocos (o `prompt` tem a regra).
