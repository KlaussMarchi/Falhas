# Searcher — busca dos parâmetros do gerador sintético

`Analysis.ipynb` procura a configuração do `Synthetic/index.py` que **maximiza o IoU no `dataset_wu`** de um
modelo treinado só com dado sintético. A ideia é operacional: o melhor sintético é aquele que ensina ao modelo
o que o `dataset_wu` ensinaria.

A otimização é feita pela **DE auto-adaptativa** (`Nature/Models/AdaptiveDE`, variante `lshade`), e cada
avaliação da função objetivo é o **pipeline inteiro do projeto**, rodado por papermill como o `Task/index.py`:

```
configuração do DE
  → Synthetic.dataset(220)              → Dataset/dataset_trial/original/
  → Dataset/dataset_trial/Format.ipynb  → images/, masks/, Dataset/DataBase.csv
  → Model/Analysis.ipynb                → Model/Backup/model_<n>/
  → Model/Predict.ipynb (dataset_wu)    → iou_wu   ← isso é a aptidão
```

---

## Como rodar

1. Abra `Searcher/Analysis.ipynb` **no kernel `torch-gpu`** (o `base` do anaconda não tem torch nem
   papermill). O `cwd` tem que ser `Searcher/` — é o padrão ao abrir o notebook de lá.
2. Rode as células **0 a 21** em ordem. São baratas (a mais lenta é a do alvo, ~30 s) e só definem as coisas.
3. **Meça o baseline antes de otimizar**: na célula 23, ponha `RUN_BASELINE = True` e rode. Isso avalia o
   preset atual (`Dataset/dataset_trial/synthetic.json`) e dá o número que a busca precisa superar. Leva de
   4 a 9 h. Volte para `False` depois.
4. Rode a célula 25 (constrói o `optimizer`, não gasta nada) e depois a **27**, que é a campanha. Pode deixar
   rodando dias e interromper quando quiser — veja *Memória* abaixo.
5. Para analisar depois, em kernel novo: rode 0-21, rode a **25**, **pule a 27** e vá para 29-33. O `plot()`
   reconstrói tudo a partir do `state.npz`, sem reavaliar nada.

### Mapa das células

| células | o que fazem | custo |
|---|---|---|
| 3, 5 | imports e constantes (todos os botões estão aqui) | instantâneo |
| 7, 9 | espaço de busca (37 variáveis) e conversão genoma ↔ gerador | instantâneo |
| 11 | mede o `dataset_wu` e confere se ele cabe no que o gerador produz | ~30 s |
| 13 | `Progress` — as barras das etapas | ~4 s (demo) |
| 15, 17, 19, 21 | `NotebookRunner`, geração, `Trial` e `Objective` | instantâneo |
| 23 | baseline com o preset atual (`RUN_BASELINE`) | 4-9 h |
| 25 | constrói o `optimizer` e mostra o orçamento do ciclo (`maxNfe`) | instantâneo |
| **27** | **a campanha** (`optimizer.update()`) | dias/semanas |
| 29, 30 | convergência do `Nature` e gráficos dos trials | segundos |
| 32, 33 | melhor configuração → `files/synthetic.json`, e histórico de campanhas | segundos |

### Acompanhando o progresso

Cada avaliação mostra uma barra por etapa, alimentada pelo que a etapa vai deixando em disco (as quatro rodam
fora deste notebook, então o `tqdm` interno delas não chega aqui):

| etapa | barra | fonte |
|---|---|---|
| geração | `gerando · n/220 vol` | `.npy` aparecendo em `dataset_trial/original/images/` |
| formatação | `formatando · n/220 vol` | `.npy` novos em `dataset_trial/images/` |
| treino | `treinando · n/100 ép` | campo `epoch` do `Model/progress.json` |
| predição | `predizendo · Ns` | cronômetro (o `Predict` só escreve no fim) |

Acima delas fica a barra da campanha (`DE LSHADE · n/1500 ev`), do próprio `Nature`. As barras de etapa somem
ao terminar, para não acumular milhares de linhas ao longo da campanha.

Duas coisas normais e que **não** são travamento: a barra de geração anda **em degraus** (os volumes saem em
20 processos paralelos e chegam em lotes, ~20 a cada ~42 s), e a de treino quase sempre termina **antes** do
total, porque quem encerra é o early stopping.

---

## Quantas iterações e quanto custa

Configuração atual: `POPULATION = 15`, `GENERATIONS = 100` → **1500 avaliações** por ciclo.

Custo por avaliação, medido nesta máquina (Quadro P6000, 20 CPUs):

| etapa | tempo |
|---|---|
| gerar 220 volumes | ≈9 min (42,5 s por volume, 20 processos) |
| `Format` | ≈3 min |
| **treino** | **3-8 h** (2,9 s/step → ≈4,8 min/época com 200 volumes) |
| `Predict` nos 220 volumes do `wu` | ≈3 min |

**Total: 4 a 9 h por avaliação.** As 1500 avaliações completas dariam ~10 meses de máquina — na prática você
roda por partes e para quando o ganho estabilizar. Se quiser mais avaliações no mesmo tempo, três botões, em
ordem de impacto:

- **`TRAIN_EPOCHS`** — teto de épocas do trial (o `Model/Analysis.ipynb` usa 100 fixo). Em 25, a avaliação cai
  para ~2 h. Todas as configurações competem com o mesmo orçamento, então a *ordem* entre elas continua justa;
  o que muda é o IoU absoluto.
- **`N_IMAGES`** — o tempo de época é proporcional. 60 volumes deixam a época em ~1,5 min.
- **`TRIAL_INFO['network']`** — `unet3d_v2` com `num_filters=16` treina bem mais rápido que o `resaceunet` 32.

Os três entram na chave do cache, então mudar de ideia no meio **não mistura medições incompatíveis** — só
custa reavaliar.

---

## Memória: como para e como continua

São duas camadas, e é a combinação que torna a campanha segura:

**1. `files/optimization/` — o `memory=` do `Nature`**
- `state.npz`: população, RNG e histórico completo. Gravado a cada **10 gerações** (`Memory.EVERY`).
- `best.json`: melhor global entre todas as campanhas.
- `history.json`: um registro por campanha (`run`, `at`, `score`, `improved`, `stopped`, `params`).

**2. `files/trials.json` — cache próprio, gravado a cada avaliação**

É o que cobre o buraco entre os saves da `Memory`. Como o `Randomizer` do `Nature` é determinístico
(`SEED = 42`), retomar reexecuta exatamente a mesma sequência de configurações — e tudo que já foi avaliado
volta do cache em milissegundos em vez de custar horas de novo. É também a tabela de análise da campanha.

### Parar e continuar

Interromper pelo kernel (⏹) **é seguro**. Sem `commit()` a campanha não fecha, então a próxima chamada de
`optimizer.update()` **continua o ciclo original**, no cronograma original. Rode a célula 27 quantas vezes
quiser.

Depois que um ciclo fecha sozinho, uma chamada nova vira **extensão**: outro ciclo de `GENERATIONS` a partir
dali, com a população reinflada até `POPULATION`.

### O que pode mudar sem perder a campanha

| mudança | efeito |
|---|---|
| `POPULATION` | pode. Numa extensão, o `AdaptiveDE` reinfla com indivíduos novos e mantém os sobreviventes como elite |
| `GENERATIONS` | pode. Junto com `POPULATION` define `maxNfe`, que é o horizonte do ciclo e a curva do LPSR — muda o *ritmo*, não o que já foi aprendido |
| `TRAIN_EPOCHS`, `N_IMAGES`, rede | pode. Muda a chave do cache: as avaliações antigas continuam guardadas, mas não são reaproveitadas |
| **`SEARCH_SPACE`** | **invalida a pasta.** A `Memory` levanta `ValueError` dizendo que ela pertence a outro problema — aponte `MEMORY_DIR` para uma pasta nova. O `trials.json` continua valendo, porque é indexado pelas options do gerador |

---

## O que a busca escreve (e apaga) no projeto

**Nenhum arquivo do projeto é editado.** Os notebooks são lidos, corrigidos *em memória* (só o `modelId` e o
`dataset` do `Predict`, e as épocas do `Analysis` se `TRAIN_EPOCHS` estiver definido) e executados a partir de
`files/runs/` com o `cwd` da pasta original.

O que ela grava é o que o pipeline normal já gravaria:

| caminho | o que acontece |
|---|---|
| `Task/info.json` | reescrito a cada avaliação com a config do trial |
| `Dataset/dataset_trial/original/` | **sobrescrito** a cada avaliação (220 volumes novos) |
| `Dataset/dataset_trial/images/`, `masks/` | sobrescritos pelo `Format` |
| `Dataset/DataBase.csv` | sobrescrito pelo `Format` |
| `Model/Backup/model_<n>/` | um por avaliação, com `iou_wu` e as options anotados no `info.json` |
| `Model/Backup/model_<n>/model.pth` | **apagado** após a medição (`KEEP_WEIGHTS = False`) |
| `Searcher/files/runs/<tag>/` | notebooks executados; apagados quando o trial dá certo (`KEEP_LOGS = False`), **preservados quando falha** |

---

## Pontos de atenção

1. **Nunca rode outro experimento em paralelo.** `Task/info.json`, `Dataset/DataBase.csv`,
   `Dataset/dataset_trial/` e a GPU são compartilhados — duas avaliações simultâneas corrompem uma à outra.
   É por isso que o DE roda com `workers=1`.
2. **`Dataset/dataset_trial/original/` é sobrescrito.** Se tiver algo lá que você queira, mova antes.
3. **Meça o baseline com `N_IMAGES = 220`.** Se baixar para um teste rápido, volte para 220 antes da campanha
   (o cache separa os dois, mas o baseline de 8 volumes não serve de comparação).
4. **Trial que falha vale IoU 0**, não penalidade — e os logs ficam em `files/runs/<tag>/` para você olhar.
   Um `-1e12` distorceria a escala dos gráficos.
5. **`POPULATION = 15` com 37 variáveis é pouco**: é menos indivíduos que dimensões, então os vetores-diferença
   geram um subespaço de posto ≤ 14. O crossover binomial evita o travamento total, mas se você conseguir
   baratear a avaliação, **subir a população rende mais que subir as gerações**.
6. **O `Format.ipynb` sempre "falha"** com `SystemExit` sob papermill (é o `sys.exit()` do modo sem tiles).
   Isso é esperado e tratado — o `Trial` valida o `DataBase.csv` em disco, não o código de saída.
7. **Espaço em disco**: cada trial deixa ~1-2 MB (`info.json`, `train.png`, `predictions/`). Os pesos são
   apagados. 1500 avaliações ≈ 3 GB.

---

## Saídas

| arquivo | conteúdo |
|---|---|
| `files/trials.json` | uma linha por avaliação: `iou`, `options`, `values`, `faults`, `spread`, `elapsed`, `error` |
| `files/optimization/best.json` | melhor configuração global e os params da campanha |
| `files/optimization/history.json` | um registro por campanha |
| `files/synthetic.json` | **a melhor configuração**, no formato do `gen.set(...)` |
| `files/convergence.png` | convergência + scatter por variável (correlação de cada parâmetro com o IoU) |

Para usar o resultado: `gen.set(json.load(open('Searcher/files/synthetic.json')))` no
`Synthetic/Generate.ipynb`, ou copie para o `synthetic.json` de um dataset novo.

---

## O que já foi verificado

- Pipeline completo rodado de ponta a ponta (versão reduzida, 8 volumes): gerou, formatou, treinou, prediu e
  devolveu `iou_wu = 0,3142` medido nos 220 volumes do `wu`. O `model.pth` foi apagado e o `info.json`
  anotado, como projetado.
- Cache, trial que falha, retomada em kernel novo e chave por contexto: testados com o código real.
- 18 configurações do espaço (16 aleatórias + os dois cantos extremos) geradas de verdade: **nenhuma quebrou o
  gerador, nenhuma produziu valor não-finito**.
- `Format` com os 220 volumes: pico de 5,4 GB de RAM (a máquina tem 125 GB).
- As seis estatísticas do `dataset_wu` **cabem dentro** do que o gerador consegue produzir (célula 11 refaz
  essa conferência ao vivo).

### Expectativa realista

O teto prático do `iou_wu` é **~0,75** — é o `test_iou` de modelos treinados no *próprio* `dataset_wu`
(`model_2` = 0,749; `model_8` = 0,751). Um modelo treinado em sintético dificilmente supera o treinado no
domínio real, então chegar perto de 0,70 já é um resultado forte.

Duas frentes onde o preset atual está claramente longe do alvo, e onde a busca tem mais a ganhar: o espectro
vertical (o preset gera pico em 0,070 contra 0,1156 do `wu` — dado mais "grosso") e o número de falhas
(`faultCount = [5, 10]` funde tudo num plano conexo só, enquanto o `wu` tem ~3 separados).
