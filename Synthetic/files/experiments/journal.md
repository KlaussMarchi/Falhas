# Diário — gerador sintético que generaliza para o bloco real

Roteiro: `Synthetic/PROMPT.md` (19/10/2026). Este diário é a memória da tarefa: depois de uma compactação de contexto ou
numa sessão nova, ler daqui para baixo antes de agir. Os números de máquina ficam em `results.jsonl`, os logs em `logs/`.

## Estado atual (atualizar a cada experimento)

- **Fase (21/10 16:10):** decisões só na receita completa (as sondas Q20/L40 não ordenam como ela). Avaliação nos 4 patches
  anotados (pedido do usuário: todos viram material de calibração; o número honesto sai do leave-one-patch-out, `lopo.py`).
  Meta de calibração medida: marcação nativa de 1,0–1,5% da seção viva (curva fina do F1, seção "Âncoras nos 4 patches").
- **GPU:** fila b treina o `FULL_C1sd_t0` (época ~45/100, fim ~22h); a fila a avalia as sondas no 1400/2600 e pausa
  enquanto um processo meu de geração usa a placa. Na fila, depois do FULL_C1sd: `eval_teste_FULL_C1sd_t0` (liberado
  sozinho pelo `release.py` do scratchpad quando o treino acabar) e `train_FULL_D1_t0` (em espera até o dataset sair). As Q20
  restantes e as P20 seguem em espera.
- **CMA-ES do perfil de falhas:** `fault_ga.py 74 0.039` (PID 1043367, `logs/fault_ga_74.log`, memória
  `Synthetic/files/memory/falhas_74_*`); a primeira tentativa (`..._v1_visivel_por_stick`) usava a visibilidade por stick,
  inflada pelos sticks curtos do tile, e foi interrompida.
- **Geração:** `D1_74half` (padrões + `faultCount (2, 6)`) em `datasets/`, `logs/gen_D1_74half.log`.
- **Ao retomar:** `python status.py`; `python full.py <tags>` (resumo das rodadas completas); `python lopo.py <tags>`;
  `queue_done/`, `queue_running/` (job morto no meio: apagar o marcador); `logs/<job>.log`.

---

## 20/10/2026 — início

### Fase 0 (feita em 19/10, copiada do roteiro)

Movidos com `mv`:
- `Blocks/Marlim/Regions/` → `Synthetic/Regions/`
- `Blocks/Marlim/files/Generator.ipynb` → `Synthetic/Regions/Generator.ipynb`
- `Nature/` (raiz) → `Synthetic/Nature/`

Caminhos corrigidos: `Regions/Marlim` (`PATCH_DIR = '../../../Blocks/Marlim/files/patches'`), `Regions/Synthetic`
(`sys.path.append('../../..')`, imports `Synthetic.Nature...`, `DATASET = '../../../Dataset/dataset_regions'`),
`Generator` (`PATCH_METADATA`, `CACHE_DIR`, `MARLIM_DIR`, `DATASET_DIR`), os 460 `img_path` do
`Regions/Marlim/files/DataBase.csv`. Verificado em 19/10: Nature (CMA-ES e DE pequenos idênticos), tabela do
`MarlimBlock`, tabela de tiles do `Regions/Synthetic`, cache do `Generator`, caminhos do `DataBase.csv`, notebooks dos
blocos independentes do que mudou. Conferido hoje: nada recriado em `Blocks/Marlim/` (só `Analysis.ipynb` e `files/`).

### Estado encontrado

- GPU: a campanha do usuário (`Task/index.py`, PID 3633873) treina a `nra_sanet` no `dataset_74_wu`: época 27 de 100,
  ~8 min por época, 16 GB de VRAM, 98% de uso. A `fault_seg_net` já terminou (`Model/Backup/model_6`). Fim previsto em
  até ~10 h (ou antes, pelo early stopping).
- Disco: 318 GB livres.
- git: `main` limpo, HEAD `8413d0f`.

### Achados da leitura (a reportar ao usuário)

1. **O `mean` do CMA-ES sumiu na mudança para `Synthetic/Nature`.** O commit `fb79acd` (19/09) acrescentou o argumento
   `mean` ao `CMAES` (primeira nuvem num ponto conhecido); o `2fa4a74` ainda o tem em `Nature/Models/Genetic/index.py`, mas
   o `Synthetic/Nature/Models/Genetic/index.py` do commit da mudança (`f1b3873`) é a versão anterior, sem ele (diff de 6
   linhas). O `CLAUDE.md` e o `Regions/Synthetic/Analysis.ipynb` (`'mean': self.encode(self.reference)`) contam com o
   argumento: hoje o `Calibration.update()` daria `TypeError`. É uma regressão, não uma escolha (o `CLAUDE.md` de 19/10
   descreve o `mean`). Correção mínima = restaurar as 4 linhas do `fb79acd`. **Perguntar antes.**
2. **Pergunta da Fase 0:** `Synthetic/Regions/Synthetic/model/info.json` diz `"network": "unet3d_v2"` (duas vezes), nome
   antigo da `dbrnet`; o `ImageSimilarity.update()` falha. Os pesos carregam na `dbrnet` com `strict=True` (verificado em
   19/10). **Perguntar antes.**
3. **A referência `Artigo/model_5` foi treinada com `ema: false`**, não com a parametrização do usuário (`ema: true`).
   O resto bate (dbrnet, dataset_74_wu, normalize, lr 1e-3, dice_focal, lote 2, plateau, dropout 0,1, nf 32, 100 épocas).
   A comparação final precisa de uma rodada da referência com `ema: true` (a mesma do `task.json` do usuário).
4. **Bug pré-existente (não corrigir):** `Blocks/Iracema/Analysis.ipynb`, célula 14, `self.MODEL_THRESHOLD` onde o global
   `MODEL_THRESHOLD` é o que existe (a classe não tem o atributo). Só quebra o título do modo de máscara.
5. As âncoras da receita completa (`Documents/Backups/Marcia/model_1..6`) são de julho; os datasets delas: `dataset_74`
   (original/ de 21/07, o mesmo de hoje), `dataset_wu` (01/06, o mesmo), `dataset_74_wu` de 440 tiles (versão antiga;
   hoje são 330), `dataset_74_mar`/`dataset_wu_mar` (não existem mais). `Marcia/*` têm `ema` desligado e escalonamento
   percentil.

### Fase B — perfil sem rótulo dos patches de dev (20/10, CPU)

Código: `profile/ruler.py` (`TileRuler`: descritores por tile em 3 seções inline + 3 crossline, na escala da rede;
`BlockSampler`: os `.dat` pelo `patch_metadata.json`), `profile/run_profile.py`, `profile/gap.py`. Saída em
`profile/*.pkl` (1820 tiles reais de 1200+1300, 220 de cada âncora). 14 s por patch com 10 processos.

Composição dos 1820 tiles reais (o que a rede vê na inferência): água 16%, zona morta 26%, misto 19%, refletivo 38%.
Os 42% de água+morto+misto nunca aparecem no treino.

Gap table (p10/p50/p90; real = tiles com live > 0,5 por faixa de z; sintético = 220 tiles):

| descritor | real 450–700 | real 700–950 | real 950–1200 | dataset_74 | dataset_wu | marlim_opt |
|---|---|---|---|---|---|---|
| período (px) | 5,4/7,9/13,6 | 13,4/17,9/21,5 | 15,8/19,7/27,3 | 8,2/9,4/10,9 | 6,3/7,4/9,4 | 4,6/9,9/20,1 |
| ruído fino HHH/σ | 0,009/0,013/0,021 | 0,005/0,008/0,013 | 0,004/0,005/0,009 | 0,016/0,047/0,073 | 0,015/0,052/0,088 | 0,009/0,024/0,057 |
| disc5 (semblance<0,8) | 0,004/0,014/0,032 | 0,012/0,029/0,071 | 0,003/0,018/0,049 | 0,044/0,127/0,288 | 0,037/0,099/0,319 | 0,019/0,112/0,289 |
| disc17 | 0,06/0,15/0,28 | 0,11/0,19/0,36 | 0,04/0,12/0,31 | 0,22/0,40/0,62 | 0,25/0,46/0,78 | 0,11/0,34/0,61 |
| lag4 (correl. lateral) | 0,79/0,89/0,95 | 0,76/0,88/0,94 | 0,84/0,93/0,98 | 0,47/0,63/0,78 | 0,25/0,51/0,76 | 0,60/0,77/0,91 |
| saturação | 0/0,008/0,10 | 0/0,046/0,20 | 0/0,066/0,24 | ~0 | ~0 | 0/0,007/0,07 |
| tendência σ topo/base (log) | −0,68/0,13/0,98 | −1,02/−0,25/0,42 | −0,50/0,35/1,45 | −0,33/0,03/0,35 | −0,41/−0,06/0,22 | −0,91/−0,13/0,67 |
| mergulho p90 (°) | 5,6/8,4/14,2 | 13,5/23,5/36,9 | 13,7/19,8/27 | 15/25/35 | 12,6/24,3/35,9 | 8/18/43 |

- Cobertura (fração dos tiles reais vivos dentro do p2,5–p97,5 do sintético): período 0,13 (74) / 0,16 (wu);
  disc5 0,23/0,30; saturação 0; o `marlim_opt` cobre 0,73–0,97 de tudo e transferiu pior — **semelhança de
  estatística não é transferência** (mesma lição de setembro). Qualquer objetivo sem rótulo vai precisar de uma
  trava contra rótulo ruim (aprendibilidade).
- O sintético é **mais** descontínuo que o Marlim em todas as escalas (disc5 4–9×, disc17 2–3×); o real cresce mais
  com a abertura (disc17/disc5 ≈ 6,5× contra 3×). A diferença não é a quantidade de descontinuidade, é o **tipo**: no
  real ela é terminação/corpo/flanco organizado; no sintético, ruído fino + falhas.

### Fase B — auditoria visual dos falsos positivos (48 sticks, dev)

`profile/audit.py` → `profile/audit/audit6_*.png`; categorias em `profile/audit/audit_categorias.csv`. Amostra
aleatória (semente 0) de 48 dos 116 sticks da `dbrnet` de referência (`Artigo/model_5`, seções do 19/10) sem par no
especialista (cobertura < 0,2) nos patches 1200 e 1300. Um observador.

| categoria | n | fração (n) | fração do comprimento | z mediano |
|---|---|---|---|---|
| N: cruza refletores contínuos, sem rejeito visível | 11 | 0,23 | 0,24 | 833 |
| S: terminação estratigráfica (onlap, downlap, truncamento) | 8 | 0,17 | 0,15 | 905 |
| B: limite de corpo transparente/caótico (monte, diápiro) | 7 | 0,15 | 0,18 | 866 |
| D: hachura da zona fraca/morta logo abaixo da banda brilhante | 7 | 0,15 | 0,12 | 1130 |
| F: falha provável que o especialista não anotou | 6 | 0,13 | 0,16 | 852 |
| K: refletor íngreme (flanco de dobra/monte, camadas inclinadas) | 6 | 0,13 | 0,10 | 803 |
| ?: incerto | 3 | 0,06 | 0,05 | 1050 |

Leitura: **S + B + K = 44%** dos falsos positivos são estruturas sem rejeito que o sintético nunca mostra sem
falha (H9); **N = 23%** é a rede vendo falha onde não há nada (densidade/prior, H8, e textura, H5); **D = 15%** é a
hachura da zona fraca abaixo da banda brilhante, que passa pelo filtro de energia do extrator (H2); só **13–19%**
parecem falha real não anotada, então a precisão oficial subestima a verdadeira em ~15%.

### Fase B — falhas do especialista perdidas pela referência (dev, z 450–1200)

`profile/missed.py` → `profile/audit/missed_*.png`. 12 das 40 falhas anotadas em z 450–1200 com cobertura < 0,5
(o raso z < 450 perde 16 de 17, inalcançável pelo extrator). Quase todas são **sutis**: rejeito pequeno diante do
período (camadas finas, período longo), flexura com arrasto na banda brilhante, ou a extensão da falha para a zona
fraca. Em 8 das 12 a rede marca uma linha paralela a 15–40 px ou pedaços curtos: ela "vê" estrutura de falha, mas fora
do lugar e picotada. Isso aponta para H4 (falha sutil com rótulo só onde o rejeito é visível) no recall.

### Infraestrutura (20/10)

- `train.py`: cópia do `Model/1 - Model.ipynb` com 6 caminhos trocados (info.json, DataBase.csv, progress.json, Backup/ ×3)
  para `models/<run>/`; diff conferido: 12 linhas (6 − / 6 +), 41 células iguais no resto. Roda por papermill com
  `cwd=Model/`. Nunca escreve `Task/` nem `Model/Backup`.
- `make_dataset.py <tag>`: `SyntheticGenerator.dataset()` com as opções de `datasets/specs.json[tag]` + o Format
  'normalize' (min-max por volume, o mesmo `Normalization` do `Dataset/index.py`) → `datasets/<tag>/{images,masks,
  DataBase.csv,synthetic.json}`; apaga o `original/` depois (regerável). Gera na CPU (`CUDA_VISIBLE_DEVICES=""`):
  14,5 s/tile sozinho, ~20–25 s/tile com dois em paralelo (banda de memória).
- `harness/marlim.py` ganhou `liveMask`, `equalize` (p^k com k tal que p^k ≥ 0,5 marque 0,5/1/2% da seção viva),
  `sectionMetrics` (sticks no limiar nativo + defeitos + F1@0,5/1/2%) e `run` (avaliação + `results.jsonl`). O cache
  de slabs saiu do `/tmp` para `cache/` (o `/tmp` é apagado no boot). Conferido nas seções float16 do 19/10:
  `art5_dbrnet` 1200 → 0,528/0,485/0,201/0,291/78 sticks (o valor documentado para o float16); F1@1% = 0,405 (k = 16,2:
  a dbrnet é bimodal, igualar a máscara pede limiar ~0,96).
- `gpuqueue.py` (fila, `queue.json`) + `job.py` (train/eval): espera a placa ficar sem processo alheio > 2 GB (alheio =
  não descende de uma `gpuqueue.py`), marca `queue_running/` com O_EXCL (dá para rodar duas filas), grava
  `queue_done/` e `logs/<job>.log`. Rodando como PID 6584 (`logs/gpuqueue_a.log`), esperando a `nra_sanet` do usuário.

**Proxy escolhida para começar (P20):** a receita do usuário (dbrnet, lr 1e-3, dice_focal, lote 2, plateau, dropout 0,1,
nf 32, ema, normalize) com **20 épocas** (2000 passos). Avaliada pelo método rápido nos patches de dev. Validação: as
mesmas proxies nos datasets das âncoras de receita completa (74, wu, 74_wu, marlim_opt) contra as âncoras.

### Fase D — etapas novas no `SyntheticGenerator` (20/10, desligadas por padrão)

Ordem no `get()`: refletividade → dobra → cisalhamento → **campo de ganho** → **corpos** → falhas → **coluna (água,
zona morta, hachura)** → wavelet (**log**, **fase**) → ruído → `* rock` (zera a água) → corte → z-score.

| atributo (padrão) | o que faz | para quê (auditoria) |
|---|---|---|
| `bodyCount (0, 1)`, `bodyHeight`, `bodyWidth`, `bodyStretch`, `bodyLevel` | 0–n corpos por tile entre topo e base de uma gaussiana elíptica (monte, canal, lente), recheio caótico fraco; as camadas terminam no limite sem rejeito | B + S + K (44% dos FP) |
| `seafloorProb 0`, `seafloorDepth` | água (refletividade −2, depois imagem zerada) acima de uma superfície inclinada/ondulada; sem rótulo | 16% dos tiles reais são água |
| `deadZoneProb 0`, `deadZoneDepth`, `deadZoneLevel`, `deadZoneHatch (0, 0)`, `hatchDip`, `hatchPeriod` | refletividade fraca abaixo de uma superfície + hachura de migração (ruído filtrado no espectro em dois cones de normais ±θ); sem rótulo onde a refletividade cai abaixo da metade | D (15% dos FP), 26% dos tiles reais |
| `gainField 0`, `gainSigma` | log-ganho suave na refletividade antes do falhamento | tendência σ topo/base (real ±1, sint. ±0,35) e N |
| `waveletLog False` | frequência log-uniforme | H1 |
| `waveletPhase (0, 0)` | rotação de fase do Ricker por Hilbert | H10 |

Desligadas, nenhuma sorteia nada. Prova contra o `git show HEAD:Synthetic/index.py` (`checks/bitcheck.py`, CPU):
padrão, `dataset_74/synthetic.json`, sem falha, estresse lístrico (10–15 falhas, curva 1,0), `fast`; sementes 7 e 2026;
máscara igual, |Δ| da imagem e estado do `np.random` depois do tile. (resultado abaixo)

### Fase B — evidência de rejeito sem rótulo (tentativa, negativa)

`profile/offset.py`: ao longo de cada stick, traço composto de 12 traços de cada lado (levados até a linha pelo
mergulho local), correlação em deslocamento vertical ±14; "rejeito" = ρ > 0,6, |lag| ≥ 2 e ρ − ρ0 > 0,15. Fração dos
pontos com rejeito (mediana por grupo): especialista 0,375 (n=40); FP: F 0,50, S 0,47, K 0,42, D 0,33, N 0,33, B 0,29.
**Não separa** as falhas do especialista dos sósias (S e K têm tanto "rejeito" quanto elas). Descartado como objetivo
sem rótulo e como filtro. Subproduto: nas falhas do especialista o |lag| mediano é 2 px com período ~17 px, isto é
rejeito/período ~0,1–0,4; no sintético o rejeito no centro da falha é 15–32 px para período 8–11 px (1,3–4 períodos).
O Marlim tem falhas de rejeito pequeno diante do período (H4).

### Fase B — perfil do Iracema com a mesma régua (portabilidade)

`profile/run_iracema.py`: 1500 tiles sorteados (semente 0) do bloco inteiro, água = −vmin/(vmax−vmin) = 0,495994
(conferido: os tiles do topo são essa constante, desvio 2e-7). 48 s com 10 processos.

- Composição: água 20%, zona morta 32%, misto 21%, refletivo 27% (Marlim dev: 16/26/19/38%).
- Período por faixa (p10/p50/p90): z 450–750 4,6/10,6/15,9; 750–1000 15/16,5/18,3; 1000–1400 18,5/21,4/25,9;
  1400–1600 ~27 px — cresce com a profundidade como no Marlim.
- Mergulho p90 ~30° (o Marlim tem 8–24°), disc5 0,04–0,12, saturação até 22% no pacote dobrado (450–750).
- Conclusão: o Iracema é mais dobrado, mais saturado e mais descontínuo que o Marlim; bom teste de portabilidade.

### Desenho guardado para a 3ª rodada (H4: falha sutil), ainda não implementado

- `faultThrowPeriod (0, 0)`: rejeito máximo = sorteio × período da wavelet do tile (o Marlim tem rejeito ~0,1–0,4
  período; o padrão dá 1,3–4). Pede a frequência sorteada **antes** do falhamento quando ligado: `getFrequency()` no
  começo do `get()`, `applyWavelet(model, f)` usa a já sorteada; desligado, o sorteio fica onde está (bit a bit).
- `faultVisible 0`: rótulo só onde |rejeito| ≥ faultVisible × período (rotular rejeito invisível tornou a tarefa
  inaprendível em julho).
- `faultDrag (0, 0)`: arrasto — o deslocamento dos dois blocos cai como exp(−|d|/w) perto do plano; c ∈ [0, 1] é a
  fração do rejeito que vira flexura (c = 1 é flexura sem quebra). Mexe no bloco de baixo, então só ligado.
- Só entra se a 1ª/2ª rodada mostrar que o recall é o gargalo depois de a precisão subir.

**Resultado da prova (versão com corpos, coluna sem hachura, ganho, fase, log):** as 10 comparações (5 configurações ×
2 sementes) deram máscara idêntica, |Δ| da imagem = 0,00 (bit a bit, não só ≤ 1e-8) e o mesmo estado do `np.random`
depois do tile — **TUDO IGUAL** (`logs/bitcheck_1.log`). A hachura e a mudança dos padrões dos corpos (altura 15–120,
largura 8–50) vieram depois; a prova é refeita no fim da Fase D.

**Conferência visual (`checks/features.png`):** água plana + refletor forte do fundo do mar e zona morta com hachura
parecem os do Marlim; campo de ganho e fase funcionam; os corpos quase não aparecem com o ruído padrão (o recheio fraco
vira ruído). O que salta na figura é o **ruído**: os tiles reais são lisos, sem granulação; os sintéticos são todos
granulados. Medido num tile sem falha: ruído 0 dá `fine` 0,000–0,003 e disc5 0,000–0,001; o ruído padrão dá `fine`
0,033 (real 0,005–0,013) — e o disc5 alto do sintético (0,127) vem quase todo das **falhas** (sem falha cai para 0,005).
Período medido acompanha a frequência (f 40 → 17–19 px; f 85 → 9,4–9,8 px).

### Sonda de sensibilidade da referência (CPU, 20/10): onde a dbrnet marca falha sem haver falha

`checks/sensitivity.py` → `checks/sensitivity.csv`, `checks/sensitivity.png`. `Artigo/model_5` (dbrnet, 74_wu) em tiles
sintéticos **sem falha** com um só tipo de estrutura (3 sementes cada, entrada min-max), fração do volume marcada ≥ 0,5:

| caso (sem falha) | marcado | × camadas |
|---|---|---|
| camadas (padrão, sem falha) | 0,07% | 1 |
| corpos (2–3 por tile) | **2,02%** | 29 |
| água + zona morta + hachura | **3,07%** | 44 |
| campo de ganho 0,6 | 0,46% | 7 |
| período longo (f 33–55) | 0,15% | 2 |
| dobras íngremes (σ 6–20, ±40) | 0,05% | 1 |
| ruído baixo (0–0,15) | 0,02% | 0,3 |
| controle com falhas | 7,1% (rótulo 7,4%, recall por voxel 0,79) | — |

- **Confirma H9 e H2 de forma controlada**: a rede marca o limite dos corpos (as camadas terminando contra o corpo
  transparente) e o fundo do mar/zona morta — o mesmo que a auditoria achou no Marlim (B, S, D). Ela nunca viu essas
  estruturas sem rótulo, então as chama de falha.
- Variação de amplitude ao longo da camada (ganho) dispara 7× — candidato para parte do N.
- Dobra íngreme lisa **não** dispara: o K da auditoria vem com terminação/espessamento, não com o flanco liso (A5 deve
  render pouco).
- Ruído baixo e período longo não disparam.

Virou a **bateria de sósias** (`battery.py`, 8 casos × 4 sementes fixas em `battery/`), rodada pelo `job.py` em todo
modelo avaliado: mostra por tipo de estrutura o que cada dado ensinou a ignorar.

### Fase D — falhas sutis (H4) implementadas (20/10, desligadas por padrão)

- `faultThrowPeriod (0, 0)`: o rejeito máximo de cada falha passa a ser um sorteio × o período do Ricker do tile
  (1/(f·dt)), no lugar do `faultThrow`; o sorteio fica no mesmo ponto da sequência (um por falha).
- `faultVisible 0`: rótulo só onde |rejeito| > max(`faultThreshold`, `faultVisible` × período).
- `faultDrag (0, 0)` + `faultDragWidth (4, 16)`: fração c do rejeito vira flexura de largura w nos dois blocos
  (deslocamento 1 − c/2·e^(−d/w) no capa, c/2·e^(d/w) na lapa); o salto no plano é (1 − c) do rejeito; os dois blocos
  são reamostrados (`applyDrag`).
- Com qualquer um dos dois primeiros ligado, o `get()` sorteia a frequência **antes** da refletividade e o
  `applyWavelet(model, f)` usa a mesma; desligados, `getFrequency()` é chamado dentro do `applyWavelet`, no lugar de
  sempre. `applyFaulting(reflectivity, frequency=None)` e `applyWavelet(model, frequency=None)` continuam aceitando a
  chamada antiga (o notebook do usuário chama as etapas soltas).
- Prova bit a bit refeita com a versão completa: `logs/bitcheck_2.log` (resultado abaixo quando sair).

### Plano para a Fase E (rascunho de 20/10, a confirmar com os números)

- **Objetivo sem rótulo candidato E1** (régua de descritores, cobertura de Naeem): no espaço de 12 descritores os
  quatro âncoras cobrem só ~2% dos tiles reais vivos (teto real×real 93%); o A1 (1–3 falhas) aproxima disc5/disc17/
  steep90 do real mas a cobertura conjunta não sai do lugar (1,4%). Estrito demais em 12 dimensões; testar versões por
  descritor/por faixa na meta-validação.
- **Objetivo sem rótulo candidato E2** (TSTR sem rótulo): treinar a sonda no candidato, predizer no bloco sem rótulo e
  medir só o que o próprio bloco diz sem intérprete — máscara na água/zona morta/traço morto (negativo certo), máscara
  no intervalo vivo, fração da máscara que vira stick longo (forma de falha). Já sai de graça das seções salvas de cada
  sonda; a meta-validação diz se correlaciona com o F1 rotulado.
- **Custo:** a P20 custa ~1,8 h por avaliação — inviável num AG. Validar uma sonda barata **P8** (dbrnet nf 16,
  10 épocas, 100 tiles, ~10 min por avaliação) contra a P20 nos mesmos ~16 datasets (ρ de Spearman). Se passar, o AG
  roda ~60 avaliações (pop. 8–10 × 6–7 gerações) em ~10 h, com sementes comuns na geração.
- **Genoma:** só os fatores que as ablações mostrarem que importam (intervalos em centro + log da largura), caixa pela
  plausibilidade física, o resto medido do bloco (período, composição água/morto).

**Prova bit a bit da versão completa (corpos, coluna com hachura, ganho, fase, log, H4):** `logs/bitcheck_2.log` —
10/10 com máscara idêntica, |Δ| = 0,00 e o mesmo estado do `np.random`: **TUDO IGUAL**. Conferência visual das falhas
sutis em `checks/subtle.png`: rejeito em 0,2–1,5 período dá deslocamentos discretos como os do Marlim, o rótulo segue
o plano onde o rejeito é visível, e o arrasto dobra os refletores perto do plano.

### Fase D — traços mortos de borda (20/10, desligado por padrão)

- `deadTraceProb 0`, `deadTraceWidth (1, 8)`: depois do corte, numa posição lateral (eixo x ou y, posição 8–63) o lado de
  fora vira o espelho do lado de dentro (como o patch é completado por reflexão nos tiles de borda do Marlim: no `w000`
  o traço morto fica na posição 32), com uma faixa de amplitude zero de 2·largura − 1 traços no eixo e uma rampa de 0–3
  traços até o dado vivo; faixa e rampa sem rótulo, os rótulos do lado espelhado acompanham o espelho. Conferido num tile
  (faixa nas colunas 13–17 para largura 3). Entrou no `B2_column` (prob. 0,2) antes de ele começar a ser gerado.
- `get()` agora corta imagem e máscara antes do z-score e passa pelas duas juntas; desligado, devolve as mesmas.
  Prova bit a bit de novo em `logs/bitcheck_3.log`.

### PORTÃO B (20/10, ~14:30) — hipóteses ordenadas pela evidência

1. **H9 (sósias sem rejeito) → precisão.** Auditoria: B+S+K = 44% dos FP. Sonda: corpos sem falha disparam 2,0% do
   volume (29× camadas). Dobra lisa não dispara: o que pega é terminação/limite de corpo.
2. **H2 (água, zona morta, hachura, borda) → precisão e limpeza.** Auditoria: D = 15% dos FP; 29–39% da máscara cai na
   zona morta; borda_x0 0,30–0,40. Sonda: coluna dispara 3,1% (44×). 42% dos tiles reais são água/morto/misto.
3. **N (falha em refletor contínuo, 23%)**: candidatos H6 (ganho: 7× na sonda) e H8 (densidade 4–8× a do especialista;
   o disc5 sintético é quase todo das falhas).
4. **H1/H4 → recall.** Falhas perdidas são sutis (rejeito/período ~0,1–0,4 no Marlim × 1,3–4 no sintético); período
   17–25 px onde estão as falhas × 8–11 no treino.
5. **H5 (ruído fino 3–5× maior no sintético)**: o ruído baixo não dispara nada; efeito na transferência a medir (A3).
6. Descartados por evidência: dobra íngreme lisa (A5 deve render pouco); evidência de rejeito sem rótulo por
   correlação (não separa as falhas dos sósias).

### Desenho das rodadas de ablação (20/10)

- **Rodada 1 (sem código, base = padrão da classe/`dataset_74`):** A0 controle, A1 1–3 falhas, A2 0–2 falhas, A3 ruído
  0–0,15, A4 período log 4–25 px (3 regiões), A5 dobra íngreme.
- **Rodada 2 (código novo, mesma base):** B1 corpos 1–3, B1L corpos + ruído baixo, B2 coluna (água 0,25, zona morta
  0,4 + hachura, traço morto 0,2), B3 ganho 0,4… (0,5), B4 fase ±180°.
- **Candidato C0_v1 (tudo junto):** período log 33–208 (4–25 px), ruído 0–0,15, 1–4 falhas, 0–2 corpos, água 0,15,
  zona morta 0,3 + hachura, ganho 0,4, fase ±90°. Os fatores interagem (corpo some sob ruído alto; falha sutil some sob
  ruído), então a leitura principal vai ser **leave-one-out a partir do C0** (rodada 3, `C1_*`): sem corpos, sem
  coluna, sem ganho, período padrão, ruído padrão, densidade padrão, sem fase; e C0 + falhas sutis, C0 + bordas.
- **Sondas:** Q20 (nf 16, 20 épocas, 100 tiles fixos — os mesmos índices em todo dataset) para triar tudo; P20 (nf 32,
  20 épocas, 220 tiles) para confirmar; receita completa só na Fase F.
- Geração na CPU em cadeias (`logs/gen_*.log`); uma cadeia disparou antes da hora porque o `ps` devolveu dois PIDs
  (bash + python) — morta pelo PID, refeita esperando o PID do python.
- `faultLabelMin 0` (H8b): falha cujo rejeito máximo sorteado fica abaixo disso (voxels) desloca as camadas mas não
  entra no rótulo — o rótulo só com as falhas principais, como o intérprete. Sem sorteio novo (o limiar vira ∞ no
  rótulo daquela falha). Padrão 0 = todas rotuladas.
- `scaleup.py` (PID 36664, `logs/scaleup.log`): espera a campanha do usuário (PID 3633873) sair, dá 15 min para a
  fila `a` aquecer e abre as filas `b` e `c` enquanto o uso médio da placa ficar < 80% com > 8 GB livres (a Q20 com nf 16
  não deve encher a P6000). Só abre filas; não mata nada.

---

## 20/10/2026 (tarde) — sessão 2 (retomada pelo diário)

Estado encontrado às 14:50: a `nra_sanet` do usuário na época 41/100 (~7,8 min/época, ainda melhorando; fim previsto
~22h30); fila `a` (PID 6584) e `scaleup.py` (PID 36664) esperando; geração na CPU: A0 88%, A1 50%, C0_v1 86%, cadeias
B/C1 esperando. Carga da CPU ~17/20; a placa segue em ~100% com a geração rodando (a geração não atrasa o usuário).

**Respostas do usuário (20/10, ~15:10):**
1. `mean` do CMA-ES: **restaurar**. Feito: `Synthetic/Nature/Models/Genetic/index.py` ficou idêntico ao
   `Nature/Models/Genetic/index.py` do `2fa4a74` (as 4 linhas do `fb79acd`). Conferido: CMA-ES pequeno (4 variáveis,
   pop. 8, 12 gerações, semente 7) dá o mesmo resultado antes e depois sem `mean`; com `mean` a 1ª nuvem nasce nele
   (média de 400 indivíduos ≈ o ponto pedido, puxada só pela borda da caixa) e o `config()` grava o `mean`.
2. `unet3d_v2` → **`dbrnet`** no `Synthetic/Regions/Synthetic/model/info.json` (2 strings). Conferido: 74 tensores
   carregam com `strict=True`, `bot_dilated` presente.
3. GPU: **esperar a `nra_sanet` acabar**; nada na placa antes disso (nem inferência).
4. Bug do Iracema: **corrigir**. Feito: `self.MODEL_THRESHOLD` → `MODEL_THRESHOLD` na linha do `detail`
   (`Blocks/Iracema/Analysis.ipynb`, 1 linha, saídas intactas). O `Blocks/Marlim/Analysis.ipynb` não tem o bug.
- Informado (sem pergunta): a referência `Artigo/model_5` foi treinada com `ema: false`; a comparação final (Fase F)
  precisa da referência retreinada com `ema: true`, a parametrização do usuário.

### Fase D — refletores lenticulares e corpo com camadas (20/10, em desenvolvimento no scratchpad)

**Hipótese (H9b):** o que mais separa o Marlim do sintético nas figuras (`analise_19-10/synreal.png`,
`checks/view_A1_dens13.png`) é o refletor **lenticular**: trechos de 20–80 px que variam de brilho e terminam sem
rejeito. No sintético a refletividade é 1D (camada contínua no tile inteiro). Medido (perfil, tiles vivos 700–1200):
o real sem falha tem disc17 0,12–0,19 contra ~0,005 do sintético sem falha; a terminação espalhada é o "S" (17%) e parte
do "N" (23%) da auditoria. Se a rede nunca vê terminação sem rótulo, ela aprende "terminação = falha".

**Mudança (desligada por padrão):**
- `lensSigma (0, 0)`, `lensCut (0, 0,4)`, `lensEdge (0,3, 2)`, `lensThick 2`: `applyLenses` antes da dobra multiplica a
  refletividade 1D por `clamp((g − Φ⁻¹(corte)) / borda, 0, 1)`, com `g` um campo gaussiano normalizado de σ (lateral,
  lateral, `lensThick`). Como é antes da dobra, o padrão acompanha a camada; a dobra já aceita refletividade 3D (o
  `getCoefficients` opera no último eixo). Borda pequena = terminação seca; grande = brilho que varia e acunha.
- `bodyLayers 0`: chance de o corpo vir entulhado de camadas planas novas (canal com onlap) no lugar do recheio caótico;
  as camadas de dentro terminam na margem sem continuar do outro lado. Sorteia só se ligado.

**Esperado:** precisão sobe (menos S e N) sem perder recall, porque a falha continua sendo a única terminação alinhada
num plano **com** rejeito. Risco: a rede passar a ignorar terminações e perder falha sutil — ver o recall por faixa.

**Resultado da conferência (20/10, ~15:20):**
- Régua (mediana de 3–4 tiles, f 40–60, ruído 0–0,15, 1–2 falhas) contra os tiles vivos do Marlim 700–1200
  (`checks/lens_view.png`; disc5 / disc17 / lag4 / lag16 / steep90):
  sem lente 0,010 / 0,056 / 0,93 / 0,43 / 0,043 · lente σ 8–30 corte 0–0,4 0,031 / 0,148 / 0,82 / 0,23 / 0,10 ·
  **σ 8–50, corte 0–0,5, borda 0,3–2, espessura 3: 0,037 / 0,174 / 0,89 / 0,47 / 0,13** · corpo com camadas
  0,025 / 0,128 / 0,89 / 0,32 / 0,09 · **Marlim 0,024 / 0,165 / 0,90 / 0,41 / 0,094**. A lente leva a descontinuidade
  ao nível do Marlim sem derrubar a correlação lateral; os padrões da classe ficaram nesses valores (`lensSigma` (0, 0)
  continua desligando).
- Sonda de sensibilidade da referência (`Artigo/model_5`, `checks/sensitivity_new.csv/png`, fração marcada ≥ 0,5,
  3 sementes): camadas 0,07% · lentes suaves 0,21% (3×) · lentes de borda seca 0,45% (6×) · **corpos com camadas 1,73%
  (25×)** · lentes + período longo + ruído baixo 0,13%. A rede marca a margem do corpo com camadas (arco onde as camadas
  terminam) e pontos nas terminações secas — o B e o S da auditoria. A lente suave dispara pouco: a terminação
  espalhada sozinha não engana a referência; a terminação **alinhada numa superfície** (margem de corpo) engana.
- **Prova bit a bit:** contra o `git show HEAD` com os padrões (padrão, `dataset_74.json`, sem falha, estresse
  lístrico, `fast`; sementes 7 e 2026): 10/10 **TUDO IGUAL** (`logs/bitcheck_4.log`). Contra o arquivo de antes da
  mudança com as etapas da sessão anterior ligadas (configs do C0_v1, B2_column, C1_subtle, B1_bodies; sementes 7 e
  2026): 8/8 **TUDO IGUAL** (`logs/bitcheck_features_1.log`, script `checks/bitcheck_features.py`). Portado para o
  `Synthetic/index.py` às 15:20 (troca atômica; os processos de geração já abertos seguem com o módulo antigo, idêntico
  nos padrões).

**Novas ablações (4ª cadeia na CPU, PID 107435, `logs/gen_C0_v2.log`: C0_v2 → B5 → B6 → B7):**
- `C0_v2` = C0_v1 + lentes (σ 8–50, corte 0–0,5, borda 0,3–2, espessura 3) + metade dos corpos com camadas +
  período nominal 5–32 px (`waveletFreq` log 26–180). O período medido dos tiles vivos de dev é bimodal (27% em 4–6 px
  no raso, 32% em 16–22 px; p10/p50/p90 4,3/12,9/21,7); o C0_v1 dá 6,0/9,6/19,5 (falta cauda longa): 26–180 põe ~36%
  da massa acima de 16 px (real 41%).
- `B5_lens` (padrão + lentes), `B6_layered` (corpos 1–3, metade com camadas; comparar com o B1), `B7_sat` (ganho 1,
  jitter 0,4, corte ±2,4: saturação por tile de ~0 a ~28%, mediana ~1,6%, como o Marlim 0,8–6,6% / p90 10–24%).
- Fila: Q20 do C0_v2 (t0 logo depois do C0_v1 t0, t1 depois do C0_v1 t1), Q20 do B5/B6/B7 depois do B1L, P20 do C0_v2
  depois do P20 do C0_v1 (66 jobs). Bateria de sósias ganhou `lentes` (borda seca) e `corpos_camadas`.

### Fase E — preparação da meta-validação (20/10, tarde, CPU)

**Três famílias de objetivo sem rótulo candidatas** (todas computáveis num bloco sem interpretação):
- **E1 — cobertura do perfil** (`profile/objective.py` → `profile/objective_e1.csv`): os tiles reais vivos de dev
  (945) contra os tiles do dataset na régua (14 descritores): cobertura marginal (`marg`), precisão marginal,
  Naeem cobertura/densidade por grupo (escala, continuidade, textura, estrutura), −energia, −PAD (classificador
  real × sintético) e composição (água/morto/misto/refletivo). Primeiros números (marg / cob_grupos / composição):
  74 0,43/0,09/0,43 · wu 0,45/0,11/0,48 · A0 0,43/0,09/0,45 · A1 0,69/0,12/0,42 · **C0_v1 0,80/0,21/0,56** ·
  **marlim_opt 0,87/0,20/0,79**. O PAD satura (o classificador separa real de sintético com ~100% em todos): inútil.
  Aviso: o E1 põe o `marlim_opt` no topo, e o `marlim_opt` transferiu pior que o 74 em toda medida antiga — o mesmo
  modo de falha da régua de setembro.
- **Trava de aprendibilidade** (`profile/visibility.py` → `profile/visibility.csv`): AUC da descontinuidade (1 −
  semblance de 5 traços) no pixel rotulado contra o pixel vivo a > 6 px do rótulo, mediana por tile. 74 0,82 · wu 0,85
  · A0 0,81 · A1 0,83 · C0_v1 0,90 · **marlim_opt 0,70 (p10 0,53; 31% dos tiles sem rótulo na seção)**. É a explicação
  candidata para o `marlim_opt`: estatística parecida com a do Marlim, rótulo pouco visível.
- **E2 — estatísticas sem rótulo da predição da sonda no Marlim** (máscara, fragmentos, horizontal, água, borda, no
  mergulho) e a bateria de sósias: já saem de graça de cada avaliação.
- **E3 — falha injetada em tile real** (`injection.py`, `injection/*.npz`, 40 tiles vivos de 1200/1300 estratificados
  por faixa 300–1200, uma falha normal/inversa por tile com rejeito 0,3–1,2 × o período do tile, rótulo só onde
  vivo; `checks/injection_view.png`): recall da falha injetada no contexto real + marcação do tile real sem injeção
  (fundo) + mudança longe da falha. Entrou no `job.py` junto da bateria: todo modelo avaliado ganha a linha
  `metodo = injecao` no `results.jsonl`.
- `meta.py` reescrito: TSTR por dataset (média dos patches e sementes), ruído de semente, ρ de Spearman com IC por
  bootstrap e a fração dos pares além de 2σ na ordem certa, para todos os candidatos E1/E1−trava/E2/bateria.

**Testes de fumaça (CPU, sem tocar na placa):** `train.py` (o mesmo patch, 1 época, nf 8, 24 tiles do A0; só o
`get_device_name` pulado no rascunho) rodou de ponta a ponta em 5 min e gravou `model_1/{info.json, model.pth,
predictions, train.png}` + `progress.json` na pasta da rodada; `Task/info.json` (09:30) e `Model/Backup` intactos.
A avaliação rápida (`harness.run`, fast) na CPU montou o cache `cache/slab_1200.npy` (1,8 GB).

### Regras de decisão pré-registradas (20/10, 16:00, antes de qualquer resultado da placa)

**Portão A (esta noite):**
- Harness: `verify_art5` (1200, método do notebook) tem de dar exatamente 0,528/0,485/0,203/0,293/77 sticks. Se não
  der, a fila para e eu depuro antes de qualquer conclusão.
- Ruído: σ de semente da Q20 pelo A0 (t0/t1/t2), C0_v1 (t0/t1) e C0_v2 (t0/t1), em F1 e F1@1. Diferença menor que
  2σ entre dois datasets não é resultado.
- Sonda: a Q20 tem de reproduzir as diferenças grandes já conhecidas da receita completa (marlim_opt abaixo do 74; 74,
  wu e 74_wu juntos dentro do ruído) e a ordem das âncoras. Com 3–5 datasets de receita completa isso é fraco; vale como
  "não contradiz". Se contradisser, a P20 vira a sonda de triagem.

**Expectativas (o que eu espero ver, e por quê):**
- A0 ≈ 74 (mesma configuração, sementes diferentes) dentro do ruído.
- A1/A2 (menos falhas): máscara e mask_frac caem, precisão sobe, recall cai; F1 nativo sobe (o F1 premia marcar pouco),
  F1@1 igual ou um pouco melhor. A2 pode aprender pior (pouco positivo).
- A4 (período 4–25 px): cobertura do especialista sobe em z 700–1200 (o teste de escala de 28/09 prevê); F1 sobe.
- A3 (ruído baixo): indefinido; a sonda de sensibilidade não mostrou disparo por ruído.
- A5 (dobra íngreme): nada (a referência não dispara em dobra lisa).
- B1/B6 (corpos): `bat_corpos`/`bat_corpos_camadas` caem muito; no Marlim a precisão sobe um pouco.
- B2 (coluna): `bat_coluna`, `water_frac` e `borda_x0` caem; a hachura da zona fraca (D) some da máscara.
- B5 (lentes): `bat_lentes` cai; efeito pequeno no Marlim (a referência quase não dispara em lente suave).
- C0_v1/C0_v2 contra A0: F1@1 acima de 2σ e máscara menor. Se o C0 não passar do A0, os sósias não são o gargalo da
  sonda pequena e a leitura volta para escala e densidade.

**Portão C:** um fator importa se |ΔF1@1| > 2σ com as figuras e as medidas de defeito concordando, ou se a bateria e a
injeção mostram o efeito previsto sem piorar o F1. Fator neutro no Marlim e positivo na bateria fica no gerador (custo
zero) mas não entra na conta de ganho.

**M2 (objetivo sem rótulo):** passa se ρ ≥ 0,6 com o limite inferior do IC 95% > 0,2 e ≥ 80% dos pares além de 2σ na
ordem certa, em ≥ 8 datasets. Se nenhum passar, a calibração usa o TSTR com a sonda validada (dev), e o relatório diz que
a portabilidade sem rótulo ficou como hipótese.

**Fila reiniciada (20/10, 16:00):** às 15:52 um kernel do VS Code do usuário (PID 86447) passou a usar a placa (4,3 GB,
SM 20–80%) junto da `nra_sanet`. A regra antiga (qualquer processo alheio > 2 GB) deixaria a fila parada a noite toda
se esse kernel ficasse aberto ocioso; a nova espera o PID da campanha (pedido do usuário) e depois só pausa diante de uso
real da placa. O `scaleup.py` antigo desistia na primeira checagem com a placa cheia; o novo tenta de novo. Os dois
processos antigos (6584, 36664) eram desta tarefa e foram encerrados pelo PID.

### Plano para quando os resultados chegarem (escrito 20/10 16:10)

1. `logs/verify_harness.log`: conferir 0,528/0,485/0,203/0,293/77 (Portão A). Depois as 7 âncoras rápidas:
   `python summary.py anc_`.
2. Ruído: A0 t0/t1/t2 (Q20). `python meta.py Q20 f1@1` e `python meta.py Q20 f1` mostram σ e a tabela por dataset.
3. Sonda: ordem Q20 de 74/wu/74_wu/mopt contra a das âncoras de receita completa (Marcia/1 74, Artigo/2 e Marcia/2 wu,
   Artigo/5 e Marcia/3 74_wu, Marcia/5–6 misturas com marlim_opt).
4. Portão C: cada A/B/C contra o controle certo (A* e B* contra A0; C1_* contra C0_v1; C0_v2 contra C0_v1), com
   F1@1, recall/precisão, máscara, cobertura por faixa, a bateria (`bat_*`) e a injeção (`inj_*`). Figuras:
   `python compare_sections.py 1200 checks/cmp_1200.png Q20_A0_t0 Q20_C0v1_t0 Q20_C0v2_t0` (e 1300).
5. M2: `python meta.py Q20 f1@1` lista ρ, IC e pares para E1, E1−trava, E2, bateria e injeção. Atualizar
   `profile/objective_e1.csv` e `profile/visibility.csv` com todos os datasets antes (`cd profile && python
   objective.py && python visibility.py <todos>`).
6. Cuidado ao ler: a Q20 usa 100 tiles; um dataset diverso (C0) pode precisar de mais tile para aprender cada tipo —
   se o C0 perder na Q20 e ganhar na P20, o problema é a sonda, não o dado.

**Referência da Fase F adiantada (20/10 16:15):** `train_FULL_74wu_ema_t0` (dbrnet no `dataset_74_wu` com a parametrização
do usuário, **`ema: true`**, 100 épocas, trial 0) entrou no topo da fila com `slots: c`: só a fila c pega (quando o
`scaleup` abrir), as filas a/b seguem nas sondas. ~12–15 h dividindo a placa. É a "re-rodada da referência" que o 2.4
pede; aqui ela só é avaliada em dev (o teste fica para a Fase F).

**Protótipo da calibração (`calibration.py`, 20/10 16:30):** `Calibration` no molde da do `Regions/Synthetic` (genoma
centro + log da largura, caixa ±25% da faixa em volta da referência, `MIN_WIDTH` 1%, sementes comuns por geração pelo
hash da população, `NatureSelector('genetic')` com `mean` = referência codificada e `Memory.EVERY = 1`, pasta da
memória com hash da caixa/objetivo/semente, `generations` 0 quando o `best.json` já existe, escolha final em sementes
novas com ganho pareado contra a referência) + `E1Objective` plugável (régua × perfil real − trava de visibilidade).
Teste de mecânica na CPU (pop. 4, 2 gerações, 3 tiles por candidato): busca em 7 min, retomada sem nenhuma avaliação
nova, escolha final (referência 0,150 / melhor 0,324 / média da nuvem 0,335, ganho 0,18 ± 0,08 → média escolhida).
Pasta de teste apagada. O objetivo de verdade (E1, E3 ou TSTR) entra depois da meta-validação.
`meta.py` testado a seco com resultados falsos (`scratchpad`): TSTR por dataset, σ, ρ/IC/pares para E1, E1−trava,
E2, bateria e E3; o `objective.py` agora acrescenta ao CSV em vez de sobrescrever (o perfil do `marlim_opt` tinha
apagado as outras linhas).

### Ondulação curta dos refletores (20/10 17:00) — testada e NÃO adotada

Na figura do C0_v1 ao lado do Marlim (`checks/view_C0_v1.png`) a diferença mais visível é o relevo curto dos refletores
reais (3–5 px em 20–40 px). Testei, na cópia de desenvolvimento, uma ondulação por pacote de camadas na dobra
(`warpAmplitude`, campo gaussiano de σ lateral 8–30 e vertical 12 somado ao zTarget). Régua (com lentes, f 40–60,
1–2 falhas; disc5 / disc17 / lag16 / steep90): sem ondulação 0,034 / 0,147 / 0,20 / 0,11; ondulação 1–3 px 0,076 /
0,22 / 0,20 / 0,20; 2–6 px 0,051 / 0,195 / 0,21 / 0,16; Marlim 0,024 / 0,165 / 0,41 / 0,094. A ondulação **afasta** o
sintético do Marlim na descontinuidade e nos lineamentos íngremes e não mexe na correlação lateral; visualmente quase
não aparece (`checks/warp_view.png`). Não entrou no `Synthetic/index.py` (superfície pequena); fica no scratchpad.

### Notebook de perfil (`Synthetic/1 - Profile.ipynb`, 20/10 16:30) — primeira peça entregável da metodologia

14 células no estilo do usuário: `BLOCKS`/`BLOCK` (Marlim = só as fatias de dev 1200/1300; Iracema = 2000 tiles
sorteados), `BlockSampler` (nomes e posição do núcleo pelo `patch_metadata.json`, água = −vmin/(vmax − vmin) do próprio
metadata: 0,50263 no Marlim, 0,49599 no Iracema — nada fixo no código), `TileRuler` (a régua de `profile/ruler.py`
sem as colunas por eixo), perfil em `files/<bloco>/profile.pkl` + composição por faixa de 250 amostras, quantis dos
descritores por faixa, figura por profundidade e a lacuna dos datasets (mediana e cobertura de cada descritor). Rodou
de cima a baixo no Marlim (43 s com o perfil em cache) e, trocando só `BLOCK`, no Iracema (1 min 43 s; a cópia
executada ficou no scratchpad, o notebook entregue está no Marlim). Cobertura do período pelos tiles vivos: 74 0,13 /
wu 0,16 / 74_wu 0,18 / marlim_opt 0,81 no Marlim; 0,07 / 0,07 / 0,09 / 0,87 no Iracema.

### Rascunho do genoma da Fase E (20/10 16:40; fechar depois do Portão C e da M2)

**Medido no bloco e fixo (M4), por bloco, sem AG:**
- período: `waveletLog` + `waveletFreq` tal que o período medido do sintético cubra o p2,5–p97,5 dos tiles vivos
  (Marlim dev: ~4–28 px; Iracema: ~5–28 px). Ajuste por varredura curta f → período medido (no C0_v1, f 33–208 deu
  6,0/9,6/19,5 contra o nominal 4–25: o centróide puxa os curtos para cima).
- composição: `seafloorProb`/`deadZoneProb`/profundidades tais que a fração de tiles água/morto/misto do sintético
  bata com a do bloco (Marlim dev: água 16%, morto 26%, misto 19%; Iracema: 23/32/20%).
- saturação: `gain`/`gainJitter`/`clip` pelo `sat` real (Marlim vivo: mediana 3,6%, p90 ~20%) — se o B7 mostrar
  que importa.

**Buscado pelo AG (não se mede sem rótulo):** expressão da falha (`faultCount`, `faultThrowPeriod`,
`faultVisible`, `faultDrag`), textura (`noiseLevel`), descontinuidade estratigráfica (`lensSigma`, `lensCut`),
sósias (`bodyCount`, `bodyLayers`) e ganho (`gainField`) — só os que o Portão C mostrar que mexem; cada intervalo em
centro + log da largura, caixa pela plausibilidade física.

**Atenção ao tempo:** às 16:30 a `nra_sanet` estava na época 47 com ~17 min/época (o kernel do VS Code do usuário
divide a placa desde 15:50): sem ele seria ~8 min/época. A fila pode começar só de madrugada. A geração na CPU ficou
em 30–50 s/tile (4 cadeias + o kernel do usuário a 90% de CPU): as últimas C1 saem ~06:30. Quando a placa liberar,
mover o resto da geração para a placa (2–7 s/tile; o tile sai igual a menos de 1e-8, máscara idêntica).

**Fila com memória (20/10 16:50):** cada job agora pega a vaga só se couber na memória livre da placa menos 1,5 GB
(treino nf 32 ~10 GB, nf 16 ~5 GB, avaliação ~3,5 GB; `need_mb` no job manda), pegando o primeiro pendente que couber.
Três filas com P20/FULL nf 32 juntas estourariam os 24 GB. Fila `a` reiniciada (PID 159843).
- `gate_c.py [Q20|P20]`: cada ablação contra o seu controle (A*/B* e C0_v1 contra o A0; C1_* e C0_v2 contra o C0_v1),
  média dos patches de dev e das sementes, Δ com * quando passa de 2σ do ruído de semente da sonda, para as métricas de
  sticks, máscara, cobertura por faixa, bateria e injeção. Testado a seco com resultados falsos.

**Fase F, logística (20/10 17:00):** a `nra_sanet` (nf 16) que o usuário está treinando agora no `dataset_74_wu` com a
parametrização dele (`ema: true`) é a linha de base pronta para a **segunda arquitetura** do 2.4: na Fase F basta
treinar a mesma `nra_sanet` nf 16 no dataset final (~13 h) e comparar com o `model_7` dele. Conta de placa da Fase F:
dbrnet no dataset final × 2 sementes (~8 h cada) + nra_sanet (~13 h) + a referência `FULL_74wu_ema_t0` (já na fila c,
~12 h) + avaliação oficial (~30 min e 29 GB de máscaras por modelo, apagadas depois de medir).

### Onde a referência erra, por contexto (20/10 17:10; usa o rótulo só para entender) — `profile/context.py`

Em cada tile vivo de dev (z 450–1200, 723 tiles), a fração do núcleo 64×64 da inline anotada marcada pela `dbrnet` de
referência (seções do 19/10) a mais de 12 px de qualquer falha do especialista ("FP"), contra os descritores sem rótulo
do tile. ρ de Spearman com o FP: disc5 0,45 · steep90 0,43 · flat90 0,40 · disc17 0,40 · dipP90 0,32 · coerência −0,39 ·
lag16 −0,24; saturação, contraste, ruído fino e curtose ≈ 0. O FP no quartil mais descontínuo (disc17) é 2,6× o do
menos descontínuo (0,060 × 0,023). Por período, o FP é máximo em 9–20 px (0,044–0,050) e menor abaixo de 9 (0,029).
A máscara perto do especialista também cresce com a descontinuidade (ρ 0,32): **a descontinuidade sozinha não separa
falha de estratigrafia**, e no sintético quase toda descontinuidade é falha (disc5 do sintético sem falha ~0,005).
Reforça H9: o treino precisa de descontinuidade sem rótulo nos contextos descontínuos (lentes, corpos, terminações).
- 17:15: o kernel do VS Code do usuário treina algo (workers de DataLoader + placa). As cadeias de geração, os laços que
  as encadeiam e o vigia do perfil foram para nice 10 (todas as threads): a CPU fica para ele primeiro.
- 17:25: o conjunto de injeção (E3) foi refeito antes de qualquer avaliação: rejeito 0,25–1,25 período (p10/p50/p90
  0,30/0,63/1,18), **metade com arrasto** (c 0–0,6, a conta do `applyDrag`; rótulo só onde o salto (1 − c)·rejeito
  passa de max(1, 0,15 período)), razão e arrasto gravados no npz; o `run()` ganhou `inj_recall_sutil` (razão < 0,6) e
  `inj_recall_arrasto`. 40 tiles, 16 com arrasto, todos com rótulo.

## 21/10/2026 — madrugada: a placa liberou

- 00:29: a `nra_sanet` do usuário terminou (100 épocas, `model_7`, val_iou 0,802; `Task/task.json` tinha só ela e a
  `fault_seg_net`). 00:30: a fila `a` começou pelo `verify_harness` (Artigo/5 no 1200 pelo método do notebook).
- Geração na CPU: só faltam `C1_noGain` (~35 min, a ~10 s/tile agora que a CPU folgou) e `C1_stdDensity` (no fim); não
  vale mover para a placa.
- **`ProbeObjective` no `calibration.py`** (objetivo por sonda para o AG, caso a M2 reprove o E1): cada candidato vira um
  dataset de `nTiles` gerado na placa (`fast`), a Q20 treina nele pelo `train.py` (`models/calib_<semente>_<k>`) e a nota
  sai do `measure` — `'tstr'` (F1@1 médio de 1200/1300, só Marlim), `'injecao'` (recall injetado − peso × fundo, sem
  rótulo) ou uma função. O dataset é apagado depois do treino; a nota fica em `calib/<semente>_<k>.json` e a retomada a
  reaproveita (a semente de cada geração sai do hash da população, então a mesma geração cai nos mesmos arquivos).
- Teste de mecânica do `ProbeObjective` na placa (C0_v1, 24 tiles, nf 8, 1 época, medida por injeção): geração fast a
  ~1 s/tile dividindo a placa com o `verify_harness`, treino 0,6 min, injeção 30 s; a segunda chamada voltou do
  `calib/<semente>_<k>.json` em 0 s sem treinar. Arquivos do teste apagados. (Faltava o `harness/` no `sys.path` do
  `calibration.py`: corrigido.)

### PORTÃO A — harness (21/10 00:39)

- `verify_art5` (Artigo/5 no 1200, método do notebook, na placa): recall **0,528**, detecção **0,485**, precisão
  **0,203**, F1 **0,293**, **77** sticks — exatamente os números do notebook. Harness aprovado; 9 min por patch no
  método do notebook (máscara 4,13%, F1@1 0,405).

### Mapa frequência nominal → período medido pela régua (21/10 00:45, `profile/period_map.py` → `profile/period_map.csv`)

C0_v2 sem coluna, `fast`, 8 tiles por f (mediana do período medido): f 16 → 39,7 px · 20 → 35,0 · 26 → 26,7 · 33 → 22,6 ·
40 → 18,7 · 50 → 15,2 · 65 → 12,1 · 80 → 10,8 · 100 → 8,8 · 130 → 7,1 · 160 → 6,2 · 208 → 5,0 · 260 → 4,4 (≈ 790/f,
puxado para cima nos períodos curtos pelo ruído). O Marlim de dev, por faixa (tiles vivos, mediana): z < 450 4,2–4,7 px
(90–100% abaixo de 6 px), 450–700 7,4, 700–950 18,2, 950–1200 19,4. Para cobrir o Marlim, f vai de ~25 a ~280, e a
distribuição é bimodal (raso curto, fundo longo); o log-uniforme não reproduz a bimodalidade, uma mistura de regiões
por faixa de período (o `dataset()` já aceita várias regiões) reproduz. Decidir depois do A4/C1_stdPeriod.

### Parte medida (M4) — `getRegions` no `calibration.py` (21/10 01:00)

O perfil sem rótulo do bloco (`Synthetic/files/<bloco>/profile.pkl`) vira a mistura de regiões do `dataset()`: uma região
por faixa de 250 amostras com ≥ 5% dos tiles vivos, peso = fração dos tiles vivos, `waveletFreq` log entre as frequências
que dão o p90 e o p10 do período da faixa pelo mapa da régua. Sem rótulo e sem constante do bloco no código:
- Marlim (220 tiles): z250 63 tiles f 130–260 (4,2–7,1 px) · z500 56 f 44–172 (5,9–16,9) · z750 64 f 34–50 (15–22) ·
  z1000 37 f 25–47 (16–28).
- Iracema: z250 23 f 222–259 · z500 86 f 48–161 · z750 62 f 42–50 · z1000 49 f 30–40.
- O período ~4 px do raso está no piso da régua (centróide do espectro branco ≈ 4 px): lá qualquer f ≥ ~200 serve.
- Composição (água/morto/misto/vivo por faixa) medida nos dois blocos: Marlim 14/28/15/43%, Iracema 19/34/17/29%.
  `seafloorProb`/`deadZoneProb` → composição pela régua ainda precisa do ajuste rápido (não é 1:1).
- 01:10: **`dataset_regions` entra na M2** (o roteiro pede os 5 âncoras): ele não tem `DataBase.csv` e os `images/` dele
  (03/10) não são o min-max dos `original/` (provavelmente percentil, o padrão da época). Cópia privada em `normalize`
  a partir de `Dataset/dataset_regions/original/faulted` → `datasets/dataset_regions_n` (220 tiles, máscara 5,3%; a pasta
  do usuário não foi tocada). `train_Q20_regions_t0` entrou na fila logo depois do `mopt` (68 jobs).
- Disco: 173 GB livres (o `Model/Backup` do usuário tem 92 GB de predições; `experiments/` 60 GB, 56 GB de datasets).
- 01:15: `profile/objective.py` rodado nos 28 perfis (o `context_art5.pkl` entrou no filtro `NOT_SYN`). E1 `marg`
  (cobertura marginal dos tiles reais vivos): C1_noBodies 0,84 · C1_subtle 0,83 · C1_edges 0,81 · C0_v1 0,80 ·
  C1_noPhase 0,79 · C1_noColumn 0,78 · C0_v2 0,76 · A2 0,76 · C1_stdPeriod 0,70 · A1 0,69 · C1_stdNoise 0,67 ·
  C1_stdDensity 0,59 · B2 0,52 · A3 0,50 · A4 0,50 · B7 0,49 · A5 0,46 · wu 0,45 · B4 0,45 · A0 0,43 · 74 0,43 ·
  regions 0,43 · B3 0,41 · B1L 0,39 · B6 0,39 · B5 0,37 · B1 0,34; **marlim_opt 0,87 no topo**. A M2 diz se isso vale
  alguma coisa. (A `energia` explode no B2/C1_edges por causa dos traços mortos: não usar sem robustez.)
- Âncoras rápidas (dev, média 1200/1300): **Artigo/2 (wu)** F1 0,259 · F1@1 0,232 · máscara 5,1% · injeção recall 0,931 /
  fundo 0,060 · **Artigo/5 (74_wu)** F1 0,272 · F1@1 0,418 · máscara 3,9% · injeção 0,930 / 0,045. O F1 nativo quase
  empata e o F1@1 separa muito (o 74_wu põe a massa da máscara nas falhas certas); o fundo da injeção acompanha. O
  recall injetado satura (~0,93) nos modelos de receita completa. `summary.py` agora ignora as linhas `injecao`.
- **Validação da sonda com ≥ 4 datasets** (o roteiro pede): só 74, wu e 74_wu têm sonda e âncora de receita completa.
  Remontei as misturas das âncoras Marcia/5 (`dataset_74_mar`) e Marcia/6 (`dataset_wu_mar`) em cópia privada, como na
  época (tudo `percentile`, cada fonte com o seu p01/p99): `datasets/dataset_74_p` e `datasets/dataset_wu_p` (dos
  `original/` do 74 — `.npy` — e do wu — `.dat` transposto (0, 2, 1), conferido Δ = 0 contra o `Format` dele) e
  `datasets/mix_74_mar`, `datasets/mix_wu_mar` (só `DataBase.csv`: 220 + 220 do `marlim_opt`). Sondas
  `train_Q20_mix74mar_t0` e `train_Q20_mixwumar_t0` (`scaling: percentile`) na fila depois do `regions` (70 jobs).
  Obs.: o `dataset_74_wu` atual tem 330 tiles (110 do 74 + 220 do wu), o mesmo número do Artigo/5; o Marcia/3 foi treinado
  com 440.
- `profile/run_profile.py` lê o `DataBase.csv` quando existe (as misturas não têm `images/`). Perfil, visibilidade e E1
  feitos para `dataset_74_wu`, `dataset_74_p`, `dataset_wu_p`, `mix_74_mar`, `mix_wu_mar`, `dataset_regions_n`.
  Visibilidade (mediana): 74_wu 0,84 · 74_p 0,82 · wu_p 0,85 · misturas 0,79 (15% dos tiles sem rótulo na seção) ·
  regions_n 0,92. E1 marg: misturas 0,78 (puxadas pelo marlim_opt), 74_wu 0,48, regions_n 0,30.
- `ProbeObjective` agora em três etapas: gera os datasets da geração em série na placa, treina `workers` sondas ao mesmo
  tempo (cada uma num processo `train.py`, log em `models/calib_*/train.log`) e mede. Teste com 2 candidatos (12 tiles,
  nf 8, 1 época, `workers=2`): 118 s e retomada em 0 s; arquivos do teste apagados.
- `ProbeObjective` agora em três etapas: gera os datasets da geração em série na placa, treina `workers` sondas ao mesmo
  tempo (cada uma num processo `train.py`, log em `models/calib_*/train.log`) e mede. Teste com 2 candidatos (12 tiles,
  nf 8, 1 época, `workers=2`): 118 s e retomada em 0 s; arquivos do teste apagados.
- 01:02: o `scaleup` abriu a fila b (uso médio 46% numa fase de CPU da âncora). Duas avaliações juntas levam ~12+ min
  cada contra 7 min sozinhas: quase nenhum ganho de vazão nas avaliações; medir nos treinos Q20.

### M5 — capacidade (21/10 01:15; cobertura marginal por descritor dos tiles reais vivos de dev)

Fração dos tiles reais vivos dentro do p2,5–p97,5 do dataset, por descritor (74 · C0_v1 · C0_v2 · C1_subtle · marlim_opt):
period 0,13 · 0,70 · 0,74 · 0,69 · 0,83 | coherence 0,51 · 0,90 · 0,87 · 0,96 · 0,95 | disc5 0,23 · 0,70 · 0,58 · 0,93 ·
0,88 | disc17 0,48 · 0,84 · 0,83 · 0,94 · 0,90 | steep90 0,27 · 0,93 · 0,72 · 0,97 · 0,88 | fine 0,51 · 0,99 · 0,99 · 0,99 ·
0,79 | **sat 0,00 em todo dataset `normalize`** (o min-max por volume não satura; real: mediana 3,6%, p90 17%) — o B7
(ganho 1, jitter 0,4, corte ±2,4) cobre 0,87 e o marlim_opt (percentil) 0,84. Tudo o que a régua mede é alcançável pelo
gerador atual (o C1_subtle cobre ≥ 0,69 em todo descritor menos a saturação, que o B7 cobre). Se o B7 importar, a
saturação vira parâmetro medido (M4) pelo `sat` do bloco.
- Duas avaliações juntas: 12 min cada (7 sozinhas) — ~15% de ganho de vazão.
- Âncoras (receita completa, rápido, dev): art2 wu F1 0,259 / F1@1 0,232 · art5 74_wu 0,272 / 0,418 · mar1 74 0,289 /
  0,399 · mar2 wu 0,270 / 0,266 · mar3 74_wu(440) 0,260 / 0,338. O F1 nativo quase não separa os datasets (0,26–0,29);
  o F1@1 separa 74 (0,34–0,42) de wu (0,23–0,27). Pares do mesmo dataset (sementes/escala diferentes): wu Δ 0,011 /
  0,034; 74_wu Δ 0,012 / 0,080 → ruído da receita completa ~0,01 no F1 e 0,03–0,08 no F1@1.
- Misturas com o marlim_opt (receita completa): mar5 74_mar F1 0,254 / F1@1 0,262 · mar6 wu_mar 0,217 / 0,230 — marcam
  pouco (máscara 1,2–1,5%), recall 0,34–0,36; o fundo da injeção do mar5 é o menor (0,019): **o fundo da injeção mede a
  tendência a marcar, não a qualidade** (o tile real tem falhas reais). Ordem da receita completa no F1: mar1 74 >
  art5 74_wu ≈ mar2 wu > mar3 74_wu440 ≈ art2 wu ≈ mar5 74_mar > mar6 wu_mar; no F1@1: 74_wu/74 (0,34–0,42) > wu e
  misturas (0,23–0,27).
- 01:26: começam as Q20 (`train_Q20_74_t0` na fila b).
- 01:30: duas Q20 juntas (filas a e b), **8,7 GB cada** (não os 5 GB estimados), SM 42% + 56%, ~2 min/época cada → ~50 min
  por Q20 com duas juntas (~27 min efetivos). As 37 Q20 da triagem: ~15 h (até ~16h de 21/10). A FULL (fila c, 10 GB)
  não cabe junto de duas Q20 (a admissão por memória segura). Primeira leitura (sonda × receita completa, ruído, C0 × A0)
  às ~07h.
- **E2+ (`e2plus.py` → `profile/e2plus.csv`)**: medidas sem rótulo calculadas depois, das seções salvas (sem placa): a
  máscara igualada em 1% vira sticks pelo `FaultStickExtractor` (sem a interpretação) — nº, comprimento total, mediana,
  **rendimento** (fração da máscara a ≤ 3 px de um stick) e mergulho. Âncoras (rendimento): art2 wu 0,06 · mar2 wu 0,11 ·
  mar3 74_wu 0,17 · art5 74_wu 0,19 · mar1 74 0,26 · **misturas com marlim_opt 0,39**. Acompanha o F1@1 nos datasets
  puros, mas as misturas traçam sticks longos nos lugares errados: o mesmo modo de falha do E1. Entrou no `meta.py`.
  Os modelos e as seções de toda sonda ficam guardados: qualquer medida sem rótulo nova se testa depois, sem treinar.
- 01:35: **admissão por memória corrigida**: a Q20 (nf 16) usa 8,7 GB na placa, não os 5 GB estimados; com uma terceira
  fila, uma Q20 admitida durante a fase de avaliação de outra estouraria os 24 GB. Todo job do `queue.json` ganhou
  `need_mb` (Q20 9500, nf 32 15000, avaliação 3800) — as filas em execução relêem o `queue.json` a cada job — e o
  padrão do `gpuqueue.py` mudou igual (vale para filas novas).
- 02:11: **primeira Q20** (74, t0; 45 min com duas juntas): F1 0,254 (1200 0,282 / 1300 0,227), máscara 9,9% da seção
  viva (a receita completa marca 2,6%), val_iou 0,54. F1@1 0,000 / 0,159 e F1@2 0,229 / 0,269: com a predição densa da
  sonda curta, o 1% mais provável do 1200 (q99 = 0,972, k 24,7; não é saturação, só 0,03% acima de 0,999) não se organiza
  em sticks ≥ 140 px. O F1@1 fica errático na Q20; F1@2 e o F1 nativo parecem mais estáveis — a validação da sonda contra
  as âncoras escolhe a métrica. Injeção: recall 0,905, fundo 0,110 (o dobro das âncoras: marca mais).
- **`f1curve.py`** (→ `profile/f1curve.csv`, das seções salvas, sem placa): F1 de sticks com a máscara igualada em 1, 2,
  3, 4 e 5% da seção viva e a média `f1_curva`. Âncoras (f1@1 / f1@2 / f1@3 / f1_curva): mar1 74 0,40/0,35/0,30/0,320 ·
  art5 74_wu 0,42/0,33/0,29/0,318 · mar3 74_wu440 0,34/0,29/0,27/0,284 · mar2 wu 0,27/0,30/0,27/0,272 · art2 wu
  0,22/0,26/0,29/0,263 · mar5 74_mar 0,27/0,22/0,22/0,226 · mar6 wu_mar 0,23/0,22/0,21/0,222. A diferença entre datasets
  está no topo do ranking (1–2%); de 3% para cima tudo converge para 0,26–0,30. Q20_74: 0,08/0,26/0,28 (curva 0,232).
  `meta.py` aceita `f1@3`…`f1_curva` como métrica de TSTR.
- 02:13: Q20 wu: F1 0,204 (74: 0,254), f1@2 0,249 (74: 0,249), f1_curva 0,222 (74: 0,232); injeção 0,948 / 0,088. Na
  receita completa: 74 0,289 > wu 0,259–0,270 (F1), 0,320 > 0,263–0,272 (curva). Por enquanto a ordem bate.
- 02:20: `Synthetic/1 - Profile.ipynb` ganhou a seção **RESPOSTA DO GERADOR** (16 células): o mapa f → período medido pela
  mesma régua, em `Synthetic/files/period_map.csv` (do gerador, não do bloco). Reexecutado no Marlim por papermill
  (3 min): o mapa sai idêntico ao `profile/period_map.csv` (Δ = 0 nas 104 linhas comuns) mais f 320 → 4,05 px. É a
  entrada da parte medida (M4) do notebook de calibração.
- Rascunho do `Synthetic/2 - Calibration.ipynb` no scratchpad (`make_calibration_nb.py`): configuração por bloco,
  `MeasuredRegions` (a parte medida, idêntica ao `getRegions`; com f 320 no mapa a faixa rasa do Marlim vai a f 130–297).
  BASE/LIMITS, objetivo, busca e escolha final entram depois do Portão C e da M2.
- 02:24: rascunho completo do `2 - Calibration.ipynb` (15 células, sintaxe conferida; executado até o objetivo sem
  treinar): CONFIGURAÇÃO (BLOCKS, BLOCK, OBJECTIVE, METRIC, BASE, LIMITS) → PARTE MEDIDA (`MeasuredRegions`) → GENOMA
  (`Genome`, ida e volta conferida) → OBJETIVO (`ProbeObjective`, com o Format 'normalize' próprio, sem tocar no
  `Task/`) → BUSCA (`Calibration` com `NatureSelector('genetic')`, `mean` na BASE, memória com hash) → ESCOLHA FINAL
  (BASE × melhor × média da nuvem em sementes novas, ganho pareado contra o erro padrão) → `files/<bloco>/calibration.json`.
- Rascunho do `3 - Dataset.ipynb` (`make_dataset_nb.py`, 10 células, sintaxe conferida): lê `files/<bloco>/calibration.json`,
  gera `Dataset/<nome>/original/<região>/` + `synthetic.json`, formata com o `Normalization` (sem `Task/info.json`; nome
  do tile com a região), copia o `Format.ipynb` do `dataset_regions` sem saídas e mostra um tile por região.

### PORTÃO A — ruído da Q20 (21/10 03:00)

- **A0 t0 × t1** (mesmo dataset, mesmos 100 tiles, semente de treino 42 × 43): F1 de sticks **0,268 × 0,195**
  (1200 0,305 × 0,222; 1300 0,230 × 0,168), f1_curva 0,202 × 0,153 — σ ≈ 0,05 no F1, 5× o das âncoras de receita
  completa (pares ~0,01). O val_iou sintético é igual (0,592 × 0,588): a transferência varia por semente muito mais que o
  desempenho no domínio (subespecificação, D'Amour et al. 2020).
- **`smooth.py`** (→ `profile/smooth.csv`, das seções salvas): ranking dos pixels vivos pela probabilidade contra a
  interpretação — AUC e AP de "perto" (≤ 8 px do esqueleto do especialista) × "longe" (> 30 px), e precisão/cobertura/F
  de pixel (tol 12 px) com a máscara igualada em 1, 2 e 4%. **A0 t0 × t1: AUC 0,624 × 0,624, AP 0,321 × 0,321, pf@1
  0,191 × 0,177, pf@2 0,184 × 0,178**: o ruído de semente do F1 de sticks vem da extração (não linear), não do ranking.
- Mas a Q20 mal separa os datasets: pf@1 74 0,209 · wu 0,203 · A0 0,184 (o A0 é a configuração do 74 com outra semente
  do gerador: 0,025 de ruído de amostra com 100 tiles); AUC 74 0,637 · wu 0,643. Receita completa: pf@1 mar1 74 0,371 ·
  mar3 74_wu440 0,349 · art5 74_wu 0,328 · mar5 74_mar 0,307 · mar2 wu 0,278 · mar6 wu_mar 0,266 · art2 wu 0,240; AUC
  74 0,700 × wu 0,643–0,655. A sonda curta ranqueia pior (pp@1 0,10–0,12 × 0,15–0,25) e quase igual para todos.
- Leitura: a validação da sonda (5 datasets com âncora) sai às ~06h com as métricas suaves; se a Q20 não ordenar como a
  receita completa, a triagem precisa de uma sonda mais longa (o roteiro: "budget the full recipe").
- `proxy.py` (sonda × receita completa nos datasets com âncora): com 74 e wu, a receita completa separa tudo (F1 0,289 ×
  0,265; pf@1 0,371 × 0,259; AUC 0,700 × 0,649; AP 0,452 × 0,373) e a Q20 só separa no F1 de sticks (0,254 × 0,204,
  dentro do ruído de ±0,05); nas medidas suaves empata (pf@1 0,209 × 0,203; AUC 0,637 × 0,643). A Q20 (900 passos) ainda
  não aprendeu o que diferencia os datasets (a receita completa dá 10 000 passos).
- 03:03: **sonda longa L40** (nf 16, os 220 tiles, 40 épocas ≈ 4 000 passos, mesma receita) no 74 e no wu entrou na
  fila logo depois das Q20 do C0 (antes do A1): se ela reproduzir a separação 74 × wu nas medidas suaves, vira a sonda
  da triagem; senão, o roteiro manda orçar a receita completa.

### Primeira leitura dos candidatos C0 na Q20 (21/10 03:45)

| Q20 (dev) | F1 sticks | precisão | máscara | f1_curva | pp@1 | pf@1 | pf@2 | AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| A0 t0 / t1 | 0,268 / 0,195 | 0,21 / 0,16 | 11,0 / 8,7% | 0,202 / 0,153 | 0,111 / 0,102 | 0,191 / 0,177 | 0,184 / 0,178 | 0,624 / 0,624 | 0,321 / 0,321 |
| 74 t0 | 0,254 | 0,19 | 9,9% | 0,232 | 0,123 | 0,209 | 0,202 | 0,637 | 0,335 |
| wu t0 | 0,204 | 0,13 | 8,4% | 0,222 | 0,119 | 0,203 | 0,205 | 0,643 | 0,342 |
| **C0_v1 t0** | **0,296** | **0,26** | 2,5% | **0,300** | 0,160 | 0,254 | 0,243 | 0,603 | 0,338 |
| **C0_v2 t0** | 0,110 | 0,54 | **0,7%** | 0,230 | **0,189** | **0,271** | **0,263** | 0,602 | 0,350 |

- O gerador novo muda muito a sonda e na direção certa no topo do ranking: pf@1 +0,06–0,08 sobre o A0 (ruído de
  semente 0,014; de amostra ~0,02), precisão do 1% mais provável 0,16–0,19 × 0,10–0,12. A máscara no limiar 0,5 cai de
  ~10% para 2,5% (C0_v1) e 0,7% (C0_v2): o C0_v2 (lentes + corpos com camadas) fica tão conservador que o F1 nativo
  desaba (recall 0,06), embora ranqueie melhor que todos. A AUC cai (0,60 × 0,62–0,64): menos pixel perto de falha com
  probabilidade média.
- Falta saber se isso sobrevive ao treino longo (a Q20 não separou 74 × wu, que a receita completa separa).
- **Fila reordenada (03:50):** L40 74/wu (rodando) → Q20 C0_v1 t1 e C0_v2 t1 (réplicas) → **L40 C0_v1 e L40 A0** (o par
  chave na sonda longa) → Q20 dos 9 C1 (qual fator do C0 importa) → Q20 dos âncoras 74_wu/mopt/regions/misturas e A0 t2
  → Q20 A1–A5, B1–B7. As 18 P20 e as 7 âncoras pelo método do notebook ficam paradas (`hold`) até a decisão da sonda.
- **Onde cai o 2% mais provável da seção viva** (média 1200/1300; `checks/cmp_1200_top2_full.png`, e `TOP=0.02` no
  `compare_sections.py`): receita completa 74 (mar1) <450 6% · 450–700 18% · **700–950 45%** · 950–1200 23% · >1200 9%;
  art5 74_wu 7/15/39/22/17%; art2 wu 18/16/30/21/17%. Q20 A0/74/wu **espalham por igual** (16–25% em cada faixa, 20–25%
  abaixo de 1200, na zona fraca). Q20 C0_v1 13/20/**29**/23/16%, C0_v2 19/23/**31**/16/12%. O gerador novo puxa a
  confiança da sonda para o intervalo falhado e para fora da zona fraca, o que a receita completa do 74 faz sozinha com
  treino longo.
- `gate_c.py` agora junta as medidas suaves e o f1_curva e soma ao ruído de semente o **ruído de amostra do gerador**
  (|74 − A0| / √2: a mesma configuração com sementes de geração diferentes) — sem ele o σ da AUC/AP sai ~0 e todo Δ
  vira "significativo". σ realista na Q20: F1 sticks 0,054 · f1_curva 0,052 · **pf@1 0,020** · pf@2 0,016 · pp@1 0,013 ·
  AUC 0,009 · AP 0,010. **C0_v1 × A0: pf@1 +0,070* · pf@2 +0,062* · pp@1 +0,053* · f1_curva +0,122* · top2 em 700–950
  +0,067* · top2 abaixo de 1200 −0,093* · AUC −0,022*; F1 sticks +0,065 (dentro do ruído).** C0_v2 × C0_v1: só pp@1
  +0,030*. (`smooth.py` ganhou `top2_<faixa>`.)

### Sonda longa L40 (21/10 06:47) — nf 16, 220 tiles, 40 épocas (~4 000 passos), 183 min cada com duas juntas

| dev (1200/1300) | F1 sticks | f1@1 | f1_curva | máscara | pp@1 | pf@1 | AUC | AP | top2 700–950 / >1200 |
|---|---|---|---|---|---|---|---|---|---|
| L40 74 | 0,290 | 0,261 | 0,287 | 4,7% | 0,145 | 0,240 | 0,623 | 0,339 | 0,26 / 0,21 |
| L40 wu | 0,212 | 0,124 | 0,208 | 6,6% | 0,122 | 0,201 | 0,610 | 0,317 | 0,25 / 0,24 |
| Q20 74 / wu | 0,254 / 0,204 | 0,08 / 0,18 | 0,232 / 0,222 | 9,9 / 8,4% | 0,123 / 0,119 | 0,209 / 0,203 | 0,637 / 0,643 | 0,335 / 0,342 | |
| completa 74 / wu | 0,289 / 0,265 | 0,40 / 0,25 | 0,320 / 0,267 | 2,6 / 4,3% | 0,254 / 0,166 | 0,371 / 0,259 | 0,700 / 0,649 | 0,452 / 0,373 | 0,45 / 0,09 (74) |

- **A L40 separa 74 > wu como a receita completa** em todas as medidas do topo do ranking (pf@1 +0,039, pp@1 +0,023,
  f1_curva +0,079, F1 sticks +0,078); a Q20 não separava. A L40 tem ~1/3 do intervalo da receita completa no pf@1 e ainda
  espalha a confiança em profundidade (0,26 no intervalo falhado contra 0,45 da completa). A AUC não acompanha em nenhuma
  sonda curta (a AUC olha o ranking inteiro, não o topo).
- Com 2 datasets é uma indicação, não a validação (≥ 4). A L40 do par C0_v1 × A0 vem a seguir (depois das réplicas Q20
  dos C0, que começaram às 06:46).
- **Calibração do limiar (risco para o F1 oficial):** densidade de rótulo no treino × máscara predita no limiar 0,5 —
  74/A0 7,3% de rótulo → Q20 9,9–11% · L40 4,7% · completa 2,6%; wu 7,0% → completa 3,6–5,1%; misturas com marlim_opt
  (~5%) → completa 1,2–1,5%; **C0 2,2–2,3% → Q20 2,5% (v1) e 0,7% (v2)**. O treino longo encolhe a máscara (9,9 → 4,7 →
  2,6% no 74), então a receita completa num C0 deve marcar < 1%. Nas âncoras, o F1 nativo é máximo marcando ~2,5–4% e cai
  abaixo de ~1,5% (misturas 0,217–0,254). O limiar é fixo (regra 2): o dataset final precisa de densidade que deixe a
  receita completa marcando ~2,5–4% — medir num treino completo de um C0 cedo (FULL) e ajustar o `faultCount` se preciso.
- 07:00: `C1_stdDensity` e `C1_noColumn` passaram para antes da L40 C0_v1 × A0: o C1_stdDensity (C0 com a densidade
  padrão, ~7% de rótulo) diz se o ganho de ranking do C0 vem da densidade baixa — se não vier, o dataset final pode
  manter a densidade padrão (boa para o limiar) com os outros fatores.
- 07:33 **réplicas dos C0 (Q20 t1)**: C0_v1 F1 0,296 / 0,290 (duas sementes quase iguais), máscara 2,5 / 2,5%, pf@1
  0,254 / 0,235, pp@1 0,160 / 0,146, top2 em 700–950 0,289 / 0,294; C0_v2 F1 0,110 / 0,130 (máscara 0,7 / 1,3%), pf@1
  0,271 / 0,242, pp@1 0,189 / 0,163, AUC 0,602 / 0,633. Médias pf@1: A0 0,184 · 74 0,209 · wu 0,203 · **C0_v1 0,245 ·
  C0_v2 0,257** (σ realista 0,024 → C0 − A0 = +0,061 / +0,073, 2,6–3,1σ). O F1 de sticks do C0_v1 é o melhor e o mais
  estável das Q20 (0,29–0,30); o C0_v2 ranqueia um pouco melhor, mas marca pouco demais no limiar 0,5.
- Figura no limiar 0,5 (`checks/cmp_1200_native.png`, 1200, z 550–1250): a completa do 74 (mar1, máscara 2,1%) marca
  ao longo de várias falhas do especialista (x 1180–1300, 1350–1450, 1700–1800) e põe FP na parte acamadada de cima
  (z 600–800); a **Q20 A0 (9,3%) marca tudo** — pontilhado sem estrutura nas zonas de amplitude baixa (z 600–800 à
  esquerda, 950–1250 embaixo); a **Q20 C0_v1 (2,0%)** fica limpa, com trechos ao longo de falhas e fragmentos curtos
  espalhados; a **Q20 C0_v2 (1,0%)** quase só fragmentos em z 900–1000. O gerador novo tira a sonda do "marca toda
  descontinuidade"; o que falta é o treino longo organizar os trechos em planos.

### PORTÃO C — primeiras ablações leave-one-out (21/10 08:20, Q20, médias por dataset)

| Q20 | F1 sticks | máscara | f1_curva | pp@1 | pf@1 | AUC | top2 700–950 / >1200 |
|---|---|---|---|---|---|---|---|
| A0 (×2) | 0,231 | 9,9% | 0,177 | 0,107 | 0,184 | 0,624 | 0,22 / 0,25 |
| C0_v1 (×2) | 0,293 | 2,5% | 0,283 | 0,153 | 0,244 | 0,599 | 0,29 / 0,15 |
| C1_noColumn | 0,268 | 4,0% | 0,219 | 0,120 | **0,199** | 0,585 | 0,22 / **0,25** |
| C1_stdDensity | 0,223 | **5,8%** | 0,258 | 0,140 | 0,229 | 0,624 | 0,25 / 0,21 |

- **Coluna (H2) é o fator principal do C0 na Q20:** sem água/zona morta/hachura o pf@1 volta para perto do A0 (0,199;
  −0,045 ≈ 2σ) e a confiança volta para a zona fraca abaixo de 1200 (0,25, como o A0).
- **Densidade (H8) mexe na calibração, pouco no ranking:** com a densidade padrão (5–9 falhas, 6,1% de rótulo) o pf@1 fica
  dentro do ruído (0,229 × 0,244), a AUC até sobe (0,624), e a sonda marca 5,8% em vez de 2,5%. Como o treino longo
  encolhe a máscara ~4× (74: 9,9% → 2,6%), a densidade padrão deve deixar a receita completa marcando ~1,5–2,5%, perto do
  ótimo do F1 nativo; a densidade baixa do C0_v1 deve marcar < 1%.
- **Decisão (08:25): primeiro treino completo = `FULL_C1sd_t0`** (receita do usuário: dbrnet nf 32, 100 épocas, ema true,
  dataset `C1_stdDensity` = C0_v1 com a densidade padrão), logo depois do par L40 C0_v1 × A0 (que termina ~11:20). As 26
  Q20 restantes ficaram em `hold` para o FULL entrar sozinho (precisa de ~15 GB); medida a memória dele, as Q20 voltam se
  couberem ao lado. A fila b vai sair quando não achar job ("fila vazia"): relançar à mão depois.
- 08:23: **candidatos da parte medida (M4) gerando na CPU** (PIDs 761571/761572, nice 10, ~20 s/tile, `logs/gen_C2_*.log`):
  `C2_m4` = C0_v1 com a densidade padrão (como o C1_stdDensity) e o **período por faixa do Marlim** (`getRegions`: z250 63
  tiles f 130–297 · z500 56 f 44–172 · z750 64 f 34–50 · z1000 37 f 25–47) no lugar do log 33–208; `C2_m4_v2` = o mesmo
  com as lentes e os corpos com camadas do C0_v2. Semente 20261021.

### L40 do par-chave C0_v1 × A0 (21/10 11:21)

| dev | F1 sticks | máscara | f1@1 | f1_curva | pp@1 | pf@1 | pf@2 | AUC | top2 700–950 / >1200 |
|---|---|---|---|---|---|---|---|---|---|
| L40 A0 | 0,210 | **10,8%** | 0,160 | 0,235 | 0,141 | 0,231 | 0,214 | 0,620 | 0,27 / 0,24 |
| L40 C0_v1 | 0,234 | **1,1%** | 0,246 | 0,282 | 0,156 | 0,237 | 0,222 | 0,609 | 0,31 / 0,23 |
| L40 74 (mesma config do A0) | 0,290 | 4,7% | 0,261 | 0,287 | 0,145 | 0,240 | 0,224 | 0,623 | 0,26 / 0,21 |
| completa 74 (mar1) | 0,289 | 2,6% | 0,399 | 0,320 | 0,254 | 0,371 | 0,355 | 0,700 | 0,45 / 0,09 |

- **Com 2× mais passos, a vantagem do C0_v1 no ranking quase some**: pf@1 0,237 × 0,231 (Q20: 0,245 × 0,184); o A0
  sobe muito (0,184 → 0,231) e o C0_v1 não (0,245 → 0,237). Sobra +0,047 no f1_curva, +0,015 no pp@1 e +0,04 da confiança
  no intervalo falhado; a fração abaixo de 1200 fica igual (0,23). O efeito da coluna na Q20 era, em boa parte, aprendizado
  mais rápido — o treino longo do A0 chega lá sozinho.
- A calibração do limiar varia muito entre sementes/amostras na L40: A0 marca 10,8% e o 74 (mesma configuração) 4,7%.
- As sondas curtas (Q20 900 passos, L40 4 000) não chegam perto da receita completa (10 000 passos, pf@1 0,37) e a ordem
  entre datasets muda com o comprimento do treino. **Conclusão do Portão A: nenhuma sonda barata ordena como a receita
  completa de forma confiável; o roteiro manda orçar a receita completa** para as decisões, e as sondas ficam só para
  efeitos grandes e para a triagem do que vale levar à receita completa.
- 11:20: `FULL_C1sd_t0` começou sozinho na placa (17,5 GB — não cabe nada ao lado; ~8 h, até ~19:30). A fila b saiu
  ("fila vazia", como previsto).

## 21/10/2026 (tarde) — instrução do usuário: os 4 patches

**Usuário (21/10 ~13:50):** usar **todos os patches anotados** (1200, 1300, 1400, 2600) como material para calibrar o
bloco e maximizar o F1, "ir ajustando até ficar perfeito"; a anotação tem só as falhas gerais; o objetivo é achar o perfil
de falha do Marlim com um sintético compatível; código organizado, limpo, com poucos comentários, no estilo da `Model/`.

- **Protocolo novo:** toda medida e toda decisão usam os 4 patches (antes: dev 1200/1300, teste guardado). O teste
  guardado da M8 deixa de existir; para o número final não ficar otimista sem aviso, a escolha final entre candidatos é
  feita também em "deixa um patch fora" (escolhe em 3, mede no 4º, girando), o que não custa placa.
- 13:57: 21 reavaliações em 1400/2600 de todos os modelos prontos (7 âncoras, 10 Q20, 4 L40; `eval_teste_<tag>`, só a
  seção: `'medidas': False` no `job.py` pula a bateria e a injeção, que são do modelo) na fila a (relançada, PID 973418),
  ao lado do FULL (que roda na fila b). A do FULL_C1sd fica em `hold` até o treino acabar. Todo treino pendente avalia nos
  4 patches. As caches `slab_1400/2600` saem na primeira reavaliação.
- O usuário commitou às 10:53 (`3922057`, "melhoria adicionado imagens dos treinos e json"): entraram o
  `Synthetic/index.py` estendido, o `1 - Profile.ipynb`, o conserto do `Nature`, o `model/info.json` e a pasta
  `experiments/` inteira (scripts, logs, `queue.json`, marcadores). **A base das provas bit a bit do gerador passa a ser
  `8413d0f:Synthetic/index.py`** (o HEAD já tem as etapas novas). Como a `experiments/` agora é versionada, o código dela
  segue o estilo pedido (limpo, poucos comentários, como a `Model/`).

### Perfil das falhas do Marlim nos 4 patches (`profile/faults.py` → `profile/faults_<nome>.pkl`; 21/10 14:10)

Sticks do especialista (`fromRaster` da anotação, 105 nos 4 patches) e dos rótulos sintéticos (60 tiles × 3 seções inline,
`fromRaster` da máscara), medidos do mesmo jeito; rejeito = |deslocamento| mediano onde a correlação dos traços dos dois
lados é boa (`offset.evidence`), período local pelo centróide.

| mediana (p10–p90) | especialista | dataset_74 | C0_v1 | C1_subtle | C0_v2 |
|---|---|---|---|---|---|
| mergulho aparente (°) | 65 (53–76) | 61 (36–76) | 60 (32–74) | 63 (33–76) | 61 (38–75) |
| comprimento (px) | 274 (69–567) | 78 (tile 128) | 87 | 88 | 93 |
| rejeito (px) | 4 (1–7,6) | 8 (1,5–13) | 7 (2–14) | 2 (1–6) | 7,8 |
| período no local (px) | 13,5 (4,3–20) | 9,2 (8–11) | 11,8 (6,9–20) | 8,7 (5,8–18) | 12,5 (6,3–26) |
| **rejeito / período** | **0,27 (0,15–0,61)** | 0,90 (0,17–1,42) | 0,52 (0,14–1,41) | **0,19 (0,11–0,66)** | 0,52 |
| espaçamento (px) | 170 (31–481) | 32 | 51 | 39 | 36 |
| densidade (px de stick / px²) | 0,0028 (0,0019–0,0048 por patch) | 0,0176 | 0,0083 | 0,0079 | 0,0102 |

- Por faixa (especialista): z < 450 período 4,2, sticks curtos (69 px); 450–700 período 6,9, razão 0,37; 700–950 período
  17, razão 0,29 (43 dos 105 sticks); 950–1200 mergulho 75°, razão 0,23; > 1200 sticks de 630 px na zona fraca.
- **O 74 tem 3× o rejeito relativo do Marlim e 6× a densidade da anotação.** O C1_subtle (rejeito 0,2–1,5 período) chega
  perto na razão; nenhum sintético chega perto do espaçamento (o tile tem 128 px; a anotação põe ~0,75 falha por janela de
  128 px). Como a anotação traz só as falhas gerais (longas, espaçadas), "imitar o geólogo" puxa para rotular só as
  principais (H8b) — o critério do especialista é extensão/continuidade, não rejeito (ele anota rejeitos de 1–7 px).
- O mergulho aparente do sintético desce a 32–38° no p10 (direção sorteada em 0–2π); no Marlim o p10 é 53°. Numa rede 3D
  isso pesa pouco (ela vê o plano), mas é o H3 do roteiro.

### Âncoras nos 4 patches e o ponto de operação (21/10 15:15–16:00)

As 7 âncoras (receita completa) reavaliadas no 1400/2600 (`eval_teste_*`, `medidas: False`); médias dos 4 patches pelo
método rápido (o model_5 dá 0,292 aqui contra 0,300 pelo notebook):

| âncora (dataset) | F1 nativo | marca (0,5) | F1@1% | P@1 | R@1 | rótulo do dataset |
|---|---|---|---|---|---|---|
| art5 (74_wu) | 0,292 | 3,9% | **0,403** | 0,46 | 0,37 | 6,9% |
| mar1 (74) | **0,304** | 2,6% | 0,367 | 0,38 | 0,37 | 6,9% |
| mar3 (74_wu 440) | 0,271 | 3,6% | 0,359 | 0,44 | 0,31 | 6,9% |
| mar5 (74 + marlim_opt) | 0,302 | 1,2% | 0,318 | 0,33 | 0,34 | 5,3% |
| mar2 (wu) | 0,271 | 3,5% | 0,274 | 0,43 | 0,21 | 6,8% |
| art2 (wu) | 0,275 | 4,9% | 0,215 | 0,50 | 0,14 | 6,8% |
| mar6 (wu + marlim_opt) | 0,259 | 1,6% | 0,264 | 0,29 | 0,27 | 5,0% |

- **Curva fina do F1 contra a fração marcada** (igualada, 4 patches): art5 0,4% 0,20 · 0,6% 0,30 · 0,8% 0,38 · **1% 0,40**
  · 1,25% 0,37 · 1,5% 0,35 · 2% 0,34 · 3,9% (nativo) 0,29; mar1 tem o platô 0,8–1,25% em 0,365. **O ótimo é ~1% da seção
  viva, e marcar de menos custa mais que marcar de mais**: a meta de calibração é 1,0–1,5%.
- O model_5 tem a ordenação para passar da meta (0,40 em 1%); o que o derruba é marcar 3,9% no limiar 0,5: 87 sticks por
  patch contra 26 anotadas (1%: 27 sticks). A figura (scratchpad `fp_art5.png`, `fp_zoom_1200.png`) mostra o mecanismo: a
  máscara nativa é feita de fragmentos curtos espalhados pela zona deformada (z 700–1000), e o extrator (`STICK_GAP 59`)
  costura os fragmentos em sticks longos que cruzam as falhas do especialista. Excesso = 2,5× o esqueleto e 1,5× a
  espessura da máscara de 1%.
- Marca/rótulo na receita completa: 74 0,38, 74_wu 0,52–0,57, wu 0,51–0,72, misturas com marlim_opt 0,23–0,32. Na L40 o
  A0 (mesma configuração do 74, outra semente) marcou 10,8% contra 4,7% do 74: a marcação varia muito de rodada a rodada.
- Objetivo de imagem sem rótulo (E1) contra o F1@1 completo nas 5 âncoras de dataset distinto: ρ de −0,7 a +0,1 em todos os
  termos (as misturas com marlim_opt cobrem melhor o perfil e ordenam pior). **Casar só a imagem não prediz o F1** (5
  pontos; evidência fraca, mas na direção errada).
- **O perfil das falhas é mensurável sem rótulo.** Os sticks da máscara de 1% do model_5 (`faults.py modelo:anc_art5_74wu`)
  dão mergulho 65° (54–74), rejeito/período 0,31 (0,17–0,78), visível 0,43, espaçamento 292 px, densidade 0,0021; o
  especialista: 65° (53–76), 0,27 (0,15–0,61), 0,41, 170 px, 0,0028. Isso torna o perfil de falha portável (o Iracema não tem
  anotação): mede-se nos sticks confiáveis de um modelo de base.

### Plano a partir daqui (21/10 16:00)

1. **Calibração pela densidade (D1):** base do 74 (padrões) com `faultCount (2, 6)` → rótulo 3,9% (o padrão dá ~7,5–8%),
   para a marcação nativa cair de ~2,6% para ~1,3%. Isola a alavanca de calibração sobre a base que melhor ordena.
   `train_FULL_D1_t0` na fila logo depois do FULL_C1sd (em espera até o dataset sair).
2. **CMA-ES do perfil de falhas (`fault_ga.py`, objetivo `FaultObjective` em `calibration.py`):** genoma
   `faultThrowPeriod`, `faultVisible`, `faultDrag`, `faultCount`; alvo = perfil sem rótulo do model_5 (quantis do
   rejeito/período e da visibilidade, em unidades do interquartil) + fração rotulada na meta. Barato (12 tiles por
   candidato, ~20 s). Teste antes da busca (12 tiles): A0/74 razão 0,69, visível 0,53, rótulo 7,9%; C1_subtle 0,25/0,50/2,3%;
   alvo 0,31/0,43.
3. O FULL_C1sd (C0 com densidade padrão) diz às ~22h se a imagem da família C melhora a ordenação (F1@1) sobre o 74.
4. Depois: o vencedor do CMA-ES na base vencedora, na receita completa, com a densidade calibrada; escolha final por
   leave-one-patch-out (escolhe nos 3 patches, mede no 4º) para o número final não ser otimista.

### CMA-ES do perfil de falhas: o objetivo (21/10 16:00–16:40)

- `FaultObjective` (`calibration.py`): por candidato, 24 tiles (`fast`) × 3 seções inline; sticks do rótulo
  (`fromRaster`), medidos pela régua do `faults.py` (`offset.evidence`). Distância ao alvo = quantis (10/25/50/75/90) do log
  do rejeito/período em unidades do interquartil do alvo + |visível − alvo|/0,1 + |log(rótulo/meta)|/log 2.
- **Ponderação ponto a ponto.** A 1ª versão comparava medianas por stick: os sticks do tile são curtos (78 px contra 274) e
  perdem os pontos da borda (a janela da correlação precisa de 34 amostras acima e abaixo e 16 traços de lado), o que
  infla a visibilidade por stick (0,55–0,70 contra 0,45). Somando os pontos (`acertos`/`pontos`, colunas novas do
  `faults.measure`), a visibilidade do 74 é 0,46 e a do alvo 0,45 (especialista 0,42): **a visibilidade já casa; o que
  separa o sintético do Marlim é o rejeito/período** (74: 0,84–0,91 nos pontos com rejeito; alvo 0,31, especialista 0,28).
  Duas campanhas interrompidas e arquivadas em `memory/falhas_74_*_v1_*`, `_v2_*`.
- Alvo = perfil sem rótulo do model_5 (`faults_modelo_anc_art5_74wu.pkl`); o do especialista dá quase o mesmo, então o
  passo funciona sem anotação.
- Encadeado (`scratchpad/chain_d2.sh`): `fault_ga.py 74 0.039` → `make_spec.py D2_74` → dataset → `fault_ga.py C1 0.035` →
  `D2_C1`. Fila: `FULL_D2_74` (liberado sozinho quando o dataset sair) → `FULL_D2_C1` → `FULL_D1` → `FULL_74wu_ema`
  (controle do ema: o model_5 é `ema: false` e a minha receita completa é `ema: true`).

### Vencedor do CMA-ES na base do 74 e por que o D2_74 ficou em espera (21/10 17:00–17:30)

- `fault_ga.py 74 0.039` (memória `falhas_74_dcb967a9`): 12 gerações × 8, 24 tiles por candidato. Escolha em 4 sementes
  novas: referência −2,52 ± 0,60, melhor da busca −1,95, **média da busca −1,41 ± 0,16 (ganho 1,11 ± 0,26 → vencedora)**:
  `faultThrowPeriod (0,10, 1,58)`, `faultVisible 0,03`, `faultDrag (0,42, 0,82)`, `faultCount (3, 6)`. Em 48 tiles: razão
  mediana 0,215 (alvo 0,31; 74 0,90), visível 0,49 (alvo 0,45), rótulo 3,9%. Dataset `D2_74` gerado (220 tiles, 3,84%).
- **Mas as falhas do D2_74 são quase invisíveis** (`scratchpad/figs/d2_74_zoom.png`): o rejeito de ~2 px no período de ~9 px
  da base do 74, com o ruído dela, não deixa marca; no Marlim a mesma razão vem num período de ~17 px, num dado liso, e a
  falha aparece como lineamento escuro cortando os refletores. A razão é livre de escala; a visibilidade para a rede não é.
- **Escurecimento ao longo da falha** (envelope de Hilbert suavizado, relativo ao de ±40 px, mediana sobre os pontos dos
  sticks): Marlim 0,86 / 0,89 / 0,92 / 0,93 / 0,95 / 0,98 em 0/3/6/10/15/25 px; família C (C1_stdDensity) 0,85 / 0,92 /
  0,94 / 0,96 / 0,97 / 0,99; dataset_74 0,90 / 0,98 / 1,01 / 1,00; D1 0,93 / 0,97 / 1,00; D2_74 0,93 / 0,96 / 0,98 / 0,99.
  **A família C reproduz o perfil de escurecimento do Marlim; o 74 só tem a quebra estreita do rejeito grande.**
- Decisão: `FULL_D1` entra depois do FULL_C1sd (calibração pura, aposta segura); `FULL_D2_74` em espera; o `D2_C1` (CMA-ES
  na base da família C, encadeado) substitui o D1 se o FULL_C1sd confirmar a família C.

### CMA-ES na base da família C (21/10 17:16–18:14)

- `fault_ga.py C1 0.035` (memória `falhas_C1_97999371`): a base C casa melhor o perfil do Marlim (melhor por geração −0,36 a
  −1,35 contra −1,08 a −1,64 na base do 74), mas a busca não converge: o objetivo de 24 tiles tem desvio de 0,28–0,38 entre
  sementes, do tamanho das diferenças entre candidatos. Escolha em 4 sementes novas: referência −1,12 ± 0,38, melhor da
  busca −0,87 (ganho 0,25 ± 0,26, dentro do erro), média −1,20 → **fica a referência**: `faultThrowPeriod (0,2, 1,5)`,
  `faultVisible 0,15`, `faultDrag (0, 0,8)`, `faultCount (2, 6)` (as falhas do C1_subtle com a densidade (2, 6)).
- Para a próxima rodada da busca: mais tiles por candidato (48) ou o termo de visibilidade com peso menor; o ruído do
  objetivo é o limite, não o orçamento de gerações.
