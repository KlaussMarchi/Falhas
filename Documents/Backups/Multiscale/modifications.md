# Redes multiescala no Marlim — diagnóstico, correções e o que de fato muda o resultado

17–18/10/2026. Campanha analisada: esta pasta — model_5 `macnn`, model_6 `fault_seg_net`, model_7 `nru_net`, model_8
`fault_edge_former`, todas no `dataset_74` (percentil), lr 1e-4, dropout 0,05, EMA, `dice_focal`, sem aumentação, 128³.
Referências: model_2 `dbrnet` (mesma campanha), `Documents/Backups/Marcia/model_1` (`dbrnet`, `dataset_74`, lr 1e-3) e
`Documents/Backups/Artigo/model_5` (`dbrnet` na parametrização atual, `dataset_74_wu`). Figuras em `outputs/modificacoes/`.

## Resumo

1. **O IoU sintético não mede o Marlim.** Ele mede interpolação dentro do gerador. No Marlim todas as redes, multiescala
   ou não, ficam no mesmo patamar — F1 de sticks 0,28–0,33, precisão 0,20–0,26, máscara de 2–5% da seção contra ~0,8% da
   anotação — e a ordem entre elas muda quando se iguala a calibração. A arquitetura não é o gargalo; o gargalo é a
   distância entre o sintético e o dado real (o mesmo teto que o experimento controlado de 16/07 já tinha medido para
   quatro datasets calibrados no Marlim: F1 0,24–0,30).
2. **Multiescala ajuda na escala, mas não fecha a lacuna.** Com o sintético ampliado 2,5× (período de ~9 para ~22 px, o do
   Marlim profundo), o F1 tolerante cai de 0,97 para 0,76–0,78 na NRU, Fault-Seg-Net e FaultEdgeFormer, e para 0,68–0,69
   na `dbrnet` e no MACNN da mesma campanha. A escala é só uma das diferenças: o Marlim também tem saturação, contraste que
   varia 10× entre bandas, zona morta e descontinuidade estratigráfica que o gerador não tem.
3. **Dois problemas de implementação, corrigidos.** A atenção da Fault-Seg-Net estava portada para 3D com o eixo da
   feature errado, e faltava a conv do gargalo da figura 1: na mesma época e semente, a corrigida marca 39% menos, tem
   menos risco horizontal e metade da coluna na borda, com o mesmo F1. A NRA-SANet como o artigo descreve não treina neste
   pipeline (IoU 0), e a perda NR dela é nociva: entrou com normalização, atalho na atenção e a perda NR desligada.
4. **NRU, MACNN e FaultEdgeFormer estão fiéis aos artigos.** Testei recuperar o protocolo de treino dos artigos (espelhos
   e giros), ampliar a escala do treino (zoom, wavelet mais longa), trocar dataset e lr, e ajustar com as inlines
   anotadas do próprio Marlim. Nada disso melhora o Marlim além do ruído da métrica; zoom e wavelet pioram.
5. **O que melhora o Marlim de forma garantida, em todos os 4 patches: usar as redes juntas.** A média das probabilidades
   de MACNN + NRU + `dbrnet` leva o F1 de 0,327 (a melhor sozinha) para 0,372, a precisão de 0,255 para 0,298 e a máscara
   de 2,0% para 1,4% — melhor que a melhor rede em cada um dos 4 patches (seção 6). Os falsos positivos variam de rede
   para rede; as falhas reais se repetem.
6. **O que faz cada rede generalizar: auto-treino com dado real** (seção 7, `Model/4 - SelfTrain.ipynb`). O consenso de
   três redes rotula tiles reais **do Iracema** (outro bloco, sem anotação) e cada rede, já treinada no sintético, é
   ajustada neles junto com o próprio sintético. Pelo método do notebook, **as seis redes melhoram em cada um dos 4
   patches** (24 de 24): NRU 0,327 → 0,369, Fault-Seg-Net 0,270 → 0,338, NRA-SANet 0,262 → 0,367, MACNN 0,314 → 0,364,
   FaultEdgeFormer 0,243 → 0,313 e a `dbrnet` na parametrização do `task.json` 0,300 → 0,365 (seção 7.3). A máscara cai de 2–7% para 1,3–1,8%
   da seção, o foco sobe de 0,12–0,24 para 0,29–0,30, o risco horizontal e a coluna da borda caem; no sintético o IoU de
   teste muda no máximo 0,02. Três ajustes independentes da NRU repetem o ganho (0,369–0,378, cada um melhor nos 4
   patches). Nenhum voxel do Marlim entra no treino. **Fica fora do `task.json`**: a campanha compara
   as redes só com o treino sintético, como antes.

## Como foi medido

- Toda medida no Marlim usa as classes do `Blocks/Marlim/Analysis.ipynb` executadas direto das células (`TileGrid`,
  `SlidingWindow`, `FaultStickExtractor` com `STICK_TOL 0.80`, `FaultComparer`), nos 4 patches anotados (1200, 1300,
  1400, 2600). Conferido contra `outputs/marlim/sticks_report.csv` do model_6: recall 0,639, detecção 0,545, precisão
  0,214, 94 sticks — idênticos —, e a predição bate com diferença máxima de 6e-8.
- As sondas de treino executam as células do `Model/1 - Model.ipynb` (mesmo `Trainer`, split, perda, EMA, scheduler) com
  a config injetada no lugar do `Task/info.json` e a saída fora do repositório.
- Para comparar muitas sondas usei um método mais barato (slab de 128 inlines montado pelos núcleos dos tiles, janelas de
  128³ com passo 112 e peso de Hann): 300 janelas por patch contra 910 tiles. Ele difere do notebook em até ±0,04 de F1 num
  patch, então só compara sondas entre si. As tabelas marcadas **(notebook)** usam o método do notebook.
- Além dos sticks, cinco medidas da máscara (limiar 0,5) na inline anotada, porque é nela que os defeitos aparecem:

| medida | o que é |
|---|---|
| máscara | fração da seção viva (fora da água) marcada; a anotação ocupa ~0,8% |
| foco | fração da máscara a até 12 px de uma falha anotada (limite inferior: a anotação é parcial) |
| fragmentos | fração da máscara em pedaços com menos de 30 px de extensão |
| horizontal | fração da máscara com orientação local abaixo de 25° (as falhas anotadas têm 40–77°): risco ao longo de refletor |
| borda | fração marcada nas 4 primeiras colunas (o traço morto da xline 0) |

## 1. Por que IoU 0,70–0,73 e Marlim ruim

### 1.1 Todas as redes no mesmo patamar (notebook, média dos 4 patches)

| modelo | rede | IoU teste | recall | detecção | precisão | F1 | sticks | máscara | foco | fragmentos | horizontal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model_7 | nru_net | 0,725 | 0,492 | 0,479 | 0,255 | **0,327** | 62 | 2,0% | 0,223 | 0,275 | 0,063 |
| model_5 | macnn | 0,729 | 0,484 | 0,440 | 0,238 | 0,314 | 66 | 2,3% | 0,241 | 0,337 | 0,143 |
| Artigo/model_5 | dbrnet | 0,757 | 0,564 | 0,551 | 0,210 | 0,300 | 81 | 4,1% | 0,186 | 0,394 | 0,074 |
| model_6 | fault_seg_net | 0,702 | 0,567 | 0,544 | 0,198 | 0,287 | 92 | 4,8% | 0,177 | 0,357 | 0,123 |
| model_8 (só 1200) | fault_edge_former | 0,630 | 0,504 | 0,394 | 0,198 | 0,284 | 72 | 4,7% | 0,157 | 0,297 | 0,039 |

- A `dbrnet` com o maior IoU de teste (0,757) é das mais ruidosas no Marlim (máscara 4,1%, 39% de fragmentos).
- O F1 de sticks recompensa marcar pouco, porque a anotação é parcial: a mesma predição da NRU elevada a p³ vai de 0,366
  para 0,407 no 1200. Com todas as redes calibradas para a mesma máscara (1% da seção) o F1 fica entre 0,34 e 0,40 e a
  ordem muda — a `dbrnet` do `74_wu` passa a primeira. **Diferenças de ±0,04 no F1 são ruído de calibração**; por isso
  este documento sempre olha a máscara junto.
- Parte do erro é inalcançável pela rede: no 2600, abaixo de z≈1150, a sísmica é zona morta (desvio 0,02–0,03) e o
  especialista estendeu as falhas pela geologia — nenhuma rede cobre isso (cobertura 0,12–0,14 em z>1200). No 1200, 9 das
  33 falhas anotadas são as rasas e curtas (z 270–420) que o extrator nem representa (mínimo 140 px, mergulho 43–76°).

### 1.2 Multiescala não é invariância de escala

Tiles de teste do próprio split ampliados (cubo central de lado 128/f reamostrado a 128), F1 tolerante (1,5·f voxels):

| modelo | f = 1 | 1,5 | 2 | 2,5 |
|---|---|---|---|---|
| nru_net (model_7) | 0,966 | 0,933 | 0,857 | **0,782** |
| fault_edge_former (model_8) | 0,955 | 0,924 | 0,852 | 0,761 |
| fault_seg_net (model_6) | 0,965 | 0,927 | 0,835 | 0,760 |
| macnn (model_5) | 0,970 | 0,918 | 0,802 | 0,691 |
| dbrnet (model_2, mesma campanha) | 0,966 | 0,915 | 0,786 | 0,679 |

O campo receptivo multiescala ajuda (+0,08 a +0,10 sobre a `dbrnet` em 2,5×), mas os filtros são aprendidos numa escala só
— o `dataset_74` tem o centroide do espectro vertical em 7,2 px; o Marlim tem 10 px em z 650 e 15–19 px de z 1050 a 1250.

### 1.3 Os defeitos vistos nas figuras e a causa de cada um

| defeito | onde | causa |
|---|---|---|
| riscos horizontais ao longo dos refletores fortes | Fault-Seg-Net (12% da máscara contra 6% da NRU) | ver 3.1 |
| colunas verticais na água e no traço morto da borda | Fault-Seg-Net (44% das 4 colunas da borda no 1200) | ver 3.1 |
| hachura de traços paralelos no fundo | todas | textura de migração na zona morta + escala; não é da arquitetura |
| traço picotado | todas (27–39% da máscara em pedaços < 30 px) | a probabilidade oscila em torno de 0,5 fora da distribuição do treino |
| falhas profundas perdidas | todas | zona morta; o especialista interpretou sem evidência sísmica |

## 2. Revisão das redes contra os artigos

| rede | confere com o artigo? | evidência |
|---|---|---|
| `nru_net` | sim | sub-redes conferidas conv a conv contra o código do autor (lanl/scf); "we only supervise the final fault map" = só o mapa final na perda. Cabeças: no voxel positivo a de resolução plena soma +6,7 (sintético) / +4,0 (Marlim) ao logit e as outras duas subtraem — nada patológico |
| `macnn` | sim | contagem de parâmetros publicada reproduzida em 4 algarismos (11,6990 M) |
| `fault_edge_former` | sim | parâmetros e FLOPs publicados reproduzidos |
| `fault_seg_net` | **não** — eixo da atenção e gargalo | seção 3.1 |
| `nra_sanet` | implementada agora | seção 3.2 |

O que nenhuma campanha daqui reproduz é o **protocolo de treino dos artigos**: a NRU treinou com 200 volumes × 8 espelhos
(1600), a NRA-SANet com 200 × 7 (1400), o MACNN com 1000 volumes gerados, o FaultEdgeFormer com 500 volumes, Ricker de
30–50 Hz "para cobrir todas as frequências do dado real", rotação, espelho e recorte. A campanha Multiscale treinou 200
volumes sem nenhuma aumentação (gap treino-validação de 0,08–0,11). Testei recuperar isso — seção 4.

## 3. Correções no código

### 3.1 Fault-Seg-Net (`Model/Network/types/FaultSegNet.py`)

**Eixo da atenção Fault-Seg.** O artigo é 2D: em cada seção, `S` (C×H×H) compara as linhas da seção ao longo dos seus W
traços (d = W = 128). O port 3D achatava o plano x·y inteiro na dimensão da feature (d = 16 384 no estágio 1). Agora cada
seção inline tem a sua `S` de z por z sobre os traços y dela, pelo `scaled_dot_product_attention` (que não guarda as
matrizes). Conferido contra o cálculo explícito: diferença de 2,4e-7.

O que a medida mostrou, e que muda a leitura: **nas duas versões a rede aprende uma `S` uniforme** — peso na diagonal
0,04–0,06 no model_6 e 0,010 na versão corrigida (o uniforme é 0,008), |Δz| médio de 42–45 contra 42,7 do uniforme. A
atenção vira uma média vertical da coluna somada à feature (66% da norma dela), um viés "falha atravessa o tile na
vertical" que o sintético premia. A rede depende dele: zerar o termo na inferência leva a máscara de 3,5% para 9,1%. A
correção deixa o operador fiel ao artigo; o viés é do próprio mecanismo com este dado.

**Conv do gargalo.** A saída do quinto estágio passa por uma conv 3×3 + ReLU antes de subir (a seta vertical do fundo da
figura 1); faltava. Com f=32 são 7 M de parâmetros a mais, em 8³ (24,5 M no total).

**Memória.** Com f=32, 128³ e lote 2 a rede passava de 19 GiB e a conv dilatada do MSR em modo determinístico pede 6,75
GiB de workspace — a mesma armadilha do MACNN. Agora o MSR e o decoder do estágio 1 são recalculados na backward
(`CHECKPOINT`; saída e gradientes idênticos, diferença 0,0) e o `1 - Model`/`0 - Model_CV` tratam a `fault_seg_net` como o
MACNN (cuDNN não determinístico): 15,0 GiB de pico.

**Efeito, na mesma config de treino do model_6** (`dataset_74`, lr 1e-4, dropout 0,05, EMA), média dos 4 patches (método
rápido; figura `outputs/modificacoes/fault_seg_net_patch1200.png`):

| versão | épocas | F1 | precisão | máscara | foco | horizontal | borda | fragmentos |
|---|---|---|---|---|---|---|---|---|
| antiga (o código do model_6), mesma semente | 20 | 0,254 | 0,169 | 5,9% | 0,158 | 0,092 | 0,247 | **0,274** |
| **corrigida** | 20 | 0,263 | 0,179 | **3,6%** | 0,176 | **0,075** | **0,134** | 0,372 |
| corrigida | 30 | 0,259 | 0,179 | 4,0% | 0,173 | 0,083 | 0,137 | 0,346 |
| model_6 (antiga, convergida) | 90 | 0,300 | 0,211 | 4,5% | 0,180 | 0,119 | 0,315 | 0,369 |

Na mesma época e semente, a corrigida marca 39% menos (menor nos 4 patches), tem menos risco horizontal (menor nos 4
patches) e metade da coluna na borda; o F1 empata (ganha em 1200 e 1400, perde em 1300 e 2600, dentro do ruído) e os
fragmentos sobem. É uma máscara mais limpa, não um F1 maior. A comparação com o model_6 de 90 épocas mistura estágio de
treino (a antiga marca 5,9% na época 20 e 4,5% na 90).

O model_6 antigo não carrega mais com o código novo (falta a conv do gargalo). As predições dele no Marlim e no Iracema
continuam em `outputs/`, e o `Analysis.ipynb` só carrega a rede para tile ainda não predito.

### 3.2 NRA-SANet (`Model/Network/types/NRASANet.py`, `'nra_sanet'` no `ModelNetwork`)

Sun, Jiang, Du & Gong (2025), *Geophysical Prospecting* 73, 575–594 (`Documents/Articles/2 - A hybrid network...pdf`).
Lida das equações 1–17 e das figuras 1–4:

- **Encoder aninhado:** cinco U-Nets residuais de 5, 4, 3, 2 e 1 níveis, todas descendo até 1/16 com a largura máxima
  (256 no artigo, `16·num_filters`); conv de entrada → CRA → U-Net → soma com a saída do CRA (eq. 3).
- **CRA** (fig. 4, eqs. 10–15): GroupNorm, pesos por canal pelos pools globais médio e máximo, limiar 0,5 que separa
  canais "de falha" e "de ruído" e reconstrução cruzada das metades (a SRU do SCConv com os pesos do CBAM). Três leituras
  decididas pelo próprio artigo: a "conv 3×3×3" da eq. 11 age num mapa 1×1×1 (pool global), então é uma conv 1×1×1 C→C;
  sem o ReLU da eq. 11 na saída (com ele o sigmoide da eq. 12 nunca fica abaixo de 0,5 e o limiar nunca separa nada — e a
  figura 4 diz que os pesos vão a ~0 fora das falhas); W1 e W2 como as eqs. 13–14 estão impressas.
- **Decoder de atenção de duas cabeças** (fig. 3, eqs. 6–9): Q e K por 1×1 + ReLU, V por 3×3 + ReLU, softmax(QKᵀ/√d)
  sobre os voxels, cabeças concatenadas e fundidas por 1×1. Global e exata no De_4 (16³ tokens: a atenção vê o tile
  inteiro); em janelas de 8³ nos níveis maiores. A versão global nesses níveis não treina em hardware nenhum: medido na
  P6000 (kernel eficiente, forward + backward, lote 2), global no De_3 custa 24,7 s por passo e janelas de 16³ no De_1
  17 s; o De_1 global teria 2,1 milhões de tokens. O artigo não diz como fez.

Três desvios, cada um medido (recortes do `dataset_74`, lr 1e-4, mesma semente):

| versão | val_iou nas primeiras épocas | conclusão |
|---|---|---|
| como o artigo (sem norma nas convs, sem atalho na atenção, perda NR 0,3) | 0,000 nas 6 primeiras | não treina |
| idem, sem a perda NR | 0,000 nas 3 primeiras | não treina |
| com GroupNorm nas convs e atalho por cabeça na atenção, sem NR (64³) | 0,162 → 0,248 → 0,279 → 0,300 | treina |
| idem, com a perda NR (eta 0,7 do artigo) (64³) | 0,001 → 0,145 → 0,198 → 0,213 | NR custa ~0,09 de IoU |

Na inicialização sem normalização o desvio cai de ~1,2 na saída do encoder para 0,07 no decoder e 0,008 no logit (com
normalização e atalho: 0,30): a rede sai constante e a atenção sem atalho só devolve médias de V numa janela. A perda NR
(`||F − resize(maxpool(F))||²`) puxa cada saída do encoder para ser constante em blocos 2×2×2, o oposto do que uma falha
de 1–2 voxels precisa — o mesmo achado da perda MSR da Fault-Seg-Net. `ETA = 1.0` desliga; `ETA = 0.7` volta ao artigo.

Custo: f=16 (o do artigo) → 15,2 M parâmetros, 16 GiB de pico em 128³ com lote 2; f=32 não cabe na placa. Treina com lr
1e-4 (val 0,641 na época 30) e com lr 1e-3 (0,412 na época 3 em 64³).

No Marlim (método rápido, 128³, `dataset_74`, lr 1e-4; figura `outputs/modificacoes/nra_sanet_patch1200.png`):

| época | val_iou | recall | detecção | precisão | F1 | máscara | foco | fragmentos |
|---|---|---|---|---|---|---|---|---|
| 20 | 0,609 | 0,557 | 0,576 | 0,185 | 0,271 | 7,2% | 0,138 | 0,231 |
| 30 | 0,641 | 0,529 | 0,510 | 0,187 | 0,269 | 7,3% | 0,127 | **0,239** |

Desenha linhas contínuas (a menor fragmentação de todas as redes medidas) mas marca demais a esta altura do treino; com a
máscara igualada a 1% o F1 vai a 0,354, o patamar das outras. O treino de 100 épocas do `task.json` é o teste dela.

## 4. O que foi testado no treino e não melhorou o Marlim

| receita | rede | resultado | por quê |
|---|---|---|---|
| **zoom isotrópico** (f 1,2–2,6, p 0,6) + espelho + giro + gamma + suavização, `74_wu`, lr 1e-3 | NRU, época 8 | máscara 8,7–8,9% (4×), bolhas na zona morta; F1 0,338/0,235 contra 0,346/0,282 | o rótulo engorda f vezes com o tile: no sintético nativo a rede passa a marcar 9,3% contra 6,5% do rótulo, espessura 6,9 contra 4,2 voxels |
| **wavelet mais longa** (gaussiana em z, σ 0,5–4 — a troca exata de um Ricker por outro de frequência menor, rótulo intacto) + espelho + giro + gamma, `74_wu`, lr 1e-3 | NRU, época 16 | F1 médio 0,282 contra 0,320; máscara 5,1%; faixa de falso positivo sob o fundo do mar | desloca o treino para baixa frequência e o raso do Marlim (4–7 px) fica ainda mais fora da distribuição |
| **espelho em x + giro de 90° em xy** (o protocolo da NRU e da NRA-SANet), `dataset_74`, lr 1e-4 | NRU, épocas 20 e 30 | F1 0,268 / 0,264; com a máscara igualada, 0,352 contra 0,344 da original — empate; mais cobertura, menos foco | — |
| **dataset e escalonamento**: `dbrnet` em `74` percentil × `74_wu` min-max | dbrnet | máscara 2,6% × 3,9%, fragmentos 0,31 × 0,38, F1 0,304 × 0,292; com a máscara igualada, 0,367 × 0,402 | confundido: muda dataset e escalonamento juntos (o Marlim tem 1,6–3,9% saturado; o `74` em percentil 2,0%, em min-max 0%) |
| **lr 1e-4 × 1e-3** no `dataset_74` | dbrnet | F1 0,301 × 0,304 | indiferente |
| **ajuste fino com as inlines anotadas** (supervisão 2D na fatia anotada, alternada com o sintético), treino em 1200/1300/1400 | NRU | no 2600, fora do treino, F1 0,377 contra 0,398; precisão 0,401 × 0,367; perde recall até nos patches de treino | a anotação é parcial: tudo que não está anotado vira negativo e a rede aprende a apagar falha real |

Também medido: o quanto a rede marca no Marlim cai com o tempo de treino (NRU: 4,7% na época 20, 3,8% na 30, 2,0% na 85).
Por isso as comparações acima são sempre na mesma época — sonda curta não se compara com modelo convergido.

O `wavelet` foi testado no `Transforms` e saiu (o arquivo voltou a ser o original). O `zoom` continua lá, como estava,
mas engorda o rótulo de falha e não deve ser usado com ele.

## 5. Média de espelhamentos (TTA)

Média da predição sobre 4 espelhamentos da janela (identidade, inline, xline, ambos), cada saída desespelhada antes da
soma (método rápido):

| rede | patch | F1 | precisão | recall | foco | máscara |
|---|---|---|---|---|---|---|
| NRU | 1200 | 0,366 → 0,399 | 0,275 → 0,353 | 0,547 → 0,459 | 0,231 → 0,304 | 1,9% → 1,0% |
| NRU | 1400 | 0,238 → 0,266 | 0,178 → 0,224 | 0,361 → 0,327 | 0,178 → 0,252 | 1,8% → 0,9% |
| MACNN | 1200 | 0,387 → 0,338 | 0,305 → 0,281 | 0,529 → 0,422 | 0,262 → 0,302 | 2,0% → 1,2% |

Não é só um limiar mais alto: a NRU elevada a p³ (máscara 1,2%) tem foco 0,241; com a média dos espelhos (máscara 0,99%)
o foco é 0,304. Mas ela troca recall por precisão e piorou o MACNN — fica como opção, não como recomendação.

## 6. O que melhora de forma garantida: média entre redes (notebook)

Média das probabilidades das seções preditas pelo próprio `Analysis.ipynb`, mesmo extrator, mesma comparação:

| | 1200 | 1300 | 1400 | 2600 | **média** | precisão | recall | máscara | foco |
|---|---|---|---|---|---|---|---|---|---|
| melhor rede sozinha no patch | 0,346 | 0,282 | 0,287 | 0,406 | 0,327 (NRU) | 0,255 | 0,492 | 2,0% | 0,223 |
| MACNN + NRU (só multiescala) | 0,342 | 0,283 | **0,298** | 0,409 | 0,333 | 0,251 | **0,528** | 1,2% | **0,295** |
| MACNN + NRU + `dbrnet` | **0,396** | **0,318** | **0,309** | **0,463** | **0,372** | **0,298** | 0,517 | 1,4% | 0,289 |

- As três juntas ganham da melhor rede **em cada um dos 4 patches** (+0,02 a +0,06 de F1), com mais precisão, mais foco e
  um terço a menos de máscara.
- Só as multiescala juntas já sobem recall e foco acima de cada uma delas. Figuras:
  `outputs/modificacoes/media_entre_redes_patch1200.png` e `..._patch2600.png` — as falhas principais ficam, a poluição
  de cada rede some.
- Funciona porque o falso positivo de cada rede vem do que ela aprendeu de errado no sintético, e isso é diferente para
  cada arquitetura; a falha real aparece em todas. É o argumento mais forte para manter as multiescala no projeto: elas
  erram diferente da `dbrnet`.

Para usar no `Analysis.ipynb` (não mexi no notebook): depois do laço dos modelos, para cada patch, a média dos
`predicted.npy` dos modelos escolhidos na inline anotada, entregue ao `FaultComparer` com `setSection`:

```python
ENSEMBLE = ['model_5', 'model_7']   # MODELOS JÁ PREDITOS NO PATCH

for pid in PATCH_IDS:
    preds    = [{'label': 'interpretado', 'path': f'{MARLIM_DIR}/{pid}/{pid}_interpretado.png', 'color': 'blue'}, {'label': 'media', 'path': 'media', 'color': 'red'}]
    comparer = FaultComparer(pid, preds)
    section  = np.mean([np.load(f'{BASE_PATH}/{m}/marlim/patch_{pid}/predicted.npy', mmap_mode='r')[comparer.inline] for m in ENSEMBLE], axis=0)
    comparer.setSection('media', section)
    print(comparer.metrics('media'))
    comparer.show('mask', title=f'patch {pid} — média de {ENSEMBLE}')
```

## 7. Uma rede só que generaliza: auto-treino com dado real

A seção 1 mostra que o teto é a distância entre o sintético e o dado real, e a seção 6 que o consenso de arquiteturas
diferentes acerta mais no dado real do que qualquer uma delas. O auto-treino junta as duas coisas: o consenso rotula
dado real que não tem anotação, e **uma** rede, já treinada no sintético, é ajustada nesse dado (pseudo-rótulo /
destilação de ensemble, a mesma família do FaultSSL, Dou et al. 2024). A rede passa a ver o que o gerador não tem —
wavelet e contraste do dado real, saturação, estratigrafia, zona morta — com um rótulo que já descontou o erro particular
de cada arquitetura.

- **Dado real: o Iracema**, outro bloco, sem anotação nenhuma. **Nenhum voxel do Marlim entra** — nem a sísmica: o
  Marlim continua sendo só a régua. 400 tiles 128³ sorteados na faixa viva do bloco (z ~450–1440, desvio > 0,06).
- **Rótulo** (`Dataset/iracema_pseudo/Format.ipynb`): a probabilidade média de MACNN (model_5) + NRU (model_7) + `dbrnet`
  (Artigo/model_5), as três da seção 6. Máscara dura: falha onde a média ≥ 0,5 (1,45% dos voxels; o rótulo sintético
  tem ~6,5%, e a anotação do Marlim ~0,8% da seção), fundo abaixo de 0,1, e o meio (8,2%), onde as redes discordam, **fica fora da perda**.
- **Ajuste** (`Model/4 - SelfTrain.ipynb`): a rede base + 1000 passos; cada passo soma um tile real (BCE + Dice só nos
  voxels confiáveis, espelhos aleatórios em inline e xline) e um tile do **split de treino do próprio sintético** com a
  perda da rodada — o sintético segura o que a rede sabe de falha 3D. AdamW lr 1e-4, clip 1, EMA dos pesos (o `ModelEMA`
  do `1 - Model`). Os splits de validação e teste do sintético ficam de fora e medem no fim.

Resultado no Marlim (método rápido, média dos 4 patches; o mesmo rótulo e os mesmos 400 tiles para as duas redes; sondas
do harness, sem a borda morta da seção 7.2):

| rede (base) | versão | 1200 | 1300 | 1400 | 2600 | **F1** | recall | precisão | máscara | foco | fragmentos | horizontal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nru_net` (model_7) | base | 0,366 | 0,277 | 0,238 | 0,398 | 0,320 | 0,460 | 0,253 | 1,9% | 0,225 | 0,278 | 0,060 |
| | auto-treino, 500 passos | 0,394 | 0,322 | 0,281 | 0,462 | 0,365 | 0,391 | 0,356 | 1,5% | 0,288 | 0,279 | 0,062 |
| | **auto-treino, 1000 passos** | **0,408** | **0,317** | **0,337** | **0,450** | **0,378** | 0,407 | **0,367** | 1,6% | **0,299** | **0,246** | 0,057 |
| `fault_seg_net` corrigida (ép. 30, seção 3.1) | base | 0,252 | 0,237 | 0,208 | 0,340 | 0,259 | 0,512 | 0,179 | 4,0% | 0,173 | 0,346 | 0,083 |
| | auto-treino, 500 passos | 0,363 | 0,258 | 0,324 | 0,345 | 0,323 | 0,339 | 0,331 | 1,5% | 0,290 | 0,292 | 0,056 |
| | **auto-treino, 1000 passos** | 0,344 | 0,277 | 0,288 | 0,402 | **0,328** | 0,346 | 0,330 | 1,7% | **0,298** | **0,254** | **0,053** |
| as três professoras juntas (seção 6) | média | 0,412 | 0,302 | 0,319 | 0,452 | 0,371 | 0,516 | 0,297 | 1,3% | 0,295 | 0,354 | 0,061 |

Com a máscara igualada (seção 1.1: o F1 recompensa marcar pouco, então é aqui que se vê se o ganho é só calibração):

| modelo | F1 com 1% | foco | fragmentos | F1 com 1,5% | foco | fragmentos |
|---|---|---|---|---|---|---|
| `nru_net` base | 0,344 | 0,241 | 0,392 | 0,327 | 0,231 | 0,316 |
| `nru_net` auto-treino 1000 | **0,366** | **0,311** | **0,314** | **0,378** | **0,300** | **0,248** |
| `fault_seg_net` base | 0,341 | 0,214 | 0,621 | 0,327 | 0,204 | 0,552 |
| `fault_seg_net` auto-treino 1000 | 0,312 | **0,311** | **0,348** | **0,343** | **0,301** | **0,270** |
| `dbrnet` (Artigo/model_5) | 0,402 | 0,232 | 0,640 | 0,356 | 0,222 | 0,574 |
| `macnn` (model_5) | 0,362 | 0,267 | 0,471 | 0,372 | 0,255 | 0,390 |

O que isso diz:

- **A NRU ajustada é a melhor rede única medida no Marlim** e chega sozinha ao nível da média das três professoras
  (aqui 0,378 × 0,371; pelo método do notebook, seção 7.1, 0,369–0,378 × 0,372): melhor que a base nos 4 patches,
  precisão +45%, e com a máscara igualada continua acima (+0,02 a +0,05).
- **O ganho que não depende de calibração é o foco**: com a mesma quantidade de máscara, 30–31% dela cai a até 12 px de
  uma falha anotada, contra 20–27% de qualquer rede só treinada no sintético, e os fragmentos caem de 0,32–0,64 para
  0,25–0,35. É exatamente a lista de defeitos da seção 1.3: menos poluição no fundo, menos traço picotado, menos risco
  ao longo de refletor (horizontal 0,083 → 0,053 na Fault-Seg-Net).
- A Fault-Seg-Net sobe 0,07 de F1 no limiar nativo porque deixa de marcar 4% da seção; com 1% igualado ela perde 0,03 de
  F1 e ganha foco e continuidade — mais limpa, não mais sensível.
- **O custo é recall**: 0,46 → 0,41 na NRU, porque o pseudo-rótulo só marca onde as três concordam. As falhas que
  continuam perdidas são as mesmas de antes — zona morta do 2600 e falhas rasas e curtas do 1200 (seção 1.1).
- 1000 passos > 500 na NRU (0,378 × 0,365); na Fault-Seg-Net empata. O padrão do notebook é 1000.
- **Alvo duro × suave.** Com a probabilidade média como alvo (`soft: true`, destilação sem região ignorada) o aluno
  herda a marcação larga da base: no limiar 0,5, F1 0,339 na NRU e 0,306 na Fault-Seg-Net (500 passos, contra 0,365 e
  0,323 do duro), máscara 2,0–2,2%, precisão 0,24–0,28. Igualando a máscara ele empata com o duro em F1 (1%: 0,376 ×
  0,366 na NRU com 250 passos, 0,347 × 0,312 na fsn; 1,5%: 0,352 × 0,378 e 0,369 × 0,343), mas com menos foco
  (0,26–0,29 × 0,30–0,31) e mais fragmentos (0,32–0,40 × 0,25–0,35). Como o `Analysis.ipynb` corta em 0,5, o padrão é o
  duro (`soft: false`).

### 7.1 Verificação pelo método do notebook, com o modelo que o `4 - SelfTrain.ipynb` grava

O `4 - SelfTrain.ipynb` rodado por papermill sobre o model_7 (1000 passos, alvo duro), e o modelo que ele salvou avaliado
com as classes do `Analysis.ipynb` (`TileGrid` + `SlidingWindow` nos 910 tiles de cada patch, o mesmo da tabela 1.1).
São três ajustes independentes — o do harness (sem borda morta), o notebook com a borda morta só no lado do tile e o
notebook como ficou (seção 7.2) — para ver se o ganho se repete:

| modelo | 1200 | 1300 | 1400 | 2600 | **F1** | recall | detecção | precisão | máscara | foco | fragmentos | horizontal | borda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NRU model_7 (só sintético) | 0,346 | 0,282 | 0,275 | 0,406 | 0,327 | 0,493 | 0,479 | 0,255 | 2,0% | 0,223 | 0,275 | 0,063 | 0,259 |
| auto-treino, harness, sem borda morta | **0,415** | 0,313 | **0,333** | 0,426 | 0,372 | 0,399 | 0,423 | **0,356** | 1,6% | 0,293 | **0,215** | 0,061 | 0,341 |
| auto-treino, notebook, borda só no lado | 0,394 | 0,330 | 0,312 | **0,474** | **0,378** | 0,454 | 0,434 | 0,337 | 1,8% | 0,280 | 0,248 | 0,062 | 0,284 |
| **auto-treino, `4 - SelfTrain` como entregue** | 0,376 | **0,337** | 0,324 | 0,438 | 0,369 | 0,421 | 0,390 | 0,342 | **1,5%** | **0,304** | 0,258 | 0,062 | **0,042** |
| MACNN + NRU + `dbrnet` (média, seção 6) | 0,396 | 0,318 | 0,309 | 0,463 | 0,372 | 0,517 | — | 0,298 | 1,4% | 0,289 | — | — | — |

- **Os três ajustes ganham da base em cada um dos 4 patches** (+0,03 a +0,07 por patch; F1 médio 0,369–0,378 contra
  0,327), com sorteios, dados de reposição e borda diferentes: o ganho não é sorte de uma rodada. A diferença entre eles
  (±0,02 por patch) é o ruído entre rodadas; a que ficou no notebook tem o maior foco, a menor máscara e a borda limpa.
- Uma rede só faz o que antes pedia a média de três: F1 empatado com o ensemble (0,369–0,378 × 0,372), mais precisão
  (0,34 × 0,30) e mais foco (0,30 × 0,29).
- No sintético a rede se mantém: IoU de validação 0,713 → 0,698–0,703 e de teste 0,725 → 0,715–0,719 (a variação entre
  rodadas idênticas é ~0,03).
- Custo medido: 1000 passos da NRU levam ~1,5 h na P6000 sozinha (~5 s por passo de um tile real + um sintético).
- **O modelo da última linha está em `model_9` desta pasta** (base model_7, `processing.selftrain` no `info.json`). Para
  ver no bloco: `Blocks/Marlim/Analysis.ipynb` com `BASE_PATH = '../../Documents/Backups/Multiscale'` e
  `MODELS = ['model_7', 'model_9']`, `PATCH_IDS = ['1200', '1300', '1400', '2600']`.
- Figuras: `outputs/modificacoes/autotreino_notebook_patch1200.png` e `..._patch2600.png` (método do notebook) e
  `autotreino_patch1200.png`/`..._patch2600.png` (NRU e Fault-Seg-Net, método rápido).

### 7.2 Borda morta do levantamento

O auto-treino sem cuidado nenhum piorava um defeito da seção 1.3: a coluna vertical na borda do patch (borda 0,259 →
0,341). A causa está no dado: o Marlim tem traços mortos na xline 0 (amplitude zero, 0,503 em `[0, 1]`; um traço no
1200–1400, cinco no 2600) com uma rampa de 1–3 traços até o dado vivo, e os tiles da borda do patch chegam **espelhados**
em torno deles (o corte completa a margem de 32 voxels por reflexão: no tile `w000` do 1200 o traço morto fica na
posição 32, com dado vivo refletido dos dois lados). Uma coluna constante no meio de dado vivo é, para a rede, uma
descontinuidade vertical perfeita. Os 400 tiles do Iracema foram sorteados vivos, então o aluno nunca viu isso e
reforça o que o sintético ensinou.

O `deadEdge` do `4 - SelfTrain` (chance `dead` = 0,25 por tile real) cria essa situação com alvo de fundo: 1–8 traços
mortos (a média do tile, que no Iracema é a amplitude zero: 0,495 × 0,496), rampa de 0–3 traços, alvo zerado até 2
traços além da rampa; metade das vezes no lado do tile, metade entre 1 e 40 traços para dentro com o dado e o alvo
espelhados do outro lado (o layout dos tiles do Marlim). A versão só com o lado do tile resolvia a janela deslizante
contínua (borda 0,20 → 0,014 no método rápido) mas não os tiles espelhados (0,284 no notebook); com o espelho, a
probabilidade na xline 0 do 1200 cai de 0,57 (base) e 0,60–0,65 (alunos sem espelho) para 0,21, e a borda para 0,042.
O F1 não paga nada por isso (0,369 contra 0,372–0,378 das outras duas, dentro do ruído).

### 7.3 Todas as redes, a mesma receita

O `4 - SelfTrain.ipynb` como entregue (padrões: 1000 passos, alvo duro, borda morta, o mesmo pseudo-rótulo) sobre uma
base de cada rede; base e aluno avaliados pelo método do notebook nos 4 patches. As bases são as que existem: as da
campanha Multiscale (`dataset_74`, lr 1e-4), a Fault-Seg-Net corrigida e a NRA-SANet das sondas (30 épocas) e a `dbrnet`
do Artigo/model_5, que **é exatamente a parametrização do `task.json`** (`dataset_74_wu`, normalize, lr 1e-3, 100 épocas).

| rede | versão | 1200 | 1300 | 1400 | 2600 | **F1** | precisão | máscara | foco | horizontal | borda | IoU teste sint. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nru_net` (model_7) | base | 0,346 | 0,282 | 0,275 | 0,406 | 0,327 | 0,255 | 2,0% | 0,22 | 0,063 | 0,26 | 0,725 |
| | **auto-treino** | 0,376 | 0,337 | 0,324 | 0,438 | **0,369** | 0,342 | 1,5% | 0,30 | 0,062 | 0,04 | 0,715 |
| `fault_seg_net` corrigida (ép. 30) | base | 0,271 | 0,238 | 0,228 | 0,345 | 0,270 | 0,184 | 4,6% | 0,16 | 0,087 | 0,23 | 0,670 |
| | **auto-treino** | 0,372 | 0,261 | 0,341 | 0,378 | **0,338** | 0,309 | 1,8% | 0,30 | 0,063 | 0,03 | 0,659 |
| `nra_sanet` (ép. 30) | base | 0,257 | 0,188 | 0,253 | 0,352 | 0,262 | 0,182 | 7,4% | 0,12 | 0,126 | 0,29 | 0,649 |
| | **auto-treino** | 0,434 | 0,294 | 0,328 | 0,413 | **0,367** | 0,362 | 1,6% | 0,29 | 0,067 | 0,18 | 0,631 |
| `macnn` (model_5) | base | 0,329 | 0,253 | 0,287 | 0,388 | 0,314 | 0,238 | 2,3% | 0,24 | 0,143 | 0,21 | 0,730 |
| | **auto-treino** | 0,373 | 0,336 | 0,314 | 0,434 | **0,364** | 0,357 | 1,3% | 0,29 | 0,090 | 0,08 | 0,720 |
| `dbrnet` (Artigo/model_5, parametrização do task) | base | 0,293 | 0,269 | 0,241 | 0,396 | 0,300 | 0,210 | 4,1% | 0,19 | 0,075 | 0,28 | 0,757 |
| | **auto-treino** | 0,381 | 0,335 | 0,293 | 0,453 | **0,365** | 0,314 | 1,7% | 0,30 | 0,036 | 0,09 | 0,757 |
| `fault_edge_former` (model_8) | base | 0,284 | 0,180 | 0,222 | 0,286 | 0,243 | 0,176 | 4,6% | 0,15 | 0,064 | 0,37 | 0,630 |
| | **auto-treino** | 0,375 | 0,282 | 0,290 | 0,304 | **0,313** | 0,382 | 1,3% | 0,30 | 0,067 | 0,08 | 0,628 |

- **As seis redes melhoram em cada um dos 4 patches — 24 de 24 comparações**, de +0,02 a +0,18 por patch; o F1 médio
  sobe de 0,24–0,33 para 0,31–0,37. Cinco das seis ajustadas (todas menos a FaultEdgeFormer, 0,313) passam a melhor rede
  só sintética (NRU, 0,327).
- **Os defeitos da seção 1.3 caem em todas**: máscara de 2,0–7,4% para 1,3–1,8% (a anotação ocupa ~0,8%), foco de
  0,12–0,24 para 0,29–0,30, risco horizontal da MACNN 0,143 → 0,090 e da NRA-SANet 0,126 → 0,067, coluna da borda de
  0,21–0,37 para 0,03–0,18. Figura: `outputs/modificacoes/autotreino_multiescala_patch1200.png` (Fault-Seg-Net,
  NRA-SANet e MACNN antes e depois, método do notebook) — a hachura e o risco ao longo dos refletores somem e ficam os
  segmentos sobre as falhas anotadas.
- **O ganho é maior onde a base era pior** (NRA-SANet +0,105, FaultEdgeFormer +0,070, Fault-Seg-Net +0,068): o
  auto-treino corrige justamente o que o sintético ensinou de errado, e a arquitetura volta a pesar menos que o dado —
  com a mesma receita, NRA-SANet, `dbrnet`, MACNN e NRU terminam empatadas em 0,364–0,369.
- No sintético nenhuma rede desaprende: IoU de teste −0,000 a −0,018.
- **Todos os modelos desta tabela estão nesta pasta**, para conferir no `Blocks/Marlim/Analysis.ipynb` com
  `BASE_PATH = '../../Documents/Backups/Multiscale'` (o `trainer.base` de cada ajustado diz de onde veio):

  | rede | só sintético | com auto-treino |
  |---|---|---|
  | `nru_net` | model_7 | model_9 |
  | `fault_seg_net` corrigida | model_13 (ép. 30) | model_14 |
  | `nra_sanet` | model_15 (ép. 30) | model_16 |
  | `macnn` | model_5 | model_10 |
  | `fault_edge_former` | model_8 | model_11 |
  | `dbrnet` | `../Artigo/model_5` | model_12 |

  O model_6 (Fault-Seg-Net antiga) não carrega com o código corrigido; a corrigida é o model_13.
- **A média entre redes continua somando por cima** (seção 6, mesmas seções do método do notebook):

  | média de | 1200 | 1300 | 1400 | 2600 | F1 | precisão | recall | máscara | foco |
  |---|---|---|---|---|---|---|---|---|---|
  | NRU + MACNN + `dbrnet` só sintéticas (seção 6) | 0,396 | 0,318 | 0,309 | 0,463 | 0,371 | 0,298 | 0,517 | 1,4% | 0,29 |
  | as mesmas três ajustadas | **0,432** | 0,309 | **0,356** | **0,463** | **0,390** | **0,375** | 0,422 | 1,2% | **0,33** |
  | só multiescala ajustadas (NRU, Fault-Seg-Net, NRA-SANet, MACNN) | 0,368 | **0,319** | 0,370 | 0,445 | 0,376 | 0,369 | 0,406 | 1,2% | 0,32 |

  Uma rede ajustada sozinha (0,364–0,369; a NRU chega a 0,378 em outro sorteio) empata com a média das três originais;
  a média das ajustadas é o melhor resultado medido no Marlim, e a das multiescala ajustadas também passa a antiga.
- **O custo é recall** (NRU 0,49 → 0,42, FaultEdgeFormer 0,44 → 0,28) em troca de precisão (0,18–0,26 → 0,31–0,38): o
  rótulo só marca onde três redes concordam. Na FaultEdgeFormer a troca é a mais forte e o 2600 quase não muda (0,286 →
  0,304).
- A `dbrnet` na parametrização do `task.json` ganha o mesmo que as multiescala (+0,065): a comparação do `task.json`
  continua justa, e as multiescala ajustadas ficam no nível dela.

## 8. Limite e próximo passo

O teto é de transferência do sintético para o real. As redes multiescala não mudam esse teto sozinhas; o auto-treino
(seção 7) é o primeiro passo que o move. O que ainda pode mudar:

1. **Segunda rodada de auto-treino — testada e rejeitada.** Com as redes ajustadas (NRU, MACNN, `dbrnet`) como
   professoras, o consenso concorda muito mais (faixa ignorada 8,2% → 2,9%) mas marca mais falha (1,45% → 2,3% dos
   voxels), e a NRU ajustada nesse rótulo a partir do model_7 cai para 0,340 (0,316/0,300/0,312/0,434) contra 0,369 da
   primeira rodada, precisão 0,342 → 0,299: as alunas já marcam com confiança o que a primeira rodada ensinou e o erro
   delas vira rótulo. O que pode ajudar é dado real de mais blocos sem anotação no lugar de só o Iracema.
2. **Supervisão com as inlines anotadas, com perda para rótulo parcial.** O teste da seção 4 mostra que tratar o não
   anotado como negativo destrói o recall; o caminho é uma perda que só confia no positivo e no negativo seguro (MD loss,
   Dou et al. 2022; aprendizado positivo-não-rotulado), com validação cruzada por patch (1200–1400 contra 2600).
3. **Gerador com o que o Marlim tem e o sintético não:** descontinuidade estratigráfica (onlap, truncamento, clinoforma)
   como negativo, saturação, zona morta.

## 9. `Task/task.json` e `Model/Backup`

Só as redes cujo código mudou vão para o treino; as outras entram como cópia dos modelos já treinados, sem retreinar.

`Task/task.json` — duas rodadas no `dataset_74_wu`, na parametrização da `dbrnet` (`normalize`, lr 1e-3, `dice_focal`,
lote 2, `plateau`, dropout 0,1, EMA, 100 épocas), sem aumentação e sem auto-treino; o `Task/index.py` é o original:

| rede | num_filters | por quê |
|---|---|---|
| fault_seg_net | 32 | atenção e gargalo corrigidos (seção 3.1); 24,5 M, 15 GiB com o checkpoint |
| nra_sanet | 16 | implementada agora (seção 3.2); o do artigo, f=32 não cabe na placa |

`Model/Backup` — cópias (`info.json`, `model.pth`, `train.png`, `predictions/`; pesos idênticos aos de origem, que ficam
em `trainer.source`), numeradas antes para o `1 - Model` gravar as duas rodadas como model_6 e model_7:

| pasta | rede | origem | treino |
|---|---|---|---|
| model_1 | dbrnet | `Artigo/model_5` | `dataset_74_wu`, normalize, lr 1e-3 — **a mesma parametrização das duas rodadas** |
| model_2 | dbrnet | `Multiscale/model_2` | `dataset_74`, percentil, lr 1e-4 (campanha Multiscale) |
| model_3 | macnn | `Multiscale/model_5` | idem |
| model_4 | nru_net | `Multiscale/model_7` | idem |
| model_5 | fault_edge_former | `Multiscale/model_8` | idem |

A `fault_seg_net` e a `nra_sanet` novas comparam direto com a model_1. As copiadas da campanha Multiscale treinaram em
outro dataset e lr: contra elas a comparação justa é a model_2 (a `dbrnet` da mesma campanha). As predições antigas no
Marlim/Iracema não foram copiadas; o `Blocks/Marlim/Analysis.ipynb` com `BASE_PATH = '../../Model/Backup'` refaz. Uma rodada custa
de 8 a 14 h; olhar a máscara junto com o F1 (seção 1.1).

## Arquivos alterados

| arquivo | mudança |
|---|---|
| `Model/Network/types/FaultSegNet.py` | atenção por seção inline (SDPA), conv do gargalo, checkpoint no estágio 1 |
| `Model/Network/types/NRASANet.py` | novo |
| `Model/Network/index.py` | import e `'nra_sanet'` no `get()` |
| `Model/1 - Model.ipynb`, `Model/0 - Model_CV.ipynb` | `fault_seg_net` na exceção do cuDNN não determinístico, junto do `macnn` (uma linha cada) |
| `Task/task.json` | `fault_seg_net` e `nra_sanet` no `dataset_74_wu` (seção 9) |
| `Model/Backup/model_1` a `model_5` | cópias dos modelos já treinados das redes não modificadas (seção 9) |
| `Dataset/iracema_pseudo/Format.ipynb` | experimento da seção 7, fora da campanha: pseudo-rótulo do Iracema (`.npy` fora do git, 5,5 GB) |
| `Model/4 - SelfTrain.ipynb` | experimento da seção 7, fora da campanha: auto-treino de um modelo salvo |
| `Documents/Backups/Multiscale/model_9` a `model_16` | novos: as redes da seção 7.3 antes (13, 15) e depois do `4 - SelfTrain` (9, 10, 11, 12, 14, 16), para comparar no `Analysis.ipynb` |
