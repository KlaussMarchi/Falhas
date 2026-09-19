# Similaridade entre uma região de Marlim e um lote de tiles sintéticos

Esta pasta calibra o `SyntheticGenerator` região a região. O `Analysis.ipynb` procura, com o
`NatureSelector`, a configuração que faz o gerador produzir tiles parecidos com os tiles reais que o
`../Marlim/Analysis.ipynb` separou em `calm`, `faulted` e `dead`.

Este documento explica a **função de similaridade** que serve de objetivo para essa busca: o que ela
mede, por que foi construída assim, quanto vale cada nota e de onde vem cada peça.

**Como rodar.** O `Analysis.ipynb` é autossuficiente: não lê arquivo de configuração nenhum — o
melhor genoma conhecido de cada região é a constante `Generator.REFERENCE` — e a única entrada são
os tiles reais de `../Marlim/files`. Basta executar de cima para baixo; cada região são duas células
(`SEARCH.update(região)` e `SEARCH.show(região)`), ~60 min por região. O que sai é o `regions.json`,
que o `../Generate.ipynb` consome para montar `Dataset/marlim_nature`. Rodar de novo **estende** a
campanha e nunca piora o arquivo, porque a escolha final compara o achado novo com a referência e
com o que já estava gravado, sempre em sementes que a busca não viu.

---

## 1. O problema

Comparar **um lote de 15–20 tiles sintéticos** com **os 123–188 tiles reais** de uma região e
devolver **uma porcentagem**: 100% quer dizer "este lote poderia ter saído do bloco de Marlim
naquela região", 0% quer dizer "não tem nada a ver". A nota precisa:

1. **enxergar falha** — é para treinar detector de falha, então a nota tem de cair quando a falha
   sintética aparece na imagem de um jeito que a falha real não aparece;
2. **funcionar com 15–20 amostras** — cada tile 128³ custa de 6 a 25 s para gerar, e a busca vai
   pedir centenas de lotes;
3. **ser diagnóstica** — quando a nota é 70%, é preciso saber *o que* está errado, senão não há como
   corrigir o gerador;
4. **não ser enganável** — não pode dar nota alta para um lote que acerta o contraste e erra a
   textura, nem para um que acerta tudo menos a falha.

## 2. Por que não usar FID, KID ou MMD direto

A rota padrão para medir "quão parecidos são dois conjuntos de imagens" é comparar distribuições num
espaço de features de uma rede pré-treinada: FID (Heusel et al., 2017), KID (Bińkowski et al., 2018)
ou MMD (Gretton et al., 2012). O estudo empírico de Xu et al. (2018) testou essas métricas e concluiu
que MMD e o teste do vizinho mais próximo (1-NN) são as que têm as propriedades desejáveis —
**desde que as distâncias sejam calculadas num espaço de features adequado**. É justamente esse o
ponto que quebra aqui:

- **não existe embedding pré-treinado para sísmica 3D.** A Inception foi treinada em fotografia; as
  features dela não descrevem frequência dominante, coerência de refletor nem rejeito de falha.
- **FID é enviesado com poucas amostras.** Com 18 tiles a estimativa da covariância é ruído; foi
  para resolver isso que o KID trocou o estimador por um **não enviesado**.
- **a nota não sai interpretável.** FID e MMD devolvem um número numa escala arbitrária; aqui a nota
  precisa ser uma porcentagem que se leia direto.
- **falha não é "textura".** Nenhuma dessas métricas separa "a imagem tem descontinuidades com
  geometria de falha" de "a imagem é rugosa".

A decisão foi manter a **ideia central** (distância entre duas amostras num espaço de features) e
trocar as duas peças: o **espaço de features** vira uma régua sísmica interpretável, e a
**distância** vira uma que já nasce normalizada entre 0 e 1.

## 3. A estatística: coeficiente de energia normalizado

Para duas amostras $X$ e $Y$, a **distância de energia** (Székely & Rizzo) é

$$D^2(F,G) = 2\,\mathbb{E}\lVert X-Y \rVert - \mathbb{E}\lVert X-X' \rVert - \mathbb{E}\lVert Y-Y' \rVert \ \ge 0,$$

com $D = 0$ **se e somente se** as duas distribuições são iguais. A revisão de Rizzo & Székely
(2016, p. 29) dá a versão normalizada:

$$H = \frac{D^2(F,G)}{2\,\mathbb{E}\lVert X-Y \rVert} = \frac{2A - B - C}{2A}, \qquad 0 \le H \le 1,$$

onde $A$, $B$ e $C$ são as médias das distâncias par a par entre as duas amostras, dentro da
primeira e dentro da segunda. A similaridade da feature é

$$\mathrm{sim} = 100 \times (1 - H)\ \%.$$

Por que esta e não outra:

- **já é porcentagem.** $H$ vive em $[0,1]$ por construção, sem escala arbitrária e sem calibrar
  temperatura de exponencial;
- **zera só quando as distribuições são iguais** — não compara só médias: pega deslocamento, escala
  e forma;
- **em 1D é exatamente a distância de Cramér** ($D^2 = 2\int (F-G)^2$), isto é, a área entre as duas
  funções de distribuição acumulada — que é o que se quer dizer com "as duas amostras se sobrepõem";
- **é invariante a escala e a translação em 1D**: numerador e denominador têm a mesma unidade, então
  nenhuma feature precisa ser padronizada antes, e nenhuma domina a conta por ter unidade maior;
- **tem estimador não enviesado (U-statistic)** e funciona com amostras pequenas e desbalanceadas
  (18 sintéticos contra 149 reais);
- **não tem hiperparâmetro** — ao contrário do MMD, que precisa de kernel e largura de banda. E não
  é uma escolha exótica: com o kernel de distância, MMD e distância de energia são a **mesma
  estatística** (Sejdinovic et al., 2013).

Calibração medida (seção 7): 18 tiles reais contra o resto da própria região dão **99%**, que é o
teto atingível; um patch inteiro contra o resto da sua região dá 97%; uma região diferente dá 58–79%.

### A resolução ε

Uma feature pode ser degenerada no real — `maskFrac` (fração rotulada como falha) é exatamente 0 em
todos os tiles `calm`, porque a região foi definida assim. Como $H$ é invariante a escala, qualquer
valor sintético não nulo, por menor que seja, daria $H \approx 1$. Por isso cada feature ganha uma
**resolução**: a menor diferença que ainda importa,

$$H_f = \frac{2A - B - C}{2A + \varepsilon_f}, \qquad
\varepsilon_f = 0.25 \times \max\bigl(\mathrm{IQR}_f(\text{região}),\ 0.05 \times \mathrm{IQR}_f(\text{bloco})\bigr).$$

A leitura é direta: *diferença menor que um quarto da variação natural daquela feature **dentro da
própria região** não conta*. O piso de 5% do IQR do bloco inteiro só entra quando a feature é
degenerada na região, e evita que uma única feature zere o grupo.

A escolha da região como escala foi medida, não adotada por gosto: com o IQR do bloco inteiro no
lugar do da região, o ε fica grande justamente nas features que separam as regiões (σ, coerência,
descontinuidade) e a nota amolece onde mais importa — `calm` × `faulted` sobe de 73% para 79% e as
configurações do `Generator2` ganham 3 a 4 pontos de graça. O teto é o mesmo nas duas versões.

## 4. A régua: 5 grupos, 42 features

O espaço de features é a peça que o Xu et al. (2018) chama de "suitable feature space". Cada tile
128³ vira **um vetor**, calculado **exatamente com o mesmo código** nos dois lados. As features saem
de 8 seções 2D por tile — 4 inline e 4 crossline, nas posições 16, 48, 80 e 112. A média das seções
é o valor do tile, **menos as que dependem de qual eixo é o lateral da seção** (`lag*` e `Px*`), que
saem uma por orientação — sufixo `X` quando o lateral é a xline, `I` quando é a inline. As duas
direções não são iguais no bloco: no `faulted` real o `lag16` vale 0.416 na inline e −0.004 na
xline, e a média das oito seções escondia isso, dando "continuidade OK" com os dois eixos errados.

| grupo | peso | features | o que captura |
|---|---|---|---|
| `amplitude` | 0.15 | `logStd`, `kurt`, `satFrac`, `q95n` | contraste, cauda da distribuição e saturação (o slab real satura no próprio p01/p99) |
| `espectro` | 0.20 | `Pz0..Pz5`, `logPeakZ`, `Px0..Px3` em `X` e `I` | assinatura da wavelet e espessura das camadas (vertical), comprimento de onda das dobras (lateral, por eixo) |
| `estrutura` | 0.20 | `cohMean`, `coh25/50/75`, `dip10/50/90` | coerência e mergulho aparente pelo tensor de estrutura: refletor contínuo e sua inclinação |
| `continuidade` | 0.15 | `lag1/2/4/8/16` em `X` e `I`, `zcr` | até onde o refletor pode ser seguido em cada direção lateral, e quantas vezes o traço cruza o zero |
| `falhas` | 0.30 | `discFrac`, `lineDens`, `lineDip`, `lineLen`, `discSharp` | **a falha como ela aparece na imagem**, sem usar rótulo |

As três primeiras famílias vêm de atributos sísmicos clássicos: **tensor de estrutura** para mergulho
e coerência (Van Vliet & Verbeek, 1995; Fehmers & Höcker, 2003) e **espectro** por janela de
profundidade, que é como o bloco de Marlim já vinha sendo medido neste projeto. A régua é uma evolução
direta das 40 features do `Generator2.ipynb`, que já tinham sido validadas neste projeto.

### Como as falhas entram

Este é o ponto que não podia se perder, e ele tem **duas metades**.

**(a) A falha na imagem, sem rótulo.** Para cada seção calcula-se a **semblance com direção de
mergulho** — semblance de 5 traços alinhados pelo mergulho local do tensor de estrutura, a mesma
receita de coerência de Marfurt et al. (1998) e Bahorich & Farmer (1995). Onde há refletor contínuo
a semblance é ~1; onde a falha corta, ela cai. Dos voxels com semblance < 0.8 saem:

- `discFrac` — fração da seção descontínua;
- `lineDens` — comprimento total de **lineamentos tipo falha** (componentes conexas com ≥ 24 px e
  mergulho entre 35° e 88°) por unidade de área: é a densidade de falha **visível**;
- `lineDip`, `lineLen` — mergulho e comprimento medianos desses lineamentos;
- `discSharp` — razão entre a descontinuidade no lineamento e fora dele: mede se a falha é um corte
  nítido ou um borrão.

Isso responde à pergunta que interessa ao treino: **a falha sintética se parece com a falha real
naquilo que a rede enxerga** — quantas, com que inclinação, quão compridas, quão nítidas.

**(b) O rótulo contra a imagem.** No real existe a interpretação do especialista, numa inline por
tile (a `inline` do `DataBase.csv`); no sintético existe a máscara do gerador. Compara-se **uma
seção rotulada por tile dos dois lados**, nas mesmas condições:

- `maskFrac` — fração rotulada;
- `visibility` — descontinuidade mediana **sob o rótulo** dividida pela **fora dele**. Razão ≈ 1
  significa rótulo em cima de imagem lisa, que é o pior caso possível: ensina a rede a inventar
  falha. No real, os tiles `faulted` dão 2.3;
- `maskDip`, `maskLen` — mergulho e comprimento dos traços rotulados (o especialista anota mergulho
  mediano de 68° e traços de ~150 px dentro de um tile).

**Por que não calibrar só pela estatística da máscara.** Em 2026-08-17 este projeto já tentou isso e
deu errado: ajustar fração, comprimento e mergulho da máscara acertou os números **pelo motivo
errado** — a busca chegou a superfícies fragmentadas, rejeito com piso zero e traço quebrado, ou
seja, rótulo bonito sobre imagem lisa. A métrica que pega esse erro é a **visibilidade**, e é por
isso que ela está aqui, junto com a metade (a): a nota só sobe se a falha existir **na imagem**.

### O que ficou de fora, e por quê

- **GLCM / Haralick (1973)** e estatísticas de textura tipo Portilla & Simoncelli (2000): descrevem
  textura bem, mas não separam falha de rugosidade, e as features não têm leitura geofísica.
- **Fault likelihood de Hale (2013)**: é o atributo de falha mais forte que existe, mas custa caro
  (varredura em mergulho e azimute) e a busca chama a função centenas de vezes. A semblance com
  direção de mergulho é a versão barata do mesmo princípio.
- **1-NN / C2ST (Lopez-Paz & Oquab, 2016)**: dá porcentagem bonita (50% de acurácia = indistinguível),
  mas com 18 amostras o desvio da acurácia é de ~8 pontos, o que vira ~16 pontos de nota — ruído
  demais para guiar CMA-ES.

## 5. As mesmas condições

A comparação só vale se os dois lados passarem pelo mesmo funil:

| item | real | sintético |
|---|---|---|
| tile | 128³, eixo `(inline, z, xline)` | idem (transposição `(0,2,1)` do gerador) |
| escala | [0,1] do p01/p99 do slab | `clip(img × ganho × jitter, ±0.45)` → [0,1] |
| zero | mediana do próprio tile | mediana do próprio tile |
| seções | 4 inline + 4 crossline (16, 48, 80, 112) | idem |
| seção rotulada | a inline anotada pelo especialista | a inline central |
| quantidade | todos os tiles da região (123–188) | 18 tiles |

O **ganho** não é procurado pela busca: para cada lote ele é resolvido por bisseção, como o escalar
que leva o σ mediano pós-format ao σ mediano da região real. Contraste é o parâmetro mais fácil de
acertar e o que mais domina qualquer métrica de imagem; tirá-lo da disputa faz a nota medir o que
interessa — textura, estrutura e falha — e economiza uma dimensão na busca. O `jitter` (variação de
contraste entre tiles, log-normal) continua sendo procurado, porque o bloco real tem essa variação —
mas o sorteio é **normalizado para mediana 1** antes da bisseção, senão o ganho ficaria preso ao
sorteio daquele lote (com 18 tiles a mediana chega a 1.17) e não valeria para um lote de outro
tamanho, como os 220 tiles por região do `../Generate.ipynb`.

## 6. Agregação

$$\mathrm{grupo}_g = \frac{1}{|g|}\sum_{f \in g}(1 - H_f), \qquad
\mathrm{nota} = 100 \times \prod_g \mathrm{grupo}_g^{\,w_g}, \qquad \sum_g w_g = 1.$$

Média **aritmética dentro** do grupo (features redundantes não desequilibram) e média **geométrica
ponderada entre** grupos: um grupo quebrado não é compensado por outro perfeito. Um lote com
textura impecável e falha errada não chega a 90%, que é exatamente o que se quer de um objetivo cujo
produto final é um detector de falhas. Feature que não existe nos dois lados (por exemplo
`visibility` no `calm`, onde o especialista não anotou nada) sai da conta, e os pesos dos grupos
restantes são renormalizados.

## 7. Calibração: quanto vale cada nota

Tudo medido nos tiles reais deste projeto (149 `calm`, 188 `faulted`, 123 `dead`), com 18 tiles no
lado "sintético".

**Teto e discriminação**

| comparação | nota |
|---|---|
| 18 tiles reais da região contra o resto da mesma região | **99.1–99.5% (± 0.7)** — teto |
| um patch inteiro contra o resto da sua região (12 combinações) | 97.1% em média, 92.6–99.8 |
| região vizinha (`calm` × `faulted`) | 73–79% |
| região oposta (`calm` × `dead`) | 58% |

A linha do meio é a régua prática: **a variação natural entre patches de Marlim custa 1 a 7 pontos**
(o `2600`, o patch estruturalmente diferente, é o que cai para ~93 nas três regiões). Então um lote
sintético acima de ~93% está dentro da variação do próprio bloco, e cada ponto abaixo de 90% é
diferença que o olho enxerga.

**Degradações controladas** (partindo de 18 tiles reais `calm`, nota 99.0)

| perturbação | nota |
|---|---|
| nenhuma | 98.9 |
| blur gaussiano σ = 0.5 / 1 / 2 | 98.8 / 96.6 / 89.9 |
| ruído branco 2% / 5% / 10% / 20% | 95.3 / 84.9 / 77.5 / 62.8 |
| contraste ×0.6 / ×1.6 | 96.9 / 95.3 |

A nota cai de forma monótona e na ordem certa. Contraste quase não mexe — como esperado, já que o
ganho é ajustado antes da comparação.

**Configurações do `Generator2.ipynb`** (as três já calibradas à mão, mais a genérica do
`dataset_74`), 18 tiles cada, contra cada região real:

| configuração | `calm` | `faulted` | `dead` |
|---|---|---|---|
| `calmo` | **83.3** | 75.9 | 56.0 |
| `falhado` | 77.4 | **86.5** | 62.5 |
| `morto` | 68.1 | 74.0 | 73.5 |
| `dataset_74` (genérica) | 51.5 | **72.1** | 51.2 |

O `calmo` e o `falhado` tiram a maior nota na região para a qual foram calibrados, e a genérica
perde para as duas. O `morto` era a exceção: 73.5 na zona morta contra 74.0 no falhado — um empate
técnico que parecia teto do gerador, porque a `SyntheticGenerator` monta refletividade 1D e convolve
em z, o campo sai localmente planar e a coerência não descia de ~0.90 contra os 0.68 do real.

**Era teto do `applyNoise`, não do gerador.** O σ do ruído estava fixo em `(1.0, 1.0, 0.5)` dentro do
método — grão fino em z e largo em x, isto é, textura *deitada*, que é coerência alta. Virou o
atributo `noiseSigma` (padrão idêntico, conferido bit a bit) e entrou na busca. O caminho da nota,
cada passo medido em sementes novas:

| configuração | `calm` | `faulted` | `dead` |
|---|---|---|---|
| `Generator2` à mão | 84.3 | 89.1 | 72.8 |
| busca de 12/09 (21 variáveis) | 84.6 | 91.9 | 74.3 |
| + nível de ruído medido | 86.0 | 91.9 | 77.7 |
| + grão de ruído medido | 87.9 | 92.2 | 82.6 |
| + zona morta sem rótulo de falha | — | — | **83.6** |
| **referência de 13/09** | **88.0** | **92.2** | **83.6** |

A zona morta sem rótulo é o último item porque é decisão de conteúdo, não de aparência: a região
real foi definida sem nenhuma falha anotada por perto, então rótulo ali é rótulo sobre ruído — a
nota concorda (grupo `rotulo` 73.5 → 100) e o treino ganha negativo puro em vez de alucinação.

A distância até o teto (99%) é o espaço que a busca do `Analysis.ipynb` tem para trabalhar.

**Ruído da nota** — o mesmo parâmetro avaliado com conjuntos de sementes diferentes:

| tiles por lote | 8 | 12 | 15 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|
| desvio da nota | 2.4 | 1.3 | 1.1 | 1.3 | 1.2 | 0.9 |
| nota média | 86.5 | 84.8 | 84.3 | 84.1 | 84.1 | 83.6 |

Daí a escolha de **18 tiles**: dentro da faixa pedida (15–20), com ruído de ~1 ponto — menor que a
diferença entre configurações que interessa distinguir — e custo de 30–60 s por avaliação. Abaixo de
12 tiles o ruído dobra e a nota ainda fica enviesada para cima (com 8 tiles ela dá 2.4 pontos a mais
que com 24, porque sobra pouca amostra para o termo `C` da fórmula). Dentro da busca as sementes são
**fixas**, então a mesma configuração devolve sempre a mesma nota e o CMA-ES vê uma superfície
determinística; no fim, a melhor configuração é reavaliada com sementes novas para conferir que a
nota não era sorte daquele conjunto.

## 8. Como a busca usa a nota

O `Analysis.ipynb` maximiza esta nota com CMA-ES (`NatureSelector('genetic', …)`), uma região por
célula. Três detalhes fazem a diferença entre a nota funcionar ou não como objetivo:

- **só 24 variáveis, as que separam uma região da outra.** Os padrões do `SyntheticGenerator` são a
  configuração boa do `dataset_74` e `set()` troca só as chaves passadas, então a busca mexe em
  estratigrafia, dobramento, wavelet, ruído (nível e grão), quantidade de falha e no jitter de
  contraste, mais o `foldAspect` — o alongamento da dobra no eixo da inline, que é a única alavanca
  do gerador sobre a anisotropia lateral: sem ele `sigmaX` e `sigmaY` saem da mesma faixa com `theta`
  sorteado, e o campo de dobra fica isotrópico por construção. O bloco de falha (rejeito, mergulho, rugosidade, arrasto, curvatura, espessura e limiar
  do rótulo), o `foldBaseShift`, o `shearOffset` e o `waveletDt` ficam no padrão: são iguais nas três
  regiões e já estão calibrados. Cada dimensão que sai da disputa é orçamento que sobra para as que
  importam — e a seção 10 mostra a medição que justifica deixar o bloco de falha de fora. No
  `dead` o `faultCount` também sai da disputa (`FIXED`), fixo em (0, 1): a régua **premia** pôr
  falha ali, porque o crosshatch de migração deixa a zona morta descontínua e falha sintética
  aproxima `discFrac`/`lineDens` — medido, o grupo `falhas` vai de 83 para 87 e a nota de 87.6
  para 89.5. Mas o rótulo viria sobre ruído que o especialista nunca anotou, e o treino ganha
  alucinação em vez de negativo puro. Os 1.9 pontos são o preço de não mentir para a rede.
- **sementes fixas na busca, sementes novas na decisão.** O lote de 18 tiles sai sempre das mesmas
  sementes, então o CMA-ES enxerga uma superfície determinística em vez de ruído — mas acaba
  aprendendo os defeitos daquele conjunto: no `calm` a busca marcou 88.0% nas sementes dela e 82.9%
  fora. Por isso a escolha final é feita num lote de sementes novas, entre **três** candidatas — o
  achado da busca, a referência e o que já estava no `regions.json` —, e o arquivo guarda as duas
  notas (`score` fora, `scoreSearch` dentro) e o `source`. Incluir o que já estava gravado é o que
  torna a campanha monótona: rodar de novo só pode melhorar.
- **ganho fora do genoma.** Resolvido por bisseção contra o σ da região antes de medir, como na
  seção 5 — uma dimensão a menos e a nota medindo textura, estrutura e falha. O sorteio de contraste
  do lote (`gainJitter`) é **normalizado pela mediana** antes da bisseção: sem isso o ganho absorve o
  viés daquele sorteio (em 18 tiles a mediana chega a 1.17) e só vale para um lote daquele tamanho —
  o dataset de 220 tiles por região sairia 13% mais fraco que o bloco real.
- **o melhor ponto conhecido é o de partida.** Mesmo com 23 variáveis, a caixa larga é espaço demais
  para poucas centenas de avaliações: partindo de um ponto aleatório, 32 avaliações chegaram a 72% no
  `calm`, contra os 84% que o genoma do `Generator2.ipynb` já dá. Por isso existe a constante
  `Generator.REFERENCE` — o melhor genoma conhecido de cada região, que começou sendo a configuração
  calibrada à mão no `Generator2.ipynb` e é corrigido sempre que uma medição controlada acha algo
  melhor (o `faultCount` do `faulted` em 12/09, o nível e o grão do ruído das três em 13/09). A caixa
  de cada região é uma vizinhança dele (`SPREAD`, por região) e ele disputa a escolha final. A pasta
  da memória leva a largura da caixa no nome: o estado do CMA-ES está em unidades reais, então
  retomar uma campanha com outra caixa partiria de uma média fora dela.
- **nenhuma configuração fora do notebook.** A `REFERENCE` é constante de classe, não arquivo: o
  notebook roda sozinho numa pasta limpa e a única coisa que ele lê de fora são os tiles reais.
  `regions.json` é saída — quando já existe, o que está lá entra como terceira candidata.

Custo: ~30–50 s por avaliação (18 tiles 128³ em 18 processos), ~60 min por chamada de 84
avaliações e por região. A memória do `NatureSelector` deixa a campanha ser retomada e estendida.

## 9. Como ler o resultado

`Similarity.groups(região, features)` devolve a nota por grupo e `Similarity.info(...)` a tabela por
feature, com o valor mediano dos dois lados. É por ela que se enxerga o que corrigir. Exemplo real
(configuração `calmo` do `Generator2`, nota 83.3):

```
amplitude 88.5 | estrutura 87.5 | rotulo 84.6 | espectro 84.3 | continuidade 78.0 | falhas 78.0

feature     sim    real   synth
discFrac    45.8   0.004  0.142   <- o sintético tem 35x mais descontinuidade que o calmo real
lag16       66.5   0.453  0.131   <- refletor sintético não se sustenta lateralmente
lineDip     68.5   78.4   66.1    <- os lineamentos sintéticos são menos íngremes
Px1         68.7   0.223  0.414   <- dobra sintética curta demais
```

Lido assim, o diagnóstico já diz o que a busca tem de fazer no `calm`: menos ruído, dobra mais larga
e camada mais contínua lateralmente.

## 10. A falha rotulada serve para treinar?

A nota mede semelhança, mas o produto final é um detector de falha. Então vale olhar a geometria do
rótulo na seção central, medida com o mesmo código, nos dois datasets que já funcionam e na anotação
do especialista:

| conjunto | fração rotulada | visibilidade | mergulho | comprimento | traços | seções sem rótulo |
|---|---|---|---|---|---|---|
| `dataset_74` (40 tiles) | 7.3% | 3.96 | 62° | 133 px | 2 | 0% |
| `dataset_wu` (40 tiles) | 7.6% | 2.57 | 69° | 136 px | 2 | 0% |
| Marlim `faulted`, especialista | 3.6% | 2.27 | 68° | 150 px | 2 | 0% |
| `faulted` calibrado | 3.4% | 17.7 | 58° | 114 px | 2 | 6% |
| `calm` calibrado | 0% | — | — | — | — | 100% |
| `dead` calibrado | 0.1% | 50.6 | 52° | 99 px | 1 | 50% |

Mergulho, comprimento e número de traços por seção já eram os mesmos nos três primeiros — a
densidade é que difere, e por um motivo conhecido: **o especialista anota só as falhas principais**,
então 3.6% é piso, não alvo.

**A nota premia a quantidade certa de falha.** Varrendo só o `faultCount` do `faulted` e mantendo
todo o resto da referência:

| `faultCount` | nota | grupo `falhas` | grupo `rotulo` | fração rotulada | seções sem rótulo |
|---|---|---|---|---|---|
| (0, 1) | 75.0 | 68 | 24 | 0% | 100% |
| (1, 3) | 88.8 | 79 | 82 | 1.9% | 11% |
| **(2, 5)** | **91.1** | **82** | **86** | **4.4%** | **0%** |
| (4, 7) | 90.8 | 80 | 85 | 5.9% | 0% |
| (6, 10) | 89.7 | 81 | 81 | 7.2% | 0% |

O máximo cai onde a densidade fica entre o especialista e o `dataset_74`, e a nota desaba 16 pontos
se a região ficar sem falha — a métrica não é enganável nesse ponto. Foi por essa medição que o
`Generator.REFERENCE` do `faulted` passou de `faultCount` (1,3) para (2,5).

**O que ainda não está certo é a visibilidade: 17.7 contra 2.3 do real.** A conta é uma razão, e o
que está fora não é o numerador, é o denominador: o fundo sintético é limpo demais (`coh50` 0.97
contra 0.93), então qualquer corte vira um degrau óbvio. Para o treino isso significa falha fácil —
o `dataset_74` está em 3.96 e o `dataset_wu` em 2.57, os dois que transferem.

**A forma da falha não conserta isso; o fundo conserta.** Medido em 13/09/2026 sobre a configuração
gravada do `faulted`, variando um botão de cada vez:

| botão | nota | `visibility` | `discSharp` |
|---|---|---|---|
| padrão (`faultRoughness` 3.54, `faultRoughSigma` 8.55) | 91.9 | 17.7 | 62.4 |
| `faultRoughness` 8 / 14 | 91.7 | 17.4 / 17.5 | 62.6 / 62.1 |
| `faultRoughness` 14 + `faultRoughSigma` 3 | 91.7 | 17.2 | 62.2 |
| `faultDecaySigma` (15, 30) | 91.5 | 16.3 | 61.1 |
| ruído (0.10, 0.35) no lugar de (0.04, 0.18) | 90.8 | **5.6** | 18.6 |

Nenhum botão de forma da falha tira a visibilidade de 17 — eles mexem no corte, e o problema está no
que está **em volta** do corte. Subir o ruído leva a visibilidade direto para a faixa do
`dataset_74`, ao custo de estourar a descontinuidade de fundo (`discFrac` 0.081 contra 0.033 do
real), que é justamente o que o grupo `falhas` pune. Foi por isso que o nível e o **grão** do ruído
entraram na busca (23 variáveis) e os botões de falha ficaram fora: a busca agora tem a alavanca
certa e a guarda que faltava em 2026-08-17 — a própria `visibility` e o grupo `falhas`, que não
deixam trocar rótulo visível por rótulo bonito sobre imagem lisa.

## 11. Limites conhecidos

- A anotação do especialista cobre **uma inline por tile** e só as falhas principais, então
  `maskFrac` é um piso, não a verdade. É por isso que o grupo `rotulo` pesa 0.10 e o grupo `falhas`,
  que não depende de rótulo, pesa 0.20.
- A régua é de segunda ordem: compara distribuições de atributos, não a aparência completa. Duas
  imagens com as mesmas 37 features podem ainda ser diferentes ao olho — por isso o notebook fecha
  com uma comparação visual lado a lado.
- **A nota é o que se quer parecer, não necessariamente o que se quer treinar.** O caso concreto é a
  `visibility` do `faulted`: o real marca 2.3 porque o especialista desenha a falha onde a
  interpretação manda, não onde a semblance quebra. Perseguir 2.3 até o fim seria rotular falha
  invisível. O `dataset_74` (3.96) e o `dataset_wu` (2.57), que são os dois que transferem, dizem
  qual é a faixa sadia — use-a como banda de sanidade ao ler o resultado, não como alvo da busca.
- O teto de 99% e não 100% é o ruído de amostragem de 18 tiles, não um defeito do gerador.
- A nota é relativa ao conjunto real usado como referência. Trocar o critério de separação das
  regiões (`../Marlim/Analysis.ipynb`) muda a escala.

## 12. Fontes

Verificadas durante este trabalho:

- Rizzo, M. L. & Székely, G. J. (2016). *Energy distance*. **WIREs Computational Statistics** 8(1),
  27–38. doi:10.1002/wics.1375 — definição de $D^2$, do coeficiente normalizado $H$ (p. 29) e da
  equivalência com a distância de Cramér em 1D.
- Xu, Q., Huang, G., Yuan, Y., Guo, C., Sun, Y., Wu, F. & Weinberger, K. (2018). *An empirical study
  on evaluation metrics of generative adversarial networks*. arXiv:1806.07755 — MMD e 1-NN são as
  métricas que satisfazem as propriedades desejáveis, **desde que num espaço de features adequado**.
- Bińkowski, M., Sutherland, D. J., Arbel, M. & Gretton, A. (2018). *Demystifying MMD GANs*.
  ICLR 2018, arXiv:1801.01401 — KID, estimador não enviesado de MMD².
- Lopez-Paz, D. & Oquab, M. (2016). *Revisiting classifier two-sample tests*. arXiv:1610.06545 —
  C2ST: acurácia próxima do acaso sob a hipótese nula.
- Quesada, J. et al. (2025). *A large-scale benchmark on geological fault delineation models:
  domain shift, training dynamics, generalizability, evaluation and inferential behavior*.
  arXiv:2505.08585 — o benchmark **não** usa métrica formal de domain shift entre sintético e campo;
  compara desvio-padrão de intensidade e densidade de falha de forma qualitativa. É a lacuna que
  esta função preenche.
- Wu, X., Liang, L., Shi, Y. & Fomel, S. (2019). *FaultSeg3D: using synthetic data sets to train an
  end-to-end convolutional neural network for 3D seismic fault segmentation*. **Geophysics** 84(3),
  IM35–IM45 — a referência do treino com sintético que este projeto segue.

Base bibliográfica dos atributos (referências clássicas da área, não refetchadas agora):

- Bahorich, M. & Farmer, S. (1995). *3-D seismic discontinuity for faults and stratigraphic
  features: the coherence cube*. **The Leading Edge** 14(10), 1053–1058.
- Marfurt, K. J., Kirlin, R. L., Farmer, S. L. & Bahorich, M. S. (1998). *3-D seismic attributes
  using a semblance-based coherency algorithm*. **Geophysics** 63(4), 1150–1165.
- Van Vliet, L. J. & Verbeek, P. W. (1995). *Estimators for orientation and anisotropy in
  digitized images*; Fehmers, G. C. & Höcker, C. F. W. (2003). *Fast structural interpretation with
  structure-oriented filtering*. **Geophysics** 68(4), 1286–1293 — tensor de estrutura.
- Hale, D. (2013). *Methods to compute fault images, extract fault surfaces, and estimate fault
  throws from 3D seismic images*. **Geophysics** 78(2), O33–O43.
- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B. & Smola, A. (2012). *A kernel two-sample
  test*. **JMLR** 13, 723–773; Sejdinovic, D., Sriperumbudur, B., Gretton, A. & Fukumizu, K. (2013).
  *Equivalence of distance-based and RKHS-based statistics in hypothesis testing*. **Annals of
  Statistics** 41(5), 2263–2291 — energia e MMD são a mesma estatística com o kernel de distância.
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. & Hochreiter, S. (2017). *GANs trained by a
  two time-scale update rule converge to a local Nash equilibrium*. NeurIPS 2017 — FID.
- Haralick, R. M., Shanmugam, K. & Dinstein, I. (1973). *Textural features for image
  classification*. **IEEE Trans. SMC** 3(6), 610–621; Portilla, J. & Simoncelli, E. P. (2000).
  *A parametric texture model based on joint statistics of complex wavelet coefficients*.
  **IJCV** 40(1), 49–70 — a ideia de descrever textura por um conjunto de estatísticas.

Dentro do projeto: `../Marlim/Analysis.ipynb` (como as três regiões reais foram separadas),
`../../Generator2.ipynb` (a régua de 40 features que deu origem a esta) e a lição de 2026-08-17
sobre calibrar falha pela estatística da máscara.
