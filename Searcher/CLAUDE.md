# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este diretório

`Searcher/` é o guarda-chuva de busca/otimização dentro do projeto `Falhas` (detecção de falhas sísmicas).
Hoje ele contém **um único código real**: `Nature/`, um framework de otimização metaheurística (disciplina
de Computação Natural, Mestrado) com cinco algoritmos independentes de mesma interface pública,
comparados no `Nature/Analysis.ipynb` sobre funções de benchmark.

Estado de reestruturação em andamento — leia antes de assumir qualquer coisa:
- `Searcher/index.py` e `Searcher/Analysis.ipynb` estão **vazios** (0 byte): esqueleto recém-criado, ainda
  sem conteúdo. Não existe nada acoplando o `Nature` ao restante do projeto `Falhas` por enquanto.
- `Nature/` é um **repositório git próprio**, aninhado no repositório `Falhas`. Commits do framework vão no
  repo interno; não misture com o repo de fora.
- O conteúdo antigo de `Searcher/` (`Dataset/`, `Model/`, `Model/Backup/model_*`) foi apagado na árvore de
  trabalho do repo `Falhas` mas ainda está rastreado — daí o `git status` gigante de deleções. Isso é
  intencional; não "restaure" esses arquivos.
- No repo do `Nature`, `CLAUDE.md`, `CODE_STYLE.md`, `prompt.md`, `Processing/index.py` (barril antigo),
  `test_memory.py` e `test_problem.py` aparecem como deletados; `Processing/Randomizer/` aparece como
  untracked. Também intencional: o barril virou folder-modules e o guia de estilo subiu para
  `Searcher/CODE_STYLE.md`.

## Comandos

- **Ambiente: `~/anaconda3/bin/python`** (conda `base`, primeiro no `PATH`; **não** existe `~/miniconda3`
  nesta máquina). Não há `environment.yml`/`requirements.txt`. Pacotes usados: `numpy`, `pandas`,
  `matplotlib`, `scipy`, `tqdm`, `deap`; `multiprocess` é opcional (se ausente, `backend='process'` cai para
  `multiprocessing` com warning — reprodutibilidade via `seed` não é garantida nesse modo).
- ⚠️ **`deap` não está instalado em nenhum env desta máquina.** `Models/Genetic` e `Models/GeneticPSO` são os
  únicos que a importam e falham no import; `pso`, `differential` e `adaptive_de` rodam normalmente.
  Para usar os dois primeiros: `pip install deap`.
- Não há linter nem build. **Não há mais testes automatizados** (`test_memory.py`/`test_problem.py` foram
  removidos). A validação é rodar `Nature/Analysis.ipynb` célula a célula, ou headless:
  ```
  cd Nature && ~/anaconda3/bin/jupyter nbconvert --to notebook --execute Analysis.ipynb --output /tmp/out.ipynb
  ```
- Smoke test de um algoritmo isolado — sempre a partir de `Nature/`, que é a raiz de import (a `objective`
  recebe o dict de variáveis já decodificadas, não um vetor):
  ```
  cd Nature && ~/anaconda3/bin/python -c "
  from index import NatureSelector
  import math
  opt = NatureSelector('pso', {'objective': lambda ind: math.sin(ind['x']),
                               'variables': {'x': {'bounds': (0, 10)}}, 'iterations': 20, 'verbose': False})
  print(opt.info())"
  ```

## Arquitetura do `Nature`

### Ponto de entrada — `Nature/index.py`
`NatureSelector` é a fachada usada pelos notebooks: recebe `name` (`'genetic'`, `'pso'`, `'genetic_pso'`,
`'differential'`, `'adaptive_de'`) + `params` (dict) + `memory=None` (atalho que só entra no dict de params),
instancia o algoritmo via `get()` e delega `update()` / `plot()` / `info()`. Nome desconhecido levanta
`ValueError` com a lista de `NAMES`. A fachada expõe `best`/`score`; o modelo por baixo expõe
`best`/**`bestScore`** — nomes diferentes para o mesmo valor, atenção ao mexer nos dois níveis.

### Núcleo compartilhado — `Processing/<Componente>/index.py`
Cada peça do núcleo é um folder-module próprio, importado pelo caminho completo
(`from Processing.Problem.index import Problem`) — **não existe barril** `Processing/index.py` nem
`Models/index.py`. Os cinco modelos **não herdam** de nada — usam o núcleo por composição. A dependência é
de mão única (`Models` → `Processing`), nunca o contrário.

- `Problem` — decodifica genes (`real`/`int`/`cat`/`bool`), aplica bounds/constraints, chama a `objective` do
  usuário; guarda `low`/`up`/`weights` (sinal de maximize) por variável. `valid()` normaliza `nan`/`inf` da
  objective para a mesma penalidade de restrição violada (senão um único `nan` vira o `argmax` e fica como
  "melhor solução" para sempre) e avisa uma vez por problema.
- `Pool` — abre `map` / `ThreadPoolExecutor` / `multiprocess.Pool` conforme `workers`/`backend`. Avisa quando
  `backend='process'` roda com BLAS multithread: cada worker abre o próprio pool do OpenBLAS e a disputa
  custa até 10× (medido: 10,3 s contra 1,0 s com `OPENBLAS_NUM_THREADS=1`). Não dá para corrigir em runtime —
  o OpenBLAS lê a variável ao inicializar, antes de o `Pool` existir.
- `Stopper` — early stopping por paciência sobre um sinal escalar ponderado. `get()`/`set()` entram no
  checkpoint: sem eles a retomada reiniciava a contagem de estagnação e parava em `checkpoint + patience` em
  vez do ponto real.
- `Recorder` — histórico por geração/iteração (`min`/`max` da população) + nuvem de amostras (geração →
  genoma → métrica) para os scatters do `Plotter`. **`sample()` copia com `np.take`**: os modelos
  sobrescrevem `X`/`raw` in-place e guardar uma view fazia todas as gerações mostrarem a população final — só
  não aparecia com população acima de `perGen`, onde o sorteio já copiava. A geração de cada bloco vem do
  `record()`, que todo modelo chama antes do `sample()`. `get()`/`set()` serializam esse histórico para a
  `Memory` (mesmo par do `Randomizer`); `record()` rebobina sozinho se a geração já existe. A nuvem tem teto
  de `BUDGET` floats no total (`perGen` cai conforme a dimensão): ela entra inteira no `state.npz` a cada
  `Memory.EVERY` gerações e sem esse teto virava 97% do checkpoint em alta dimensão. Até ~62 variáveis nada muda.
- **Retomada não re-grava a geração do checkpoint** — todo modelo só chama `recorder.record()`/`sample()`
  antes do loop quando `state is None`. O histórico salvo já termina exatamente nessa geração, com os valores
  originais; re-gravar trocava o `nevals` do AG (que conta só os filhos inválidos) e duplicava a amostra da
  nuvem a cada queda.
- `Plotter` — retrato de **um** run: painel de convergência (scatter dos indivíduos avaliados, `y` = métrica e
  `x` = geração, com a linha do melhor até aqui) + scatter por variável (mono-objetivo) ou fronteira de Pareto
  (multiobjetivo); só é chamado depois de `update()`. Contra saturação: indivíduo penalizado
  (`|f| >= Problem.WORST`) fica fora de todos os painéis e a escala vira log quando a métrica cai ordens de
  grandeza (`SPAN`) — em maximização, onde a métrica cruza o zero, o eixo continua linear.
- `Comparer` — retrato de **vários** modelos no mesmo problema: barras do erro/aptidão e, quando a função é 2D,
  contorno + superfície 3D com a solução de cada um marcada. Delega o scatter por variável ao `Plotter` do
  vencedor. É o que os notebooks usam para responder "qual modelo é o melhor".
- `Memory` — persistência opcional (`memory=<pasta>`), pensada para runs de semanas que sobrevivem a um
  desligamento: `state.npz` (estado completo — população, auxiliares, RNG, contador `done` **e o snapshot do
  `Recorder`** — gravado a cada `EVERY=10` gerações com `os.replace` atômico, nunca apagado), `best.json`
  (melhor global entre campanhas + params) e `history.json` (um registro por run: `run`, `at`, `model`,
  `score`, `improved`, `stopped`, `best` e `params` aninhados; vira tabela com
  `pd.json_normalize(json.load(open(...)))`). Guarda o `recorder` recebido no `__init__` para embutir o
  histórico no dump — é isso que faz o `plot()` mostrar a curva inteira (0..N) e não só o trecho posterior ao
  restart. O estado é **opaco**: cada algoritmo empacota `(arrays, meta)` e a `Memory` só persiste e valida
  (mesmo problema, mesmo modelo).
- **`plot()` sem `update()`** — se o modelo ainda não rodou nesta sessão mas `memory=` aponta para uma pasta
  com campanha salva, o `plot()` de todo modelo monta um `Recorder` vazio, deixa o `Memory.start()` enchê-lo a
  partir do `state.npz` e pega `best`/`bestScore` do `best.json`. A memória é autossuficiente: dá para abrir o
  notebook num PC novo e desenhar o retrato de uma campanha de semanas sem reavaliar nada. Sem pasta salva,
  continua levantando `RuntimeError`.
- **Retomar vs. estender** — `Memory.origin(done)` devolve a origem do ciclo atual, decidida pela marca
  `ended` que só o `commit()` grava (uma queda de energia nunca chega lá):
  - estado **sem** `ended` (o processo morreu no meio) → `origin = 0`: a chamada nova só fecha o `generations`
    original, no cronograma original, e sai bit-idêntica a um run nunca interrompido;
  - estado **com** `ended` (campanha fechada, inclusive por early stopping) → `origin = done`: a chamada nova é
    uma **extensão**, um ciclo próprio de `generations` a partir dali — 0→100, 0→200, 0→300.

  Todo modelo lê `origin = self.memory.origin(gen0)` logo depois do `memory.start()` e daí tira duas coisas: o
  horizonte (`self.recorder.span = self.stopped = origin + self.generations`) e a fase do cronograma
  adaptativo, sempre `(gen - origin - 1) / (self.generations - 1)` — progresso dentro do ciclo, nunca do
  total. É isso que preserva as duas propriedades ao mesmo tempo: retomada bit-idêntica (origin=0 reproduz a
  fórmula original) e extensão como ciclo novo com eta/inércia reiniciados.

  No `AdaptiveDE` o orçamento é avaliação e não geração, então `origin` se aplica ao `nfe`/`maxNfe` e o LPSR
  usa `nfe - origin`. Uma extensão ali **reinfla a população** de volta a `nInit` (os sobreviventes viram
  elite): sem isso o LPSR já a teria reduzido a `N_MIN=4` e o ciclo novo rodaria inteiro com 4 indivíduos, que
  é busca degenerada.
- `Randomizer` — RNG determinístico das duas fontes (`random` alimenta os operadores da DEAP, `np.random` os
  genomas): `reset()` semeia e devolve o `rng`, `get()`/`set()` fazem o snapshot/restauração que torna a
  retomada bit-idêntica.

### Modelos — `Models/<Nome>/index.py`
Cada pasta é um módulo isolado e self-contained (`Genetic`, `PSO`, `GeneticPSO`, `Differential`,
`AdaptiveDE`), importando só os componentes de `Processing/` que precisa. Os nomes das classes **não** seguem
o nome da pasta (`Genetic` → `GeneticOptimizer`, `Differential` → `DifferentialEvolution`); o mapa
autoritativo é o `NatureSelector.get()`. Mesma API em todos: `__init__` leve (guarda params, valida),
`setup()` (prepara estruturas dependentes do `problem`), `update()` (roda a otimização, popula
`self.best`/`self.bestScore`/`self.recorder`), `plot()`. `update()` sempre segue o mesmo esqueleto:
`Randomizer.reset() → Recorder/Memory/Stopper/Pool → tenta retomar de Memory.start() → origin() → loop
principal com early stopping e Memory.save() por iteração → Memory.commit() → return best, bestScore`, com o
`pool.stop()` num `finally`.

Params comuns a todos: `objective`, `variables`, `maximize=True`, `patience=None`, `constraints=None`,
`seed=None`, `memory=None`, `workers=1`, `backend='thread'`, `verbose=True`. O orçamento e os params próprios
mudam por modelo — **`pso` é o único que usa `particles`/`iterations`**, os outros quatro usam
`population`/`generations`:
- `genetic` — `population=100`, `generations=200`, `crossover=0.7`, `mutation=0.2` (soma <= 1, é `varOr`).
  Único que aceita multiobjetivo (vira NSGA-II). **Precisa de `deap`.**
- `pso` — `particles=50`, `iterations=200`, `inertia=(0.9, 0.4)`, `cognitive=2.0`, `social=2.0`,
  `topology='global'|'ring'`.
- `genetic_pso` — `population=100` (>= 4, metade AG metade PSO), `generations=200`, `crossover=0.9`,
  `mutation=0.3`. **Precisa de `deap`.**
- `differential` — `population=50`, `generations=200`, `F=(0.5, 1.0)`, `CR=0.9`,
  `strategy='current-to-best/1/bin'` (`base/diffs/cx`, validada no `__init__`). O default **não** é o
  `rand/1/bin` canônico do Storn & Price: das cinco estratégias medidas na suíte CEC (5 sementes, 70k
  avaliações), `current-to-best/1/bin` ganha em rank médio em D=10 (1,19) e D=30 (1,63) contra 3,38/3,25 do
  antigo `rand/1/exp`, inclusive nas multimodais.
- `adaptive_de` — `population=None` → **`18·nVars`**, `generations=300`, `variant='lshade'|'shade'|'jade'`,
  `pbest=0.11`, `archive=2.6`; o orçamento vira `maxNfe = population · generations`.

`workers=-1` usa `os.cpu_count()`. Restrição violada não é rejeitada: `Problem` devolve a penalidade `±WORST`
(`1e12`) no lugar da aptidão, então a `objective` do usuário nem chega a ser chamada.

Para adicionar um modelo novo: criar `Models/NomeNovo/index.py` seguindo esse esqueleto e registrar em
`NatureSelector.get()` (em `Nature/index.py`) — só isso, sem herdar de nada existente.

### Notebook — `Nature/Analysis.ipynb`
Único notebook com conteúdo, em duas metades. A primeira é a bateria de testes de referência da API (seno 1D,
tipos mistos `int`/`cat`/`bool`/`real`, restrições com penalização, multiobjetivo NSGA-II só no `genetic`). A
segunda é o benchmark CEC'14: a classe `Cec` (definida no próprio notebook) monta cada função com
deslocamento, rotação e viés, guarda o ótimo conhecido e o erro vira a métrica; os exercícios rodam em `D=2`
(com relevo 2D pelo `Comparer`) e a suíte inteira em `D=10` sobre várias seeds, comparando os cinco
algoritmos pelo rank médio. As funções CEC usam `backend='vector'`.

Notebooks importam só `from index import NatureSelector` e, quando comparam, `from Processing.Comparer.index
import Comparer` — sempre com o cwd em `Nature/`.

`Nature/files/` guarda os PDFs de referência (GA/PSO/ACO/DE) e recebe as saídas de memória geradas pelo
notebook (`files/test_genetic/`). Não há `.gitignore` no repo do `Nature` — `files/test_genetic/` e os
`__pycache__/` aparecem como untracked; não commitá-los.

## Invariantes que já quebraram e não têm mais teste guardando

Os testes automatizados foram removidos. Ao mexer em `Memory`/`Recorder`/`Randomizer`/`Stopper` ou no loop de
qualquer modelo, verificar à mão:
- **Retomada bit-idêntica** — matar o processo no meio e retomar tem que dar igualdade **exata** (não
  `np.isclose`) em score, trajetória, `nevals` e geração de parada contra um run ininterrupto.
- **`nan`/`inf` da objective nunca viram a melhor solução**, nos 5 modelos e em todos os backends.
- **Bordas não quebram nenhum modelo**: variável fixa (`low == up`), tipos degenerados, restrição inviável,
  `generations=1`, `patience=1`.
- **Índices do doador da DE** mutuamente distintos e distintos do alvo.

## Convenções específicas deste projeto

- O guia de estilo canônico é `Searcher/CODE_STYLE.md` (cópia de `~/Documents/Prompts/CODE_STYLE.md`) — ler
  antes de gerar ou refatorar código aqui.
- **Exceção local ao "não comente"** do `CODE_STYLE.md`: no `Nature` toda classe abre com um comentário-cabeçalho
  em PORTUGUÊS MAIÚSCULO dizendo o que ela é e quem a chama, e as decisões não óbvias (com os números que as
  motivaram) ficam em comentários curtos no ponto exato. Mantenha esse padrão ao editar — é a documentação do
  framework.
- Mensagens de erro/warning voltadas ao usuário (`raise ValueError(...)`, `warnings.warn(...)`) e os
  comentários são em português; nomes e lógica seguem em inglês, como no resto do código.
- Todo `plot()` do projeto aceita `save=None` — a cadeia `NatureSelector` → modelo → `Plotter` e também o
  `Comparer`; quando dado, grava com `dpi=150, bbox_inches='tight'`.
- Todo algoritmo é mono-objetivo e levanta `ValueError` se `problem.multi`, exceto `genetic` (vira NSGA-II
  quando `maximize` é lista/tupla com mais de um elemento). `memory=` também é mono-objetivo.
- `variables` é um dict `{nome: {'type': 'real'|'int'|'cat'|'bool', 'bounds': (lo, hi)} | {'values': [...]}}`;
  `type` tem default `'real'`.
- `backend='vector'` só é válido com variáveis reais (a `objective` recebe o array `(N, nVars)` inteiro) e
  ignora `workers`.
