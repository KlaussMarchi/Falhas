# 🔬 Tutorial — Algoritmo de Busca Bayesiana para Parâmetros Geológicos

## O que este script faz?

O `Task/index.py` é um **otimizador automático** que busca os melhores parâmetros para
gerar dados sísmicos sintéticos que maximizam a métrica **IoU no dataset Wu** (`iou_wu`).

Em vez de testar parâmetros manualmente, ele usa **Otimização Bayesiana** (Optuna) para
aprender com cada tentativa e propor parâmetros cada vez melhores.

---

## Arquitetura — As 4 Classes

### 1. `SearchSpace` — O Espaço de Busca

Define **23 parâmetros geológicos** que controlam a geração sintética:

| Categoria | Parâmetros | Exemplos |
|-----------|-----------|----------|
| Camadas | 2 | `layerRange`, `layerThickness` |
| Dobras | 5 | `foldCount`, `foldSigma`, `foldDamping`, etc. |
| Cisalhamento | 2 | `shearOffset`, `shearGradient` |
| Falhas | 10 | `faultCount`, `faultThrow`, `faultDipAngle`, etc. |
| Wavelet | 3 | `waveletFreq`, `waveletDuration`, `waveletDt` |
| Ruído | 1 | `noiseLevel` |

Cada parâmetro tem **limites (bounds)** que definem onde o Optuna pode buscar.
Por exemplo, `faultDipAngle_min` pode variar de 40° a 68°.

**Tipos de parâmetros:**
- `int_range` → Par `[min, max]` de inteiros (ex: `faultCount: [3, 10]`)
- `float_range` → Par `[min, max]` de floats (ex: `noiseLevel: [0.01, 0.55]`)
- `float_scalar` → Valor único float (ex: `foldDamping: 1.45`)

### 2. `HistoricalDataManager` — Gerenciador de Histórico

**Responsabilidade:** Ler os resultados anteriores em `Searcher/database/model_*/info.json`
e injetá-los no estudo do Optuna.

**Como funciona:**
1. Escaneia todas as pastas `model_*` procurando `info.json`
2. **Filtra** modelos inválidos:
   - Sem `info.json` → rejeitado
   - Sem chave `iou_wu` → rejeitado
   - `iou_wu ≤ 0` → rejeitado (trial falhou)
   - Sem `config` completo → rejeitado
3. Injeta os válidos como trials completas no Optuna
4. Rastreia quais já foram injetados (evita duplicação)

**Opção de limpeza:** `filter_and_clean(delete_failed=True)` pode deletar
pastas de modelos inválidos do disco (desativado por padrão).

### 3. `PipelineExecutor` — Executor do Pipeline

**Responsabilidade:** Executar a sequência de 4 notebooks para cada tentativa.

**Fluxo de execução por trial:**
```
1. Salva config.json     → Synthetic/config.json (parâmetros propostos pelo Optuna)
2. Salva info.json       → Task/info.json (configuração fixa de treino)
3. Executa notebooks via Papermill:
   a) Synthetic/Generate.ipynb       → Gera volumes sísmicos 3D
   b) Dataset/dataset_synthetic/Format.ipynb → Pré-processa os dados
   c) Model/Analysis.ipynb           → Treina a U-Net 3D
   d) Model/Predict.ipynb            → Avalia no dataset Wu
4. Detecta o novo model_* criado
5. Lê iou_wu do info.json resultante
```

Se qualquer notebook falhar, o pipeline para e retorna `iou_wu = 0.0`.

### 4. `StudyManager` — Orquestrador Principal

**Responsabilidade:** Coordena tudo — cria o estudo, configura os samplers,
executa o loop de otimização, e salva resultados.

---

## O Algoritmo de Busca — Como Funciona

### Otimização Bayesiana (TPE)

O coração do algoritmo é o **TPE (Tree-structured Parzen Estimator)**:

```
┌─────────────────────────────────────────────────────┐
│  Para cada trial:                                    │
│                                                      │
│  1. TPE analisa TODOS os resultados anteriores       │
│  2. Constrói um modelo probabilístico:               │
│     - "Quais parâmetros geram IoU ALTO?"             │
│     - "Quais parâmetros geram IoU BAIXO?"            │
│  3. Propõe novos parâmetros que maximizam a chance   │
│     de IoU alto                                      │
│  4. Executa o pipeline com esses parâmetros          │
│  5. Atualiza o modelo com o novo resultado           │
│  6. Volta ao passo 1                                 │
└─────────────────────────────────────────────────────┘
```

**`multivariate=True`**: O TPE modela correlações entre parâmetros.
Exemplo: se `faultDipAngle` alto + `faultThrow` baixo = IoU alto,
ele aprende essa combinação.

### Sampler Híbrido — 3 Estratégias

O script usa **3 samplers diferentes** para balancear exploração vs exploração:

| Sampler | O que faz | Quando |
|---------|-----------|--------|
| **TPE** 🧠 | Busca inteligente baseada no histórico | Maioria das trials |
| **Random** 🎲 | Busca aleatória pura | Para escapar de máximos locais |
| **CMA-ES** 🧬 | Refinamento local fino | Após 60 trials, para polir a região boa |

### Exploração Adaptativa por Fase

A taxa de exploração **muda conforme o estudo avança**:

```
Fase 1 — Descoberta (< 30 trials):
  ┃ 30% Random ████████░░░░░░░░░░░░░░░░░░░
  ┃ 70% TPE    ████████████████████░░░░░░░░

Fase 2 — Convergência (30-79 trials):
  ┃ 20% Random █████░░░░░░░░░░░░░░░░░░░░░░
  ┃ 80% TPE    ████████████████████████░░░░

Fase 3 — Refinamento (60-79 trials):
  ┃ 20% Random █████░░░░░░░░░░░░░░░░░░░░░░
  ┃ 10% CMA-ES ███░░░░░░░░░░░░░░░░░░░░░░░░
  ┃ 70% TPE    ██████████████████░░░░░░░░░░

Fase 4 — Polimento (80+ trials):
  ┃ 10% Random ███░░░░░░░░░░░░░░░░░░░░░░░░
  ┃ 10% CMA-ES ███░░░░░░░░░░░░░░░░░░░░░░░░
  ┃ 80% TPE    ████████████████████████░░░░
```

**Por que isso importa?**
- **No início**: muita exploração para não ficar preso num máximo local
- **No meio**: reduz exploração, foca em regiões promissoras
- **No final**: adiciona CMA-ES para refinamento fino + mantém 10% random
  como "seguro" contra convergência prematura

### Startup Trials

As **15 primeiras trials** do TPE são **completamente aleatórias** (internamente).
Isso garante uma base sólida antes de começar a busca inteligente.
Como já temos 62 trials históricas, o TPE começa inteligente desde a trial 1.

---

## Persistência e Resiliência

### Banco de Dados SQLite

O estudo é salvo em `Searcher/study.db`. Isso significa:

- **Fechar o terminal** → reiniciar continua de onde parou
- **Reiniciar o computador** → mesmo comportamento
- **Deletar study.db** → recria automaticamente e re-injeta os 62 modelos do disco
- **Corrupção do DB** → detecta, deleta o arquivo corrompido, recria limpo

### Dados no Disco

Os resultados reais ficam em `Searcher/database/model_*/info.json`.
Estes são a **fonte primária de dados** — mesmo que o `study.db` seja deletado,
todos os resultados são re-injetados no novo estudo.

### CSV de Histórico

`Searcher/historico_otimizacao.csv` é um log legível por humanos.
Cada trial nova adiciona uma linha (modo append, nunca sobrescreve).

---

## Como Rodar

### Pré-requisitos
- Ambiente conda `base` com `optuna`, `papermill`, `jupyter` instalados
- Estar no diretório `Task/`

### Comando

```bash
cd /home/grva-mintcave/Downloads/Falhas/Searcher/Task
conda activate base
python index.py
```

### O que acontece ao rodar:

```
1. Cria/carrega o study "falhas_synthetic_hpo_v4" do SQLite
2. Escaneia Searcher/database/ → encontra 62 modelos válidos
3. Injeta os que ainda não estão no study (primeira vez: todos os 62)
4. Mostra estado atual: "Completed: 62, Best iou_wu: 0.6618"
5. Inicia o loop de 150 trials
6. Para cada trial:
   - Seleciona sampler (TPE/Random/CMA-ES)
   - Propõe parâmetros
   - Executa os 4 notebooks (~2-4 horas cada)
   - Coleta iou_wu
   - Salva no CSV e no SQLite
```

### Parar e Continuar

- **`Ctrl+C`**: Para graciosamente, salva o progresso
- **Rodar de novo**: Continua exatamente de onde parou
- **Nada se perde**: O study no SQLite + models no disco garantem isso

### Ao Rodar de Novo, Não Perde Nada?

**Correto!** Ao rodar `python index.py` novamente:

1. O script abre o `study.db` existente → encontra o study v4 com todas as trials
2. Escaneia `Searcher/database/` → encontra os mesmos 62 models
3. Verifica que todos já foram injetados → **"No new historical trials to inject"**
4. Continua a otimização do trial seguinte

Os 62 modelos + qualquer trial nova que você rodou ficam preservados.

> **Nota sobre o study v3 → v4:** O estudo anterior era `falhas_synthetic_hpo_v3`
> com bounds mais estreitos. O novo é `falhas_synthetic_hpo_v4` com bounds corrigidos.
> Os 62 modelos do disco foram re-injetados no v4 com os bounds novos.
> Os 14 trials FAIL (que tinham iou=0.0 no v3) **não** são re-injetados —
> isso é bom porque trials com iou=0 confundem o TPE.

---

## Resultados Esperados no Longo Prazo

### Situação Atual
- **62 trials** completadas
- **Melhor IoU: 0.6618** (model_55)
- Top-5: model_55 (0.662), model_61 (0.661), model_54 (0.659), model_52 (0.654), model_33 (0.652)
- A curva de otimização está subindo mas com diminishing returns

### Projeção com o Novo Algoritmo

| Trials | IoU Esperado | O que acontece |
|--------|-------------|----------------|
| 62-80 | 0.66-0.68 | TPE refina região dos top models com bounds corrigidos |
| 80-100 | 0.67-0.70 | CMA-ES ativa, faz busca fina ao redor do melhor |
| 100-120 | 0.68-0.72 | Combinação TPE+CMA-ES converge na região ótima |
| 120-150 | 0.70-0.73 | Ganhos marginais, polimento fino |

### Por que o Novo Algoritmo Deve Melhorar?

1. **Bounds corrigidos**: Os bounds antigos **excluíam** os parâmetros dos melhores modelos!
   Exemplo: `foldDamping` estava limitado a (0.5, 1.5) mas model_33 (IoU 0.652) usava 2.43.
   Agora o TPE pode explorar essa região.

2. **CMA-ES**: Faz refinamento local que o TPE sozinho não faz bem. É especialmente
   eficaz quando já temos uma região promissora identificada.

3. **Sem trials FAIL poluindo**: O v3 tinha 14 trials com iou=0.0 no estudo.
   Esses pontos confundiam o TPE. No v4, eles não existem.

4. **Exploração adaptativa**: No v3, 20% fixo de random. Agora começa em 30%
   (mais exploração com bounds novos) e vai caindo.

### Indicadores de Saúde da Otimização

**Bom sinal:**
- IoU subindo gradualmente trial a trial
- Top-5 mudando (novos modelos melhores aparecendo)
- Parâmetros convergindo para uma região consistente

**Sinal de atenção:**
- IoU estagnado por 15+ trials → o algoritmo pode estar num máximo local
  (mas o Random de 10% + CMA-ES devem ajudar a escapar)
- Muitas trials FAIL consecutivas → possível bug nos notebooks

---

## Estrutura de Arquivos

```
Searcher/
├── Task/
│   ├── index.py              ← Script principal (este tutorial)
│   ├── tutorial.md           ← Este arquivo
│   ├── info.json             ← Config de treino (escrito a cada trial)
│   └── logs/                 ← Saídas dos notebooks (Papermill)
├── Synthetic/
│   ├── config.json           ← Parâmetros geológicos (escrito a cada trial)
│   └── Generate.ipynb        ← Notebook de geração sintética
├── Dataset/
│   └── dataset_synthetic/
│       └── Format.ipynb      ← Notebook de pré-processamento
├── Model/
│   ├── Analysis.ipynb        ← Notebook de treino
│   └── Predict.ipynb         ← Notebook de avaliação
└── Searcher/
    ├── study.db              ← Banco SQLite do Optuna (persistência)
    ├── historico_otimizacao.csv ← Log CSV legível
    └── database/
        ├── model_1/info.json ← Resultado do trial 1
        ├── model_2/info.json ← Resultado do trial 2
        └── ...               ← model_62 até agora
```
