# Code Style Guide — Klauss Marchi

> Universal style reference for AIs and collaborators. Applies to every language and project.
> Read this before generating, reviewing or refactoring code.
>
> Gold standard extracted from real projects: `Etilometro/hardware/Main` (embedded C++, ESP32),
> `DataScience/Telecom` and `DataScience/Alcohol` (notebooks + sklearn modules), `Facies` (3D deep learning)
> and `MRU/Calibration/Table` (signal analysis and instrumentation). Every example below comes from these projects.

---

## 1. Language and Communication

| Context                                   | Language   |
|-------------------------------------------|------------|
| Code (names, logic)                       | English    |
| Commits, PRs, issues                      | Portuguese |
| Project documentation                     | Portuguese |
| Notebook markdown (titles, theory)        | Portuguese |
| Chart titles and end-user messages        | Portuguese |
| Code comments                             | Avoid — code must be self-explanatory |

---

## 2. Naming Conventions (All Languages)

### Universal pattern

| Element                  | Pattern             | Example                              |
|--------------------------|---------------------|--------------------------------------|
| Variables                | `camelCase`         | `startTime`, `meanConc`, `newByte`   |
| Functions / methods      | `camelCase`         | `getFiles`, `plotViews`, `isOutlier` |
| Arguments / parameters   | `camelCase`         | `xData`, `maxIter`, `breakLine`      |
| Classes / structs        | `PascalCase`        | `CrossValidation`, `AlcoholSensor`   |
| Constants                | `UPPER_SNAKE_CASE`  | `TARGET`, `K_CV`, `SENSOR_PREFIX`    |
| Enum members             | `UPPER_SNAKE_CASE`  | `Status.ACTIVE`, `MSG_ORANGE`        |
| Dict / JSON keys         | `snake_case`        | `'rms_stat'`, `'batch_size'`         |
| Module files             | `camelCase` or `index.*` | `index.py`, `discoverBaud.py`   |
| Component folders        | `PascalCase`        | `Calibration/`, `Metrics/ROC/`       |

### Short, direct names

Prefer concise names. Never use a long name when a short one works:

```
✅ conc, ref, ix, n, uc, uA, coefs, calib
❌ concentrationValue, referenceEntries, indexArray, numberOfItems
```

Globally accepted abbreviations:

| Short  | Meaning          | Short  | Meaning          |
|--------|------------------|--------|------------------|
| `n`    | count / length   | `msg`  | message          |
| `ix`   | index / indexes  | `err`  | error            |
| `ref`  | reference        | `res`  | result/response  |
| `src`  | source           | `req`  | request          |
| `dst`  | destination      | `tmp`  | temporary        |
| `cfg`  | config           | `cb`   | callback         |
| `ctx`  | context          | `fn`   | function         |

### Canonical data science names

Always use exactly these names — never variations:

```python
df                              # main DataFrame
df_categ, df_numeric, df_train  # derived variants (df_ prefix)
xData, yData                    # full features and target
xTrain, yTrain, xTest, yTest, xVal, yVal
TARGET = 'Churn'                # target column name, constant at the top
K_CV   = 5                      # cross-validation k
model, params, history, metrics
```

### Signal processing names

Preserve the field's mathematical notation — never "translate" it into verbose names:

```cpp
const float Yn = Xn*(0.196462) + Xn1*(0.137177) + Yn1*(1.010643) + Yn2*(-0.344283);
Xn2 = Xn1; Xn1 = Xn;
Yn2 = Yn1; Yn1 = Yn;
```

Exception to camelCase: when a name mirrors a library kwarg (`batch_size`, `n_components`,
`num_filters`), keep the library's snake_case for direct readability.

---

## 3. Standard Method Vocabulary

Every class uses the same vocabulary, in every language. Never invent synonyms
(`compute`, `run`, `execute`, `calculate`) when one of these applies:

| Method                | Role                                                              |
|-----------------------|-------------------------------------------------------------------|
| `update()`            | Heavy computation: fit, metric calculation, fills attributes       |
| `get()` / `set()`     | Read / write the object's main value                               |
| `info()`              | State summary as a `dict` (to export or build a DataFrame)         |
| `plot()`              | Matplotlib charts of the current state                             |
| `display()`           | Shows DataFrames with `display()` in notebooks                     |
| `print()`             | Formatted text summary; may return the internal `df`               |
| `process()`           | Intermediate transformation reused inside `update()`               |
| `setup()` / `handle()`| Embedded lifecycle: one-time init / continuous loop                |
| `ready()`             | Time or condition predicate (`if(!timer.ready()) return;`)         |
| `reset()`             | Resets internal state                                              |
| `check()`             | Verification with side effects (error screens, events)             |
| `connect()` / `disconnect()` / `send()` / `wait()` / `expect()` | Serial / network IO       |
| `export()`            | Saves results to disk (plots, json, csv)                           |
| `start()` / `stop()`  | Long-running processes, threads, training                          |

### Light `__init__`, heavy `update()`

The constructor only stores parameters and initial state. Computation lives in `update()`,
and results become public attributes:

```python
class CrossValidation:
    scoring = {'accuracy': 'accuracy', 'precision': 'precision_weighted', 'recall': 'recall_weighted'}

    def __init__(self, model, xData, yData, k=4, seed=42):
        self.model = clone(model)
        self.xData = xData
        self.yData = yData
        self.scores = {}
        self.seed   = seed
        self.k = k

    def update(self):
        self.cv = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        result  = cross_validate(estimator=self.model, X=self.xData, y=self.yData, cv=self.cv, scoring=self.scoring)

        self.scores    = [self.process(result, metric) for metric in self.scoring.keys()]
        self.accuracy  = float(np.mean(self.scores[0]['values']))
        self.precision = float(np.mean(self.scores[1]['values']))
        self.df = pd.DataFrame(self.scores)
```

---

## 4. Formatting (All Languages)

### Assignment alignment

Align `=` across blocks of related assignments:

```python
self.xData = xData
self.yData = yData
self.scores = {}
self.seed   = seed
self.k = k
```

```cpp
device.sensors.pressure.debug = false;
device.sensors.dht.debug      = false;
device.sensors.alcohol.debug  = false;
device.test.alcohol_debug     = false;
```

### One line per complete statement

If it fits in ~110 columns, it stays on one line. Never break expressions unnecessarily:

```python
# ✅
self.r2 = (1.0 if ssRes == 0 else 0.0) if ssTot == 0 else 1 - float(ssRes / ssTot)
budget.append({'conc': float(conc), 'n': n, 'uA': uA, 'uB': float(uB)})

# ❌
self.r2 = (
    (1.0 if ssRes == 0 else 0.0)
    if ssTot == 0
    else 1 - float(ssRes / ssTot)
)
```

Calls with many kwargs also stay on one line, even when long:

```python
pm.execute_notebook(path, out, kernel_name='torch-gpu', log_output=True, progress_bar=True, cwd=str(dir_path))
return SegResNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, init_filters=self.num_filters)
```

### Ternaries and guard clauses over if/else

```python
uA = float(np.std(predicted, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
yTensor = torch.from_numpy(yData).long() if MULTCLASS else torch.from_numpy(yData).float()
```

Early return instead of nesting — including returning the result of a side-effect call:

```python
def scan(self):
    ports = self.getPorts()

    if not ports:
        return sendEvent('error', 'no port found')
    ...
```

### Related short statements on the same line

Pairs and sequences that form one logical unit may share a line with `;`:

```python
plt.grid(alpha=.3); plt.legend(); plt.xlabel('time'); plt.ylabel('response')
plt.subplot(1, 2, 1); plotViews(getTest(df, 'pitch'), 'pitch', limits=(0, 1))
plt.subplot(1, 2, 2); plotViews(getTest(df, 'roll'),  'roll',  limits=(0, 1))
```

```cpp
Xn2 = Xn1; Xn1 = Xn;
Yn2 = Yn1; Yn1 = Yn;
```

### Spacing

- One blank line between logical blocks inside a function.
- Never consecutive blank lines inside functions.
- Two blank lines between a class definition and the code that uses it (notebook or script).
- No section comments like `// --- Section ---` inside methods.

---

## 5. Class Structure

### Member order (any language)

```
1. Constants / static members
2. Constructor (__init__, constructor, etc.)
3. Model / core methods (update, get, set)
4. Calculation / business logic methods
5. Utility methods
6. Visualization / output methods (plot, display, print)
7. Inference / transformation methods
8. Export / serialization methods (info, export)
```

### Constants and configuration as class properties

Configuration and reference data live as static dicts/members on the class,
never in an external file (json/yaml) nor hardcoded in the constructor:

```python
class ModelSelector:
    options = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42),
            'params': {'C': loguniform(1e-4, 1e2), 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': randint(2, 20), 'weights': ['uniform', 'distance']}
        },
    }

    def __init__(self, name):
        self.chosen   = name
        self.selected = self.options[name]
```

Reference tables (t-student, labels, protocol keys) follow the same pattern:

```python
class GaussianAnalyser:
    student = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, "infty": 1.960}
    metric_labels = ['accuracy', 'precision', 'recall', 'specificity', 'auc']
```

### Dispatch through sequential ifs with return

To select between variants, use sequential `if` + `return` — no `elif`, no `else`, no factory dict
when plain ifs read better. Close with `return None`:

```python
def get(self):
    if self.network == 'standard':
        return UNet3D(img_channels=self.channels, num_filters=self.num_filters, dropout=self.dropout)

    if self.network == 'vnet':
        return VNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes)

    return None
```

A factory that returns objects of other classes uses `__new__`:

```python
class Scheduler:
    def __new__(cls, selected, optimizer):
        if selected == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        if selected == 'cosine':
            return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        return None
```

### No unnecessary fragmentation

Never create private methods for logic used a single time. Keep it inline:

```python
# ✅ Uncertainty computed directly inside update()
def update(self):
    # fitting → metrics → uncertainty, all inline

# ❌ Fragmenting for no reason
def update(self):
    self._fit()
    self._computeMetrics()
    self._computeUncertainty()
```

Exception: logic reused in more than one place (e.g. `process()` called per metric).

---

## 6. Jupyter Notebooks (Gold Standard)

### Numbered pipeline

An analysis/ML project is a sequence of numbered notebooks, each stage saving files
the next one consumes:

```
Telecom/
├── 1 - Processing.ipynb    # cleaning, EDA, correlations → files/Model.csv
├── 2 - Model.ipynb         # model selection, CV, grid search → final model
├── 3 - Compare.ipynb       # result comparison
└── files/                  # intermediate data between stages
```

Pipeline automation through a papermill runner (`Task/index.py`), reading parameters from
`task.json` and writing an `info.json` the notebooks load at startup.

### Notebook anatomy

1. **Cell 0** — all imports grouped + global config (`pd.set_option('display.max_columns', None)`).
2. **Constants at the top** — right after loading the data: `TARGET = 'Churn'`, `K_CV = 5`, `SENSOR_PREFIX = 'target_'`.
3. **Markdown sections** — `#` titles in UPPERCASE Portuguese: `# EXTRAINDO DADOS`, `# VALIDAÇÃO CRUZADA`,
   `# GRÁFICOS`, `# SALVANDO O MODELO`. Subsections use `###`, also uppercase. Short notes as lowercase bullets.
4. **Theory in markdown** — when there is a mathematical foundation, a markdown cell with LaTeX and
   bullets explaining each term (notebook markdown stays in Portuguese):

   ```markdown
   # AVALIAÇÃO GAUSSIANA
   $$\mu \pm t\cdot\frac{\sigma_a}{\sqrt{n}}$$
   - $\sigma_a$ é o desvio padrão amostral, com $n-1$ graus de liberdade (ddof no numpy)
   - $t$ corrige a amostra pequena (tabela a 95% de confiança)
   ```

5. **Local imports** — libraries used by a single section are imported in the cell right before it
   (`from torch import amp` in the training section), not in cell 0.

### Class + usage in the same cell

A class defined in a notebook is instantiated and executed **in the same cell**, separated by two
blank lines. The whole cell is self-contained and testable:

```python
class GridSearch:
    def __init__(self, model, xData, yData, test_size=0.20):
        ...

    def update(self):
        ...


grid = GridSearch(model, xData, yData)
grid.update()
grid.grid.best_estimator_
```

### Every cell ends with visual proof

The cell's last expression shows the result — DataFrame, object, chart or a short print of shapes.
Never a "mute" cell that only defines things without showing anything (imports excepted):

```python
df = pd.read_csv('files/Model.csv')
TARGET = 'Churn'
df                                  # ← renders the table
```

```python
print('xTrain Shape: ', xTrain.shape)
print('yTrain Shape: ', yTrain.shape)
print('yTrain Unique:', unique)
```

- DataFrames: **always** the cell's final expression or `display(pd.DataFrame(...))` — never `print(df)`.
- `print()` only for short facts: shapes, counts, f-string metrics (`print(f'{name}: {mean:.2f}% (±{std*100:.2f}%)')`).

### Charts

- Wide figures with side-by-side subplots: `plt.figure(figsize=(20, 5))` + `plt.subplot(1, N, i)`.
- Chained config with `;` on one line: `plt.grid(alpha=.3); plt.legend(); plt.xlabel('time')`.
- Every plot has `grid`, `title`, and `legend` when there is more than one series.
- Plot functions take `save=None`; when given, `plt.savefig(save, dpi=150, bbox_inches='tight')`.

### Promoting a class to a module

A class is born in the notebook. Once it stabilizes or gets reused in another project, it is promoted
to a folder-module with an identical API, and the notebook switches to importing it:

```
Metrics/
├── CrossValidation/index.py    # same class born in the Telecom notebook
├── GaussianAnalyser/index.py
├── ConfusionMatrix/index.py
└── ROC/index.py
```

---

## 7. Project Architecture

### Folder-as-module

Every component is a folder with `index.*` as its entry point:

```
Project/
├── Calibration/
│   ├── index.py
│   └── Analysis.ipynb
├── Monitor/
│   ├── index.py
│   ├── Device/index.py
│   └── Plotter/index.py
└── Certificate/
    └── index.py
```

### Strict OOP

All structural code lives in single-responsibility classes. Loose functions are a rare exception,
allowed only for pure file/math utilities (e.g. `utils/Files.py` with `getFiles`, `showTile`, `pasteMask`).

### Integration

Always look at the existing structure before creating anything new. New modules must follow the
pattern of existing ones — same method vocabulary, same folder hierarchy.

---

## 8. Per-Language Rules

### Python

- **No type hints or annotations** — anywhere (parameters, returns, variables).
- **No docstrings** — the method name says what it does.
- `None` as default for optional actions (`save=None`, `limit=None`, `port=None`).
- List comprehensions and `next()` when clean:
  ```python
  xTrain = np.array([np.load(path) for path in getFiles('Database/Processed/train/images')])
  self.scores = [self.process(result, metric) for metric in self.scoring.keys()]
  ```
- Named `lambda` for one-line expressions: `getStatus = lambda var: 'pitch' if var in ['pitch', 'wx', 'ay'] else 'roll'`.
- try/except with silent fallback returning `None` in IO code; errors reported via `sendEvent`/short `print`.
- Tabular results: dict → `pd.DataFrame(data)` for display.
- Environment: **conda** (`conda activate base` before running).

### C / C++ Embedded (ESP32 / Arduino)

**Component-tree architecture.** Everything is header-only (`index.h` with `#ifndef` guard), organized
as a folder tree that mirrors the device's physical composition:

```
Main/
├── Main.ino                    # minimal: creates Device, setup() delegates, loop() → tasks.handle()
├── globals/constants.h         # #defines grouped by prefix (MSG_*, BUZZER_*, TEST_*)
├── device/index.h              # root Device class, composes all objects
├── objects/
│   ├── sensors/alcohol/index.h
│   │   ├── heater/index.h      # subcomponent = subfolder
│   │   ├── calibration/index.h
│   │   └── storage/index.h
│   └── telemetry/index.h
└── utils/                      # Listener, Smoother, filters — pure reusable classes
```

**Parent injection via template.** Components receive a pointer to the root device and reach
siblings through it — no singletons, no globals:

```cpp
template <typename Parent> class AlcoholSensor{
  private:
    Parent* device;

  public:
    bool debug;
    Calibration<Parent> calibration;
    Heater<Parent> heater;

    AlcoholSensor(Parent* dev): device(dev), calibration(dev), heater(dev){}

    void setup(){
        heater.setup();
        calibration.setup();
        device->telemetry.event("$ETEV09" + storage.id.toString() + "!");
    }

    void handle(){
        calibration.handle();
        heater.handle();
        check();
    }
};
```

**`setup()` / `handle()` lifecycle.** Every component has both; the parent propagates the call to its children.

**Static local timers.** Rate-limiting with a `static Listener` inside the method, not as a member:

```cpp
void check(bool force=false){
    static Listener timer = Listener(60000);

    if(!force && !timer.ready())
        return;
    ...
}
```

**Remaining rules:**

- Public members by default (`public:` at the top); `private:` only for the parent pointer and true internals.
- Behavior flags as public members: `bool debug;`, `bool bypass;`.
- `if(...)` with no space before the parenthesis; single-statement bodies without braces, on the next line.
- Short guard bodies may stay inline: `{screens.sensorFail(); return true;}`.
- Global protocol/UI constants in `globals/constants.h` as `#define` grouped by prefix;
  class constants as `const int pin = 1;` or `static constexpr`.
- Never dynamic `String` where a fixed buffer works (`Text<20> id;`).

### JavaScript / TypeScript

- `const` by default, `let` when needed, never `var`.
- Arrow functions for callbacks and lambdas.
- Template literals over concatenation.
- Destructuring when it simplifies.
- In TS: types only where the compiler requires them, never annotating the obvious.

```javascript
const getData = async (sensorId) => {
    const res = await fetch(`${API_URL}/sensors/${sensorId}`);
    return res.json();
};
```

### SQL

- Keywords in `UPPER CASE`, tables in `PascalCase`, columns in `snake_case`.

```sql
SELECT sensor_id, AVG(analog) AS avg_reading
FROM Calibration
WHERE mgl > 0
GROUP BY sensor_id
HAVING COUNT(*) >= 9;
```

---

## 9. Comments and Documentation

### Golden rule: self-explanatory code

Well-chosen names eliminate the need for comments. If a comment was needed, the name is bad:

```python
# ✅ The name says it all
outliers = np.array([self.isOutlier(i) for i in range(len(self.xData))])

# ❌ Bad name compensated with a comment
mask = np.array([self._check(i) for i in range(len(self.xData))])  # outlier mask
```

### When to comment

- Mathematical formulas with a reference to the source document:
  ```python
  # Eq. 10 — PPTC 0001-02
  uc = float(np.sqrt(uA**2 + uB**2))
  ```
- Environment instructions at the top of executable scripts:
  ```python
  # conda activate ENV_NAME
  # pip install papermill
  ```
- Temporary workarounds with `TODO:` or `HACK:`.
- Non-obvious design decisions (one line).

### When NOT to comment

- What the code does (the name already says it).
- Section separators inside methods — in notebooks, a section is a markdown cell, never a comment.
- Docstrings — on no method or class.
- Block-closing comments (`# end if`, `// end for`).

---

## 10. Anti-Patterns (Never Do)

| ❌ Avoid                                      | ✅ Prefer                                   |
|-----------------------------------------------|---------------------------------------------|
| Obvious comments                              | Self-explanatory names                      |
| Docstrings                                    | Clear code without documentation            |
| Type hints in Python                          | No type annotations                         |
| Breaking simple expressions across lines      | One complete line (~110 cols)               |
| Private methods for logic used once           | Inline in the method that uses it           |
| Verbose names (`calculatedMeanValue`)         | Short names (`mean`, `conc`, `ref`)         |
| 4-line `if/else` for an assignment            | One-line ternary                            |
| `elif`/`else` chains for variant dispatch     | Sequential `if` + `return`                  |
| Method-name synonyms (`run`, `compute`)       | Standard vocabulary (`update`, `get`, ...)  |
| Configs in separate files (json/yaml)         | Dicts as class constants                    |
| `print(df)` in notebooks                      | Cell's final expression or `display()`      |
| Cell that defines a class without using it    | Class + instance + usage in the same cell   |
| Loose procedural code                         | Encapsulated in classes                     |
| Heavy computation in `__init__`               | Light `__init__`, heavy `update()`          |
| `var` in JavaScript                           | `const` / `let`                             |
| Singletons/globals in embedded C++            | Parent injection via template               |
| Complex anonymous functions                   | Named method on the class                   |
| Deep inheritance                              | Composition                                 |
| Trivial getters/setters                       | Direct public attribute                     |

---

## 11. Checklist Before Delivering Code

1. Are all names in English and following the convention (`camelCase` / `PascalCase` / `UPPER_SNAKE`)?
2. Is no name unnecessarily long? Does ML data use the canonical names (`df`, `xData`, `TARGET`)?
3. Do methods use the standard vocabulary (`update`, `get`, `info`, `plot`, `setup`, `handle`, ...)?
4. Are related assignments aligned? Are simple expressions on a single line?
5. No obvious comments, docstrings, type hints or section separators?
6. Is the logic encapsulated in a single-responsibility class, with a light `__init__` and heavy `update()`?
7. Are configuration constants class-level dicts/members?
8. Does the module follow the `Folder/index.*` pattern and integrate with the existing structure?
9. In notebooks: UPPERCASE markdown sections, class used in the same cell, every cell ending with visual proof?
10. In embedded: does the component have `setup()`/`handle()`, receive the parent via template and live in `folder/index.h`?
