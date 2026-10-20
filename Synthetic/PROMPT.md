# PROMPT — A synthetic generator that generalizes to the real block (Marlim first, any block next)

Written on 19/10/2026 for a fresh Claude Code session opened in `/home/grva-mint/Projects/Falhas`. You start with no memory
of earlier conversations: everything you need is in this file or in a file it points to. Files may have changed since it
was written; when the code contradicts this file, trust the code and tell the user.

---

## TL;DR (read this first, then the details)

1. **Goal.** Networks trained only on synthetic data must find Marlim's faults the way the geologist does. The lever is
   the data: a better `SyntheticGenerator` plus a GA calibration methodology that works for any block. Networks, the
   training recipe, the evaluation and self-training are off the table (section 1).
2. **The gap, measured on 19/10/2026 (section 5.8):**
   - In the faulted interval (z 450–1200), the reference `dbrnet` draws 4× the expert's stick length, and 80% of it
     matches nothing.
   - A visual audit found most of those unmatched sticks on fold or mound flanks, onlap and pinch-out surfaces, or smooth
     reflectors. In the synthetic data, such features never occur without faults.
   - The synthetic data has 4–8× the expert's fault density.
   - It has 5× more fine-scale discontinuity.
   - Its faults are 4× sharper.
   - Its periods are 8–11 px, against 17–25 px where most of Marlim's faults are.
   - Half of every evaluated section is water or dead zone, which the training data never shows.
3. **So the first candidates are structural:**
   - realistic fault density and size;
   - unlabeled look-alikes: steep fold and mound flanks, onlaps, pinch-outs, channel margins, lenticular reflectors;
   - Marlim's period range with thin labels;
   - subtle faults with visibility-gated labels;
   - a smoother texture;
   - water, dead-zone and edge negatives.

   Several of these can be tested with no code change (Phase C).
4. **Test before adopting.** Each factor gets a controlled ablation with a fast proxy that has been validated. Decide on
   the dev patches (1200, 1300); use the test patches (1400, 2600) once, at the end.
5. **The GA calibrates ranges; it doesn't replace structure.** Its objective is label-free and must be shown to rank
   datasets by their measured transfer before any long campaign (M2).
6. **Ready-made tools** (Phase A):
   - the verified evaluation harness and the baseline numbers, in `Synthetic/files/experiments/`;
   - seven models already trained with the reference recipe on five datasets, which give full-recipe transfer for free;
   - the analysis behind item 2.
7. **The GPU may be busy** with the user's campaign. Start with CPU work (profile, generator code) and ask before
   stopping their job (section 10).
8. **Follow the user's rules** on style, structure and language (sections 1 and 11). They are strict, and past work
   was rejected for breaking them.

---

## 0. Mission

Build, and prove with measurements, a **methodology** that:

1. takes **unlabeled samples of a real 3D seismic block**,
2. learns the block's **profile** (scales, ranges, textures, structures),
3. calibrates the project's synthetic generator (`Synthetic/index.py`, class `SyntheticGenerator`) with the user's
   **genetic algorithms** (`Synthetic/Nature/`), and
4. generates a dataset on which networks trained **only on synthetic data** detect the block's faults well, close to
   what the geologist interpreted.

Marlim is the first target: it has an expert interpretation for validation. The same code must then work for any other
block by changing only its configuration. Iracema is the second block; it has no interpretation.

In the user's own words (Portuguese):

> "a ideia é a gente criar uma metodologia que a partir de amostras do dado de marlim a gente consiga através do gerador
> sintético entender o perfil dele e após um número de iterações ele vai conseguir ter os parâmetros corretos para gerar
> os dados em escalas e ranges adequados que vão englobar toda a estrutura do bloco alvo"
>
> "eu gostaria de continuar usando os algoritmos genéticos porque estou mais familiarizado com essa mesma estrutura, só
> tem que ver as métricas e metodologia por trás disso, só tem que garantir que é uma metodologia robusta, reprodutível e
> científica e que seja possível usar o mesmo raciocínio para gerar dados sintéticos para qualquer bloco"
>
> "o foco final é o modelo ser capaz de identificar as falhas no bloco de Marlim substituindo o trabalho do geólogo"
>
> "vamos centralizar tudo na pasta Synthetic, mova os arquivos de Marlim e amostras necessários pra lá"

The lever is the **data**. The networks, the training recipe, the evaluation and the stick extractor stay as they are.
The `SyntheticGenerator` class may change as much as the problem requires, within the rules of section 1.

Work like a careful research scientist:

- Form hypotheses and test them with controlled experiments.
- Measure the noise of every number you compare.
- Decide only on evidence.
- Keep checking yourself against the final goal: faults found in Marlim, as the geologist would find them.

Think deeply before acting. The costly mistakes in this project were conceptual, not coding errors: an objective that
rewarded the wrong thing ran for 21 hours, and a similarity score ran opposite to transfer. Before each phase, write
down what you expect and why. After each result, reflect before choosing the next step.

Expect this to take days of compute. Take the time it needs, explore thoroughly, and make every GPU hour count. Don't
stop at the first improvement, and don't declare success before section 14 is satisfied.

Begin with section 3, then follow section 9. Phase 0 (centralizing everything in `Synthetic/`) was already done and
verified on 19/10/2026; only its open question remains.

---

## 1. Hard rules

Each rule comes from an explicit decision of the user, and the reason is given.

1. **Training data is 100% synthetic**, produced by `SyntheticGenerator`. That excludes:
   - self-training and pseudo-labels;
   - real seismic pasted into training tiles;
   - fine-tuning on the expert's inlines.

   On 19/10/2026 the user rejected self-training even though it had improved the metric ("não pedi self train, não
   quero isso"). The goal is a generator that is good enough on its own.
2. **Don't change the networks, the training logic or the evaluation in order to improve your numbers.**
   - Networks live in `Model/Network`; the training logic is in `Model/1 - Model.ipynb` and `Model/Losses`.
   - The evaluation is `Blocks/Marlim/Analysis.ipynb`: `TileGrid`, `SlidingWindow`, `FaultStickExtractor` with
     `STICK_TOL 0.80 / STICK_MIN 140 / STICK_GAP 59`, `FaultComparer` with matching at 12 px and 20°.
   - No augmentation changes either. The user vetoed the custom `RandomScale`/`GainField`; zoom and longer-wavelet
     augmentations were tested in October and made Marlim worse.
   - If you believe something there is wrong, report it separately.
3. **`SyntheticGenerator` stays one class in `Synthetic/index.py`.**
   - The user rejected both a parallel GPU folder and a subclass: "usa um ou o outro ou junta os dois numa classe só".
   - New behaviour goes into the class itself, controlled by attributes whose defaults reproduce the current output.
     Check against `git show HEAD:Synthetic/index.py` with the same parameters and seed: the mask must be identical and
     the image within |Δ| ≤ 1e-8.
   - Run that check on several configurations, including `Dataset/dataset_74/synthetic.json`, a config without faults
     and a stress config with many listric faults. The user always asks for proof against "their" code.
   - The class defaults are the `dataset_74` configuration, chosen by the user. Don't change them without asking: the
     calibrated configuration lives in the dataset's `synthetic.json`.
4. **No new structure beyond what section 9 pre-approves, unless you ask first.**
   - Don't edit `CLAUDE.md` unless the user asks. It is current as of 19/10/2026, including the move to `Synthetic/`.
   - Don't touch what wasn't asked. These belong to the user:
     - `Model/Backup/model_1..5`, their comparison copies;
     - `model_6`/`model_7` (`fault_seg_net`/`nra_sanet`), which their running campaign will save there;
     - `Task/task.json` and `Task/info.json`.
   - In notebooks: no extra cells or markdown beyond what the work needs, and don't clear outputs.
5. **Optimization uses the user's `Synthetic/Nature/` framework.** With the repo root on `sys.path`, import it as
   `from Synthetic.Nature.index import NatureSelector`.
   - Use `NatureSelector`: `'genetic'` is CMA-ES with IPOP restarts, and `'de'`, `'lshade'`, `'lsrtde'` and `'pso'` are
     also available.
   - Don't add an optimization library.
   - Change `Synthetic/Nature/` only to fix a real bug, minimally, and report it.
6. **Never commit and never create branches** unless the user asks. They commit on `main` themselves.
7. **Language.**
   - Portuguese for everything the user reads: code comments, notebook markdown, plot titles, the journal, the final
     report and your chat messages.
   - Identifiers in English, `camelCase`.
8. **Processes.** Kill only processes you started, by explicit PID (`ps -eo pid,args | grep <x> | grep -v grep`). Never
   use `pkill -f` or `pgrep -f`: they match their own command line, and this has already killed the agent's shell.

---

## 2. What success means

### 2.1 The metric

The primary metric is the official one: **stick F1 against the expert** on the annotated inline of each Marlim patch,
as computed by `Blocks/Marlim/Analysis.ipynb`. Report recall, detection (`deteccao`: fraction of expert faults covered
≥ 50%) and precision (`precisao`) next to it.

The expert annotation is **partial**: only the main faults are drawn. That has three consequences:

- precision is a lower bound;
- F1 rewards marking less (the same NRU prediction raised to p³ goes from 0.366 to 0.407 F1 on patch 1200);
- **±0.04 F1 can come from calibration alone**.

So every comparison also reports:

- `mask`: fraction of the live section with probability ≥ 0.5 (live = local energy > 0.05);
- `F1@1%`: F1 at an equalized mask, with the threshold chosen per patch so that the mask is 1% of the live section;
- `cobertura`: fraction of the expert skeleton within 12 px of the mask;
- `foco`: fraction of the mask within 12 px of an expert fault;
- defect measures: fragments under 30 px, horizontal streaks, the first 4 xline columns, coverage per depth band.

Also look at the figures yourself; you can read PNGs. When your model marks a fault the expert did not, check the
seismic: if reflectors are visibly offset, it is not an error. A small visual audit helps here: about 20 random
"false-positive" sticks per dev patch, each classified as real fault, edge artifact, stratigraphy or noise. That
estimates the true precision better than the partial annotation does.

### 2.2 Fixed split (decided now, before any experiment)

- **dev = patches 1200 and 1300.** Every decision is taken on these.
- **test = patches 1400 and 2600.** Their seismic and their labels stay untouched until the final evaluation (Phase F).
  2600 is the most distinct patch, farthest from the others: it is the honest generalization test.
- While developing, the block profile uses only the dev slabs' seismic. After the test evaluation you may rebuild the
  profile with all four slabs for the production dataset. Report that as a separate result.

### 2.3 The reference comparison

The user compares everything against `dbrnet` with their parametrization:

```json
{"network": "dbrnet", "dataset": "<dataset>", "img_size": null, "scaling": "normalize", "lr": 0.001, "loss": "dice_focal", "batch_size": 2, "scheduler": "plateau", "dropout": 0.1, "num_filters": 32, "ema": true, "epochs": 100, "n_trials": 1}
```

On `dataset_74_wu` this run is `Documents/Backups/Artigo/model_5`; there is an identical copy in `Model/Backup/model_1`.

Baselines below come from the official method, with F1 given per patch for 1200 / 1300 / 1400 / 2600:

| model | training data | F1 per patch | mean F1 | recall | precision | mask |
|---|---|---|---|---|---|---|
| `Artigo/model_5` `dbrnet`, the reference recipe | `dataset_74_wu` | 0.293 / 0.269 / 0.241 / 0.396 | **0.300** | 0.564 | 0.210 | 4.1% |
| `Multiscale/model_7` `nru_net` | `dataset_74` (percentile, lr 1e-4) | 0.346 / 0.282 / 0.275 / 0.406 | 0.327 | 0.492 | 0.255 | 2.0% |
| `Multiscale/model_5` `macnn` | `dataset_74` (percentile, lr 1e-4) | 0.329 / 0.253 / 0.287 / 0.388 | 0.314 | 0.484 | 0.238 | 2.3% |
| mean of the NRU + MACNN + dbrnet probabilities | — | 0.396 / 0.318 / 0.309 / 0.463 | 0.372 | 0.517 | 0.298 | 1.4% |

- Per split, the reference is 0.281 on dev and 0.319 on test.
- The forbidden self-training reached up to 0.378 with a single network. That shows the ceiling is set by the data,
  not by the architectures.
- The models in `Documents/Backups/Multiscale/model_9..16` are self-trained. Don't use them as baselines.

### 2.4 Minimum bar before claiming any improvement

- The reference recipe trained on **your final dataset** beats the same recipe on `dataset_74_wu` on the **test
  patches**:
  - by more than the seed-to-seed spread you measured, with ≥ 2 seeds for yours and `Artigo/model_5` plus at least one
    re-run for the baseline;
  - at the official threshold **and** at `F1@1%`;
  - without being worse on dev.
- The gain replicates with a second architecture (`nru_net` or `macnn`, same recipe otherwise). That proves the effect
  belongs to the data.
- No new failure mode appears in the figures: water, dead zone, edges, horizontal streaks.

### 2.5 Target

A single `dbrnet` trained only on synthetic data reaching **≥ 0.37 mean F1 over the four patches**. So far that level
was reached only by ensembles or by self-training on real data. The sticks should be long and continuous enough that a
geologist would accept them.

### 2.6 Methodology success (as important as the number)

- The GA objective is label-free and was shown to rank datasets in the order of their measured transfer (section 8, M2).
- The procedure reruns from a clean checkout with fixed seeds and resumable checkpoints.
- The same notebooks run on Iracema by changing only the block configuration.

If the target is not reached, the deliverable is an honest account:

- what moved the metric, by how much, and with what uncertainty;
- what was ruled out;
- what blocks further progress.

---

## 3. Read first

Before writing any code:

- Read these files; they show where everything lives and how the user codes.
- Check the GPU state with `nvidia-smi` and the running jobs with `ps -eo pid,etime,args | grep -E "index.py|papermill|ipykernel"`.

**Code and notebooks**

| file | what to take from it |
|---|---|
| `CLAUDE.md` (auto-loaded) | pipeline, conventions and style |
| `Synthetic/index.py` | the generator, line by line; every stage is a public method |
| `Synthetic/Analysis.ipynb` | how the user inspects the generator (2D/3D views of each stage) and writes a dataset |
| `Synthetic/utils/index.py` | `formatAxis`, `showTile` |
| `Documents/Articles/fault_seg (wu).pdf` | Wu et al. 2019, FaultSeg3D, which the user calls "fault_seg_3d": the procedure the generator follows (pages 2–3). Read it with `pdftotext` or the Read tool |
| `Blocks/Marlim/Analysis.ipynb` | the official evaluation, end to end |
| `Synthetic/Regions/Marlim/Analysis.ipynb` | `MarlimBlock` (water, seafloor, dead traces, envelope, dip-steered semblance) and `RegionFinder` (calm/faulted/dead tiles) |
| `Synthetic/Regions/Synthetic/Analysis.ipynb` | the previous GA calibration, which "does not work": `ImageSimilarity`, `Calibration` (genome as center + log-width, common seeds per generation, fresh-seed final choice), `NatureSelector('genetic', ...)` with `Memory` |
| `Synthetic/Regions/Generator.ipynb` | legacy `MarlimSyntheticGenerator`: depth-varying wavelet/amplitude, water, crosshatch, domino families, visibility gate |
| `Synthetic/Nature/index.py`, `Synthetic/Nature/Processing/Problem/index.py`, `Synthetic/Nature/Models/Genetic/index.py`, `Synthetic/Nature/Processing/Memory/index.py` | how the GA is called, the `problem` dict, `mean`/`sigma`, `backend='vector'`, checkpoints |
| `Dataset/index.py`, `Dataset/dataset_74/Format.ipynb`, `Dataset/dataset_regions/Format.ipynb` | `Normalization`, how a dataset is formatted (flat or by region) |
| `Model/1 - Model.ipynb`, `Model/Network/index.py`, `Model/Losses/index.py`, `Model/EMA/index.py` | training, the networks available, and **the coding style to imitate** |
| `Task/index.py`, `Task/task.json` | how the user runs campaigns |
| `Documents/Backups/Multiscale/modifications.md` | the latest diagnosis (17–18/10/2026): why every network sits at the same level on Marlim |
| `Synthetic/files/experiments/` | the verified harness (`harness/marlim.py`), `baseline.jsonl`, and the data analysis of 19/10 (`analise_19-10/`, section 5.8). **Open its PNGs first** |

**Project memory**

The index `MEMORY.md` is auto-loaded. The notes live in
`/home/grva-mint/.claude/projects/-home-grva-mint-Projects-Falhas/memory/`; read at least these:

- `marlim-block-measurements.md`
- `marlim-escala-generalizacao.md`
- `synthetic-generator-limites.md`
- `marlim-synthetic-generator-v2.md`
- `marlim-generator2-tres-regioes.md`
- `marlim-regions-tiles.md`
- `marlim-similaridade-regioes.md`
- `marlim-regions-analysisbetter.md`
- `marlim-transfer-probe.md`
- `marlim-similaridade-rede.md`
- `calibracao-colapso-iou.md`
- `synthetic-gpu-generator.md`
- `synthetic-generator-otimizado.md`
- `multiescala-diagnostico-out2026.md`
- `marlim-stick-extractor-v6.md`
- `wu-erro-de-borda-e-capacidade.md`
- `campanha-sem-autotreino.md`
- `sem-estrutura-nova.md`
- `nature-framework-simplificado.md`
- `marlim-pipeline-contract.md`
- `iracema-pipeline.md`
- `fork-cv2-deadlock.md`
- `falhas-pipeline-automation.md`

They hold measurements that took days to obtain. Re-measure only what you will build on.

---

## 4. Map of the project (as of 19/10/2026)

```
Synthetic/            index.py (SyntheticGenerator), utils/, Analysis.ipynb, PROMPT.md
                      files/experiments/  harness/marlim.py (verified evaluation), baseline.jsonl, analise_19-10/ (section 5.8)
                      Nature/  genetic algorithms: NatureSelector -> CMAES ('genetic'), PSO, DE, LSHADE, LSRTDE; Memory checkpoints
                               (import: repo root on sys.path, then `from Synthetic.Nature.index import NatureSelector`)
                      Regions/Marlim/    Analysis.ipynb + files/{calm,faulted,dead}/*.npy + files/DataBase.csv (absolute img_path)
                      Regions/Synthetic/ Analysis.ipynb + files/cache/real_*.pkl + files/memory/calibration_22f75483/ + model/ (Unet3D trained on dataset_wu)
                      Regions/Generator.ipynb (legacy MarlimSyntheticGenerator)
Dataset/              index.py (Normalization, TilesBuilder, ...) + one folder per dataset:
                      original/{images,masks} or original/<region>/{images,masks}, synthetic.json, Format.ipynb, images/, masks/, DataBase.csv
Model/                0 - Model_CV, 1 - Model (training), 2 - Compare, 3 - Predict; Network/, Losses/, EMA/, EarlyStopping/, Transforms/, utils/
                      Backup/model_N (info.json, model.pth, train.png, predictions/)
Task/                 index.py (campaign runner), task.json (list of rounds), info.json (current round), logs/
Blocks/Marlim/        Analysis.ipynb (official prediction + evaluation)
                      files/patches/<1200|1300|1400|2600>/{tiles/*.dat + patch_metadata.json, <id>.png, <id>_interpretado.png}
                      files/cache/{inline_index.json, real_center_<id>.npy}, files/patch_metadata.json
Blocks/Iracema/       Analysis.ipynb (prediction + figures, no metric), files/patches/*.dat (38 528 tiles, 302 GB), files/patch_metadata.json, files/cache/
Documents/            Articles/ (PDFs), Backups/{Artigo,Marcia,Multiscale}/model_N (saved models), 1 - Metodologia.md, 2 - Treinamento.md
```

**Datasets that matter**

| dataset | content |
|---|---|
| `dataset_74` | 220 tiles from the generator defaults |
| `dataset_wu` | FaultSeg3D, 220 tiles |
| `dataset_74_wu` | 330 = 110 of `dataset_74` + all of `dataset_wu` (it has no images of its own) |
| `dataset_zu` | the same data as `dataset_wu`, standardized |
| `marlim_opt` | legacy Marlim generator, percentile scaling |
| `dataset_regions` | `original/` is the 03/10 CMA-ES winner, a collapsed configuration, but `images/` was formatted on 20/09 from the `dataset_74` configuration; regenerate or reformat it before using it as an anchor |

**Networks** in `ModelNetwork`: `unet_3d`, `dbrnet`, `segresnet`, `resaceunet_grva`, `resaceunet_wu`, `resaceunet_zu`,
`macnn`, `fault_seg_net`, `nru_net`, `fault_edge_former`, `nra_sanet`.

---

## 5. The Marlim block: what is known

### 5.1 Files and geometry

- **Four slabs** ("patches" 1200, 1300, 1400 and 2600), each 128 inlines × 1601 z × 2240 xlines.
  - Each is stored as 910 tiles of 128³ raw float32 (`patch_d000_hHHH_wWWW.dat`), overlap 32, stride 64, grid
    1 × 26 × 35.
  - The metadata says `original_shape [64, 1601, 2240]`, but that is only the core. All 128 inlines of each `.dat` are
    real: neighbouring inlines correlate ≥ 0.99 and there is no mirror.
  - Only the z and xline borders are reflect-padded. Tiles on the lateral edge of a patch come **mirrored around the
    dead trace**.
- **Tile axes:** (inline, z, xline), which equals the datasets' (x, z, y).
- **Scale:** [0, 1] from the block's own p01/p99. The water is the constant **0.50263**, and 1.6–3.9% of voxels are
  saturated at 0 or 1.
- **Depth domain:** 1601 samples × 5 m; xline spacing ≈ 12.5 m. An apparent dip in voxels is therefore steeper than
  the physical dip.
- **Annotated inline:** `files/patches/<id>/<id>_interpretado.png` (2240 × 1601) matches the section 1:1.
  - It is **not** the central inline: it is slice 16 / 52 / 24 / 8 of the 64-inline core (`files/cache/inline_index.json`),
    which is slice 48 / 84 / 56 / 40 of the 128-inline `.dat`.
  - `files/cache/real_center_<id>.npy` holds that annotated inline, despite its name.
- **Dead traces:** xline 0 in every patch; in 2600 also xlines 0–6 and 2238–2239.
- **Seafloor:** z ≈ 157–425, dipping east.

### 5.2 Vertical structure

| band | z (samples) | character | σ in [0, 1] |
|---|---|---|---|
| water | above the seafloor (z ≈ 160–425) | constant 0.50263 | ≈ 0 |
| shallow package | 250–600 | strong thin reflectors | 0.17 |
| weak interval | 600–800 | weak reflectors, many faults | 0.09–0.11 |
| bright folded band | 800–1200 | strong, folded | 0.15–0.20 |
| dead zone | > ~1200 | almost no reflectors; migration crosshatch at ±45°; coherence 0.64–0.68 | 0.02–0.04 |

### 5.3 Scale, contrast and noise

**Peak of the vertical spectrum** per 128-sample window, in cycles/sample (mean of 4 sections):

| z | 250 | 400 | 550 | 700 | 850 | 1000 | 1150 | 1300 |
|---|---|---|---|---|---|---|---|---|
| peak | 0.230 | 0.142 | 0.102 | 0.055 | 0.041 | 0.047 | 0.044 | 0.039 |

**Period** (spectral centroid) where faults are annotated:

| z | 200–450 | 520–600 | 650–720 | 776–1400 |
|---|---|---|---|---|
| period | ~4 px | 7–9 px | 13–16 px | **17–25 px** |

- The 776–1400 band holds most of the annotated fault length.
- In the synthetic data the period is 8.4–11.2 px in `dataset_74` (p10–p90) and 6–9 px in `dataset_wu`. **The training
  data covers about 10% of the scale where Marlim's faults are.**

**Scale test** (Fault-Seg-Net, patch 1200, real data resampled before the network):

- Each depth band is detected best when its period is brought to the training period.
- Compressing by 0.5× raises recall in z 900–1150 from 0.49 to 0.71; compressing by 0.7× raises z > 1150 from 0.62 to
  0.70.
- The shallow band collapses under compression.
- So the networks find faults only at the scale they were trained on.
- Zoom augmentation does not fix this: it thickens the labels.

**Contrast and noise**

- Contrast per real 128² crop: σ p5 / p25 / p50 / p75 / p95 = 0.013 / 0.038 / 0.126 / 0.182 / 0.282.
- Fine noise, measured as the MAD of the Haar HHH sub-band divided by the tile σ: **Marlim** 0.0105 (faulted) and
  0.0128 (calm); **synthetic** 0.056 (`dataset_74`) and 0.068 (`dataset_wu`). Marlim has much **less** fine noise than
  our synthetic data.

### 5.4 The expert's faults

Measured through `FaultStickExtractor.fromRaster`:

- 33 / 24 / 18 / 30 sticks (1200 / 1300 / 1400 / 2600); total length 7454 / 5345 / 5605 / 13087 px.
- Apparent dip p5 / p50 / p95 ≈ 40 / 65 / 77°: never below 40° nor above 80°.
- About 80% of the sticks belong to one dip family: domino families with ~20–60 px spacing and occasional conjugates.
- Median length 220–440 px, and 83% are longer than 128 px: they cross a whole tile.
- z range ≈ 250–1500.
- In 2600 (and partly in 1400) the expert extends faults into the dead zone (z > 1150) by geological reasoning, with
  almost no seismic evidence there. No network reaches those parts: coverage is 0.09–0.36.
- In 1200, 9 of the 33 faults are shallow and short (z 270–420), below what the extractor represents (minimum 140 px,
  dip 43–76°).

### 5.5 What current models do on the annotated inlines (threshold 0.5)

- They mark 2–5% of the live section; the expert's annotation covers about 0.8%.
- 60–90% of the expert's traces lie within 12 px of the mask, but only about **16–24% of the mask** lies within 12 px of
  an expert fault (`foco`; 0.29 for the ensemble). The models see most annotated faults; the problem is everything else
  they mark. Some of it may be real but unannotated, which is why you must look.
- 27–39% of the mask is in fragments under 30 px.
- 21–54% of the first four xline columns are marked in 1200, 1300 and 1400: a dead-trace and mirrored-edge artifact.
- There are horizontal streaks along strong reflectors and a parallel hatch in the dead zone.
- Synthetic test IoU (0.70–0.76) does not predict any of this. The `dbrnet` with the highest synthetic IoU (0.757) is
  among the noisiest on Marlim, with a 4.1% mask and 39% fragments.

### 5.6 Old real-tile regions

Built with the labels in `Regions/Marlim`: calm 149, faulted 188 and dead 123 tiles, median σ 0.192 / 0.146 / 0.016.
The regions were selected **with the expert's labels** (`faultLength`, `faultNear`), which a new block would not have.

The same notebook found a label-free discontinuity measure: the fraction of voxels with dip-steered semblance < 0.8 has
AUC 0.914 between tiles with and without annotated faults.

### 5.7 Iracema

- `Blocks/Iracema/files/patches/*.dat` is a flat folder of 38 528 tiles of 128³, plus `files/patch_metadata.json`.
- The block is (inline, z, crossline) = 2751 × 2000 × 1754, grid 43 × 32 × 28, [0, 1] from the block's p01/p99.
- There is no interpretation. `Blocks/Iracema/Analysis.ipynb` predicts the 896 tiles of one inline (`INLINE = 1375`) and
  draws figures, without a metric. Predicting the full block (~308 GB) does not fit on disk.
- Its sampling interval and domain (time or depth) are not recorded in the repo. Work in voxel units, which is what the
  network sees.
- **What it looks like** (`analise_19-10/iracema_1375.png`, from the cached `files/cache/real_inline_1375.npy`). It is
  quite unlike Marlim:

  | z (samples) | character | σ / saturation |
  |---|---|---|
  | above ~450 | water, flat seafloor | 0 |
  | 500–950 | strongly folded package with domes and steep flanks | 0.26 and 12% saturated in z 500–750 |
  | ~750–1000 | a transparent, possibly salt-like body above a strong reflector near z ≈ 1000 | — |
  | ~1000–1400 | layered reflectors | — |
  | below ~1500 | dead zone | 0.007–0.016 |

  A methodology that profiles the block will need the H9 mechanisms here too (steep flanks, domes, transparent bodies),
  plus saturation. It is a good portability test precisely because it differs so much.

### 5.8 What the previous session measured on 19/10/2026 (look at the data first)

Scripts, figures and the reference model's predicted sections are in `Synthetic/files/experiments/analise_19-10/`. They
run from there on CPU in about a minute each; `sticks.py` and `audit.py` take a few minutes. Look at the four PNGs before
anything else:

- `crops_1200.png` and `crops_2600.png`: native-resolution crops of the annotated inline at three depths. Each shows the
  seismic, the expert's faults, and the reference `dbrnet` mask (`Artigo/model_5`) over the expert.
- `synreal.png`: 128×128 sections of `dataset_74` and `dataset_wu` with labels, next to real faulted Marlim tiles with
  the expert's faults, at the same pixel scale.
- `audit.png`: 16 random unmatched predicted sticks from the dev patches (z 450–1200, length ≥ 150 px).

| measure | Marlim | synthetic | script |
|---|---|---|---|
| fault traces crossing a 128-px horizontal line, live interval | 0.33–0.59 (expert; a lower bound) | 2.5–2.6 (74 and wu) | `measure.py` |
| label fraction | 0.8–1.8% of the live section (expert) | 6.5–7.3% of the section | `measure.py` |
| background discontinuity (dip-steered semblance < 0.8, 5 traces, > 40 px from faults) | 0.018–0.021 | 0.108 | `measure.py` |
| discontinuity on the faults | 0.07–0.09 | 0.31–0.35 | `measure.py` |
| background discontinuity over 17 traces | 0.13–0.16 | 0.22 (74), 0.39 (wu) | `lateral.py` |
| section composition, cells of 64×64 | water 18%, dead zone 30–39%, live 42–52% | live 100% | `grid.py` |
| seafloor / dead-zone top (median z) | 252–262 / 1156–1188 | — | `grid.py` |
| expert skeleton inside the dead zone | 16–42% (2600 highest) | — | `lateral.py` |

What the table says:

- **Locally, Marlim is smooth.** It is 5× less discontinuous than the synthetic data, and its faults are 4× less sharp;
  synthetic faults are sharp cuts in a speckled background.
- **Over 17 traces, Marlim's discontinuity grows 8×, against 2× in `dataset_74`.** Its reflectors are lenticular and
  terminate on a scale of tens of pixels: lenses, pinch-outs, bright segments, lateral truncations. Synthetic reflectors
  are continuous across the whole tile.
- **Amplitude along a reflector does not vary more in Marlim than in the synthetic data** within 17 px. The lateral
  variation lives at larger scales.

Where the reference `dbrnet` mask (≥ 0.5) falls, as a fraction of the mask:

| zone | share of the mask |
|---|---|
| dead zone | 29–39% |
| live interval, more than 40 px from any expert fault | 24–43% |
| live interval, 12–40 px from an expert fault | 11–15% |
| within 12 px of an expert fault | 11–20% |
| water and dead traces | 2–3% |
| first and last 8 xlines | 3–4% |

Where the **official stick metric** loses (`sticks.py`; the reference `dbrnet` and NRU `Multiscale/model_7` agree):

- **Precision.** The extractor's energy gate already removes the dead zone: z > 1200 is only 5–8% of the unmatched
  predicted length. The loss is in z 700–950 (54–57% of it), 950–1200 (19–25%) and 450–700 (13–14%). In z 700–950 the
  `dbrnet` draws 163 sticks totalling 50,754 px, against the expert's 43 sticks and 12,848 px, and 80% of that length
  is unmatched.
- **Recall.** Within each band the miss rate is 34–46% in z 450–1200. Two groups are almost unreachable with the current
  extractor and are not worth chasing:
  - the shallow, low-angle faults near the seafloor (z < 450): 94% missed;
  - the expert's extensions into the dead zone (z > 1200): 77–84% missed.

  Together they are about 20% of the annotated length.
- **Dip.** Unmatched predicted sticks have p10/p50/p90 dips of 49/64/73°, against 57/67/74° for matched ones.

**Visual audit** of the 16 unmatched sticks in `audit.png` (one observer, rough):

- about 6 follow fold limbs, mound or diapir-like flanks, or onlap and pinch-out surfaces;
- about 4 cross smooth reflectors with no visible offset;
- 2 follow sharp lateral boundaries (a channel margin or truncation; arguable);
- 1 sits at the patch edge;
- only 2–3 look like real faults the expert left out.

So most of the precision loss is the network calling steep structural or stratigraphic boundaries "fault". In the
synthetic data, steep reflectors and terminations appear only at faults. Redo this audit more carefully, with more
sticks and per category, as part of Phase B.

Caveat: these measurements use the annotated inline only, and the dead-zone and live classes come from a simple envelope
threshold (0.0552, the Otsu value from `Regions/Marlim`). Treat the numbers as ±20%; what matters is the order of
magnitude.
- Known pre-existing bug, from the user's in-progress edit: a cell uses `self.MODEL_THRESHOLD` where the global
  `MODEL_THRESHOLD` is meant. Tell the user; don't fix it silently.

---

## 6. The generator today and what it cannot represent

### 6.1 Wu et al. 2019 (FaultSeg3D)

1. A 1D random reflectivity in [−1, 1].
2. Folding by a sum of 2D Gaussians with depth-linear damping:
   `s1 = a0 + 1.5 z/zmax Σ b_k exp(−((x−c_k)² + (y−d_k)²) / 2σ_k²)`.
3. Planar shearing `s2 = e0 + f x + g y`.
4. Planar faults with Gaussian or linear throw, 0–40 samples, more than 5 per 128³ cube and not too close together.
5. Ricker convolution after faulting, then noise.
6. A 128³ crop from a larger volume.
7. Labels 2 px across each fault.
8. 200 training + 20 validation pairs.
9. Augmentation: vertical flip plus rotations about the vertical axis.
10. Each image is z-scored.

### 6.2 `SyntheticGenerator` (`Synthetic/index.py`, ~310 lines)

The same workflow, in this order:

| stage | what it does | parameters |
|---|---|---|
| reflectivity | random layers of thickness 1–4 px | `layerRange`, `layerThickness` |
| folding | sum of 2D Gaussians with depth damping | `foldCount`, `foldSigma`, `foldAmplitude`, `foldDamping`, `foldBaseShift` |
| shearing | planar shift | `shearOffset`, `shearGradient` |
| faulting | `faultCount` planes; origin in 15–85% of the cube; dip from `faultDipAngle`; **strike uniform in [0, 2π]**; Gaussian or linear throw up to `faultThrow`; plane roughness; listric with probability `faultCurveProb` | `faultCount`, `faultDipAngle`, `faultThrow`, `faultRoughness`, `faultRoughSigma`, `faultCurveProb` |
| label | voxels with \|distance\| ≤ `faultZoneWidth` where the throw exceeds `faultThreshold` | `faultZoneWidth`, `faultThreshold` |
| wavelet | **one Ricker per tile**; peak at `freq · waveletDt` cycles/sample, with `waveletDt` 0.0012 | `waveletFreq`, `waveletDt` |
| noise | `noiseLevel` × σ, smoothed by `noiseSigma`, then a (0.5, 0.5, 0) blur | `noiseLevel`, `noiseSigma` |
| output | 128³ crop of a 256³ cube (margin 64), z-score; optional gain/jitter/clip when saving | `gain`, `gainJitter`, `clip` |

- **Defaults** = the `dataset_74` configuration.
- **GPU:** the cube is computed on the GPU in torch float64 and reproduces the old scipy implementation bit for bit.
  `fast=True` draws the noise fields in float32 on the GPU: same distribution, a different tile, ~0.7 s instead of
  2–7 s per tile. Use it in searches, never for a final dataset.
- **API:**
  - `get()` returns (z-scored image in (x, y, z), uint8 mask).
  - `set(options)` overrides attributes.
  - `info()` returns all attributes.
  - `dataset(options)` takes `{'directory', 'seed', 'regions': {name: {'n_images', 'output', 'params'}}}`; unknown keys
    raise. It makes one generator copy per region, uses seed `seed + global index`, and saves (x, z, y) through
    `transpose(0, 2, 1)` into `<directory>/<output>/{images,masks}/img_XXXX.npy`.
- **Draw order:** the order of the `np.random` calls defines the tile. Changing it changes every dataset.
- **Ranges:** every range parameter is a tuple `(low, high)` drawn per tile, and `randint` upper bounds are exclusive.
  `faultCount (0, 1)` therefore means zero faults.

### 6.3 Measured limits of the class against Marlim

These are modelling problems: no parameter search can fix them.

- **One stationary Ricker per tile.** It cannot make the period change with depth inside a tile, and the default range
  gives 8–11 px against Marlim's 4–25 px.
- **Uniform strike.** The apparent dip on a section is atan(tan δ · |sin φ|), so about 13% of faults look nearly flat.
  There is no azimuth or family control and no domino families.
- **Planar reflectors.** Folding alone never brings the structure-tensor coherence below ~0.90; Marlim's dead zone is
  at 0.64–0.68. `noiseSigma` was exposed in September to lower it, but the smooth noise that matched coherence
  destroyed transfer (section 7).
- **Missing elements:**
  - water and seafloor;
  - dead traces and mirrored edges;
  - dead zone and migration crosshatch;
  - stratigraphic discontinuities (onlap, truncation, clinoforms, channels);
  - amplitude variation with depth;
  - saturation: a z-score followed by min-max per volume leaves 0% saturated.
- **Fault visibility.** The fault-shape knobs (roughness, decay, zone width, threshold) barely change how visible a
  fault is; the background controls it. `dataset_74` faults are easier to see than Marlim's: label visibility 3.96 for
  74, 2.57 for wu and 2.27 for Marlim.
- **Small throws.** A model trained on throws of 4–26 px had recall 0.78 on synthetic control tiles, 0.21 on tiles
  with throws of 2–5 px, and 0.00 on 2–5 px flexures with drag. The real faults it missed looked like those flexures.
  But training on a log-uniform throw of 2–26 px plus drag made the task unlearnable (val IoU 0.35). Any small-throw
  fault needs a visibility-gated label.

---

## 7. What was already tried, and why it failed

Don't repeat any of these without a new reason, and say what the new reason is.

1. **Similarity of image statistics** as the GA objective (the 33–40-feature ruler with energy coefficients, 3
   regions, CMA-ES). It runs **opposite to transfer**: `dataset_regions` scored 94.4 similarity and gave F1 0.068,
   while `dataset_74` scored 81.6 and gave 0.358. Matching statistics was achieved through unrealistic mechanisms such
   as clean backgrounds, few faults and smooth noise. About 20 h of CPU produced the worst dataset of the series.
2. **KID in the bottleneck of a network trained on `dataset_wu`**, plus the fraction of detected faults, plus the
   vertical period (the current `Regions/Synthetic` notebook).
   - It only separates groups, with ~2 points of noise; the old anchors were 74 91.9, regions 91.7, wu 82.6,
     marlim_opt 74.6 and the worst data 66.7.
   - It is nearly blind to vertical zoom (zoom z × 1.5 changes the KID by 0.1 for τ ≈ 23.7).
   - The extractor sees what it was trained on.
3. **IoU of that fixed network on the synthetic labels** as an objective term.
   - The campaign `calibration_22f75483` ran 200 generations in 21 h 45; the ranges **collapsed** onto easy, wu-like
     data.
   - The result: period 8 px against 16 px real, `layerRange` (185, 195), `faultCount` (5, 6), throw 25.9–28.3.
   - All the gain came from the IoU (61 → 80); the similarity did not move (92 → 90).
   - Directions the objective cannot see drifted at random, and the elite was just the luckiest draw.
4. **Transfer probe (TSTR).** A 0.37 M U-Net was trained 2000 steps on 44 candidate tiles and scored by stick F1 on 3
   patches, ~6 min per evaluation. It is the right signal, but noisy (± 0.026).
   - 64 evaluations in a ± 25% box around `dataset_74` found nothing above noise.
   - The elite fell from 0.353 to 0.286 on fresh seeds (optimizer's curse).
   - A 2D version of the probe is useless.
5. **Matching coherence with smooth noise** was "the biggest villain": replacing it with `dataset_74`'s noise took the
   probe F1 from 0.063 to 0.285.
6. **Calibrating fault parameters by mask statistics** (fraction, length, dip) produced labels on invisible faults:
   roughness 9, throw floor 0, threshold 4.65.
7. **Log-uniform small throws (2–26) + drag 5.5** made the task unlearnable (see 6.3).
8. **Scale via augmentation.** Isotropic zoom thickens the label f-fold: mask × 4 and blobs. The longer-wavelet
   augmentation was worse. Mirrors and rotations tied.
9. **Fine-tuning on the annotated inlines**, with unannotated pixels treated as negatives, loses recall even on the
   training patches.
10. **Self-training on Iracema pseudo-labels** improved every network by 0.04–0.10 F1, but the user **rejected** it.
11. **Legacy `MarlimSyntheticGenerator`** (`marlim_opt`): depth-varying wavelet and amplitude, water, crosshatch,
    domino families, visibility gate.
    - In July it beat a Wu-type generator calibrated per band (F1 0.304 against 0.235, older extractor).
    - Later `dataset_74` passed it (probe 0.28 against 0.22).
    - Mixing `marlim_opt` into 74/wu lowered recall.
    - Some of its ideas are right; its whole package was not enough.
12. **Smaller training windows** (64³, 32³) gave fragmented predictions on Marlim, even with better synthetic IoU (there
    was also leakage in that split).
13. **Training hyperparameters** (lr, dropout, loss, epochs) are saturated on synthetic data, and synthetic IoU does not
    predict Marlim. The multiscale networks did not close the gap either; all architectures sit at 0.28–0.33.

**What did help:**

- Averaging architectures (+0.04): the real faults appear in all networks, while the false positives differ.
- Matching the period per band: the scale test in 5.3.

Both say the same thing: the networks learned the generator's world, not Marlim's.

---

## 8. Methodology requirements

These are the properties that make the method robust, reproducible, scientific and portable.

- **M1. Label-free calibration.** The GA objective may use only unlabeled seismic of the target block, so the same
  procedure works on a block without interpretation. Expert labels serve only to validate the methodology on Marlim.
  This also means that region or band definitions must not use labels: water, seafloor, dead traces, dead zone and
  discontinuity can all be detected from the seismic (5.6).
- **M2. Validate the objective before trusting it.**
  - Before any long GA campaign, show that the objective ranks datasets in the order of their measured transfer
    (TSTR F1 on the dev patches).
  - Use ≥ 8 datasets: the anchors (`dataset_74`, `dataset_wu`, `dataset_74_wu`, `marlim_opt`, `dataset_regions`) plus
    the controlled variants of Phase C.
  - Measure every anchor's TSTR yourself, with one recipe. The old numbers came from different recipes and extractor
    versions.
  - Report Spearman ρ with a bootstrap interval, and check that every pair whose TSTR differs by more than the noise is
    ordered correctly.
  - An objective that fails is not used, however principled it looks. Every past objective failed exactly here.
- **M3. Coverage, not closeness.**
  - A dataset is a distribution, so the genome encodes ranges or distributions, as the user's `Calibration` already
    does with center + log-width.
  - The objective rewards covering the target's variability per depth band and region: Marlim's whole 4–25 px period
    range, all its contrasts, dips and textures. It also penalizes synthetic mass where the block has none. Distribution
    precision/recall in the descriptor space, as in Kynkäänniemi 2019 and Naeem 2020, is one way to express this;
    design and validate your own.
  - Monitor range widths every generation. A direction the objective does not see must be fixed or measured, not left
    to drift.
- **M4. Measure what can be measured; search only what can't.**
  - Measure the period/spectrum against depth, amplitude/envelope against depth, saturation, reflector dip
    distribution, water and dead-zone fractions, fine-noise ratio, and the dip/spacing/length of fault-like
    discontinuities directly from the block. Map them to generator parameters analytically or by a quick fit.
  - Leave to the GA what has no direct measurement: fault expression, noise texture, mixture proportions, interactions.
- **M5. Capacity before search.**
  - Before running the GA, check that each target descriptor is **reachable**: sweep the generator and see whether the
    real values fall inside its envelope.
  - If they don't, extend the class. Searching harder cannot fix a missing mechanism (6.3).
- **M6. Statistics.**
  - Common random numbers within a generation: all individuals use the same seeds, as in the user's `Calibration`.
  - Fresh seeds for every final decision, paired differences with standard errors, and ≥ 2 training seeds for any TSTR
    claim near the noise.
  - Report effect sizes with uncertainty, re-evaluate elites to remove the optimizer's curse (Smith & Winkler 2006),
    and size each GA batch so that its noise is well below the differences you care about.
- **M7. Reproducibility.**
  - Everything is seeded.
  - `synthetic.json` fully specifies a dataset, and regenerating from it is bit for bit.
  - GA campaigns are resumable (`Memory`, with `Memory.EVERY = 1` as the user sets it), and the memory folder name
    hashes everything that defines the problem.
  - Notebooks run top to bottom.
  - Log the git hash and the config of every experiment.
- **M8. A strict held-out test** (section 2.2). Dev patches drive every decision; the test patches are used once, at the
  end.
- **M9. Portability.** Block-specific values live in a block configuration and in the measured profile, never
  hard-coded in the generator. The same notebooks must run on Iracema, whose tile layout differs from Marlim's: read it
  through `patch_metadata.json` and the `TileGrid` logic.
- **M10. Interpretability.** Every calibrated range has a physical justification: say what in the block it matches. The
  final report shows the profile match per descriptor and per depth band, real against synthetic.

---

## 9. Plan with gates

The plan is a strong default, not a cage. If evidence contradicts it, change it and write down why. Every phase ends
at a gate: check it, write the result in the journal (section 12), and send the user a short message in Portuguese.

**Suggested order for the first two days**, so that neither the GPU nor you sits idle:

1. **First hours:**
   - Read section 3, look at the figures of 5.8, and check the GPU and the running jobs.
   - Create the journal.
   - Ask the user the Phase 0 question (`unet3d_v2`).
2. **If the GPU is free, start these and keep them queued:**
   - verify the harness;
   - evaluate the 7 anchor models on dev (~2 h);
   - train the proxy on 3–4 anchor datasets (Phase A).
3. **Meanwhile on CPU:**
   - the label-free sampler and profile of the dev slabs, with the gap table (Phase B);
   - the first generator features, off by default and with the bit-identical proof: H9 look-alikes, H1 log-uniform
     period, H4 visibility-gated subtle faults.
4. **As soon as the proxy is validated:** queue the no-code ablations of Phase C overnight.
5. **Day 2:** read those results, ablate the new features, and build candidate C0.

### Phase 0 — Centralize in `Synthetic/` (done and verified on 19/10/2026)

The previous session did this at the user's request. Don't redo it.

**Moved with `mv`:**

- `Blocks/Marlim/Regions/` became `Synthetic/Regions/`.
- `Blocks/Marlim/files/Generator.ipynb` became `Synthetic/Regions/Generator.ipynb`.
- `Nature/` (repo root) became `Synthetic/Nature/`.

**Paths fixed:**

- `Regions/Marlim`: `PATCH_DIR = '../../../Blocks/Marlim/files/patches'`.
- `Regions/Synthetic`:
  - `sys.path.append('../../..')`;
  - `from Synthetic.Nature.index import NatureSelector`;
  - `from Synthetic.Nature.Processing.Memory.index import Memory`;
  - `DATASET = '../../../Dataset/dataset_regions'`.
- `Generator`: `PATCH_METADATA`, `CACHE_DIR`, `MARLIM_DIR` and `DATASET_DIR` now point to `Blocks/Marlim/files/...` and
  `Dataset/marlim_opt/original`. Its cache is still the one `Blocks/Marlim/Analysis.ipynb` uses.
- The 460 absolute `img_path` values in `Regions/Marlim/files/DataBase.csv`.
- Notebook outputs were left as they were.

**Verified:**

- `Nature`: a small CMA-ES and a small DE gave identical results before and after the move.
- `Regions/Marlim`: the `MarlimBlock` table reproduces the saved output exactly.
- `Regions/Synthetic`: the tile table reproduces exactly; the imports resolve to `Synthetic.Nature`; the campaign memory
  and the cache are found.
- `Generator`: the four real sections and the annotations load from the shared cache.
- `DataBase.csv`: every path exists, and only `img_path` changed.
- `Blocks/Marlim/Analysis.ipynb` and `Blocks/Iracema/Analysis.ipynb` don't depend on anything that moved.

**Open question for the user** (ask once, with your recommendation):

- The old similarity model `Synthetic/Regions/Synthetic/model/info.json` says `"network": "unet3d_v2"`, the old name of
  `dbrnet`, which `ModelNetwork` no longer knows.
- So `ImageSimilarity.update()` fails. This predates the move.
- The previous session verified that the weights load into `dbrnet` with `strict=True` (74 tensors, `bot_dilated` hook
  present). Renaming the network in that `info.json`, or mapping it in the notebook, is the whole fix.
- Do it only if the user agrees; don't edit their model files silently.

If they have one of the moved notebooks open in VS Code/Jupyter from the old path, an old save would recreate it there:
check `Blocks/Marlim/` once at the start.

**Pre-approved layout.** Anything else needs a question with your recommendation.

```
Synthetic/
├── index.py               SyntheticGenerator: one class, extended in place
├── utils/index.py         existing helpers
├── Analysis.ipynb         existing visual walk-through (keep it working; add new stages to it only if they need a visual check)
├── PROMPT.md              this file
├── Nature/                the user's genetic algorithms (moved from the repo root)
├── Regions/               the previous attempt (moved), kept runnable as reference and as a baseline methodology
│   ├── Marlim/            label-based calm/faulted/dead real tiles
│   ├── Synthetic/         similarity + CMA-ES calibration, cache, memory, model/
│   └── Generator.ipynb    legacy MarlimSyntheticGenerator (reference only)
├── "1 - ....ipynb", ...   the new methodology: at most 2–3 numbered notebooks, like Model/ (e.g. profile, calibration, dataset)
├── modifications.md       the final report (Portuguese)
└── files/
    ├── <block>/           label-free samples and the measured profile of each target block (marlim/, iracema/)
    ├── memory/            Nature checkpoints, one folder per campaign
    └── experiments/       already holds harness/, baseline.jsonl and analise_19-10/; add journal.md, results.jsonl, logs/,
                           datasets/<tag>/, models/model_N/ (delete heavy items once logged)
```

Also pre-approved:

- `Dataset/<final name>/` for the final dataset, with a copy of the matching `Format.ipynb`;
- edits to `Synthetic/index.py`.

Classes shared by two notebooks (for example the block sampler and the profile) may go into `Synthetic/index.py` next to
the generator, but ask once before doing it. Classes used in a single notebook stay in it, as in `Regions/`.

**Gate 0** (passed on 19/10/2026): the moved notebooks run exactly as before. Copy the list of moves above into the
journal when you create it.

### Phase A — Harness, baselines and noise

1. **Evaluation harness.**
   - It reuses the class cells of `Blocks/Marlim/Analysis.ipynb` verbatim (exec the cells) and keeps only the
     annotated inline. Prediction costs the same, since every tile covers the inline, but nothing large is written to
     disk.
   - A verified harness is in `Synthetic/files/experiments/harness/marlim.py`, copied there on 19/10/2026 so it
     survives reboots. It provides:
     - `evaluate(modelDir, patches, save=..., tag=...)`: official predictions, stick metrics and `defectMetrics`, one
       JSON row per patch;
     - `stickMetrics(pid, section)`;
     - `evaluateFast`, an approximate slab-based prediction used only to compare probes among themselves.

     It execs the notebook's class cells, so it follows the notebook if the notebook changes. Its slab cache lives in
     `/tmp/falhas_slabs`, is disposable, and is rebuilt when missing.
   - `Synthetic/files/experiments/baseline.jsonl` holds the official-method metrics of the existing models on the four
     patches:
     - `art5_dbrnet` = `Artigo/model_5`;
     - `ms7_nru` = `Multiscale/model_7`;
     - `ms5_macnn` = `Multiscale/model_5`;
     - `ms6_fsn` = `Multiscale/model_6`, the old Fault-Seg-Net; `FaultSegNetOld.py` beside the harness loads it;
     - `ms8_fef` = `Multiscale/model_8`, patch 1200 only.
   - Verification target: `evaluate('Documents/Backups/Artigo/model_5', ['1200'])` must give exactly recall 0.528,
     deteccao 0.485, precisao 0.203, F1 0.293 and 77 sticks. The float16 sections saved in `analise_19-10/sections` give
     0.528 / 0.485 / 0.201 / 0.291 / 78 sticks, because of the storage precision.
   - Add `F1@1%` and a threshold sweep.
2. **Training harness.**
   - Use a scratch copy of `Model/1 - Model.ipynb` run by papermill with `cwd=Model/`.
   - Redirect every path it reads or writes: `../Task/info.json` to your own info.json,
     `../Dataset/<d>/DataBase.csv` to your CSV, `Backup/` to `Synthetic/files/experiments/models/`, and `progress.json`
     to your own file.
   - Diff it against the original to prove that only those strings changed.
   - Never write `Task/info.json` or `Model/Backup` while the user may be running a campaign.
3. **Proxy recipe.** The full reference recipe costs ~8–12 h per training, so the many ablations need a cheaper proxy:
   fewer epochs, fewer filters or fewer tiles.
   - Validate it: it must order ≥ 4 datasets (`dataset_74`, `dataset_wu`, `dataset_74_wu`, `marlim_opt`, ...) the same
     way as the full recipe on dev.
   - **Free anchors with the full reference recipe already exist:** `dbrnet`, lr 1e-3, `dice_focal`, dropout 0.1, nf 32,
     100 epochs. Evaluating one on dev costs ~15 min of GPU and no training.

     | model | dataset | scaling |
     |---|---|---|
     | `Documents/Backups/Artigo/model_2` | `dataset_wu` | normalize |
     | `Documents/Backups/Artigo/model_5` | `dataset_74_wu` | normalize |
     | `Documents/Backups/Marcia/model_1` | `dataset_74` | percentile |
     | `Documents/Backups/Marcia/model_2` | `dataset_wu` | percentile |
     | `Documents/Backups/Marcia/model_3` | `dataset_74_wu` | percentile |
     | `Documents/Backups/Marcia/model_5` | `dataset_74_mar` | percentile |
     | `Documents/Backups/Marcia/model_6` | `dataset_wu_mar` | percentile |

     - Percentile against normalize costs only ~0.005 IoU (measured).
     - The `*_mar` mixes (`dataset_74`/`dataset_wu` plus `marlim_opt`) no longer exist as folders. Rebuild them from
       their sources if you need their images; the counts are in each model's `info.json`.
     - Together these give the full-recipe transfer of 5 datasets. That is enough to validate the proxy (train it on 3–4
       of those datasets and compare the order) and to start the M2 meta-validation set.
   - Short trainings mark more (NRU marks 4.7% of the section at epoch 20 and 2.0% at epoch 85), so compare at `F1@1%`
     too.
   - If no proxy agrees with the full recipe, say so and budget the full recipe.
4. **Noise.** Measure the seed-to-seed spread of the dev metrics under the proxy, and under the full recipe for the
   reference. Differences below this are not results.

**Gate A:** the harness matches the notebook exactly; the noise is known; the proxy is validated, or rejected with a
cost plan.

### Phase B — Target profile and gap analysis

1. **Label-free sampler.** It draws tiles from a block through its `patch_metadata.json`. For Marlim, use the dev slabs
   only.
   - Classify each sample label-free: water, seafloor, dead traces, reflective interval, dead zone, and a "discontinuous"
     class by semblance.
   - Start from `MarlimBlock` and `RegionFinder`.
2. **Profile.** Compute descriptors per depth band and per sample class, with **the same code on real and synthetic**
   and in the network's input scaling.
   - Start from what already exists: `ImageSimilarity.process` (mean, std, kurtosis, saturation, period, lateral
     correlation, coherence), `MarlimBlock`, and the old 40-feature ruler.
   - Add what the gap analysis needs:
     - spectral centroid and bandwidth against z;
     - reflector dip distribution from the structure tensor;
     - fine-noise ratio;
     - crosshatch and coherent-noise energy;
     - saturation;
     - the dip, length, spacing and density of fault-like discontinuities.
3. **Error analysis** of the baseline predictions on dev, with metrics and figures:
   - which expert faults are missed, and why (band, period, apparent throw, visibility);
   - where false positives fall (dead-zone hatch, strong reflectors, edges and dead traces, seafloor, stratigraphy).
4. **Gap table.** Profile `dataset_74`, `dataset_wu`, `dataset_74_wu`, `marlim_opt` and `dataset_regions`, and list
   every descriptor where the synthetic support does not cover the real one.

**Gate B:** a ranked list of hypotheses of the form "factor X causes error Y", each with its evidence.

**Starting hypotheses**, ordered by the evidence in 5.8. Each maps to a generator factor and to a measurable error mode.
Confirm or reject each one, and add your own.

- **H8. Fault density and size.** The synthetic data has 4–8× the expert's fault density, and the network draws 4× the
  expert's stick length in z 450–1200. Fewer, longer, more significant faults per tile, with some tiles that have none,
  should lower the over-prediction. Testable now with `faultCount`.
  - **H8b, to test and not assume.** The geologist draws only the main faults, and "replacing the geologist" means
    producing that interpretation. So a variant labels only faults above a size threshold (throw relative to the local
    period, and length) and leaves the small, visible ones unlabeled.
  - Risk: unlabeled visible offsets can also teach the network to ignore real faults. Measure both recall and precision
    on dev, and look at the figures.
- **H9. Unlabeled structural and stratigraphic look-alikes.** Most unmatched sticks follow steep fold or mound flanks,
  onlaps, pinch-outs and lateral truncations, and Marlim's reflectors are lenticular at a scale of tens of pixels.
  - This needs code: laterally varying reflectivity (lenses, pinch-outs, terminations), sharp or steep folds and
    mounds, and channel-like erosional cuts.
  - None of it gets a label. It teaches that steep reflectors and terminations are not faults.
- **H1. Scale.** Marlim's faults sit mostly at periods of 17–25 px; the training data has 8–11 px.
  - Generate faults at Marlim's periods (4–25 px, thin labels), per tile or varying with depth inside a tile.
  - The scale test in 5.3 predicts gains in z 700–1400.
  - Testable now with `waveletFreq`: the period is ≈ 1/(f·0.0012), so f in 33–208 covers 25 down to 4 px.
  - Uniform in f over-samples short periods, so approximate log-uniform with 2–3 regions of `dataset()`.
- **H4. Subtle faults and visibility.** Marlim's faults are 4× less sharp. By construction, the synthetic offsets are
  1.3–4 periods at the fault centre (`faultThrow` 15–32 px against periods of 8–11 px). That is an estimate from the
  parameters: measure the real throw-to-period ratio along the expert's sticks.
  - Use smaller throws relative to the period, with a fault zone and drag.
  - Label only where the offset is visible: labelling invisible offsets made the task unlearnable before (7.7).
- **H5. Texture.** Marlim is 5× smoother at fine scale. But a past ablation found the speckle noise of `dataset_74`
  better than a smooth noise (7.5), so test `noiseLevel`/`noiseSigma` rather than assume. Testable now.
- **H2. Water, dead-zone and edge negatives.** Half of every evaluated section is water or dead zone, and 29–39% of
  the mask lands in the dead zone.
  - The stick metric barely sees this, because the extractor's energy gate removes it. It still matters for a clean
    output that could replace the geologist, and for the edge artifacts.
  - The expert does extend faults into the dead zone (16–42% of the annotation), so compare an unlabeled dead zone
    against labels continued through it.
- **H3. Fault population geometry.** Domino families, with the strike controlled relative to the inline so apparent
  dips fall in 40–78°, and 20–60 px spacing.
- **H6. Contrast.** Amplitude varying with depth inside a tile, isolated bright segments, and Marlim-like saturation
  (1.6–3.9%).
- **H7. Mixture.** The proportions of each tile type. The evaluated sections are ~18% water, ~35% dead zone and ~47%
  live.
- **H10. Wavelet shape.** Randomize phase rotation and polarity. Cheap.

### Phase C — Causal ablations (TSTR)

- For each top hypothesis, generate a controlled variant that changes **only that factor**: same seeds, same everything
  else.
- Train it with the proxy and evaluate it on dev with the band-wise and defect metrics. Use 2 seeds when the effect is
  near the noise.
- Start with the factors whose expected effect is largest. Don't spend GPU on ±10% tweaks before the structural factors
  are settled.
- **Run the no-code ablations first**, while you implement the rest. They only change options of the current class.
  These are starting values, not prescriptions:
  - density: `faultCount` (1, 4) and (0, 3) against (5, 10);
  - noise: `noiseLevel` (0.0, 0.15) against (0.015, 0.618);
  - period: `waveletFreq` covering 33–208, for example three `dataset()` regions at (33, 55), (55, 100) and (100, 208);
  - the best combination of the three.
- Phases C and D interleave. Implement a factor in the class, off by default, ablate it, and keep it only if it moves
  dev.
- **Candidate C0.** Build by hand a broad configuration that covers every measured Marlim range, using the factors that
  helped. It is the first serious candidate, and the GA has to beat it.
- These variants double as the meta-validation set for M2.

**Gate C:** the factors that matter, the direction and size of each effect, and the resulting generator design.

### Phase D — Generator changes

- Implement the factors that matter inside `SyntheticGenerator`, in the user's style (section 11).
- Keep the GPU path and the `fast` mode working, and keep `dataset()` compatible; extend it if the mixture needs it.
- Prove backward compatibility (rule 3).
- Check every new stage visually, with real and synthetic tiles side by side on the same scale, and statistically, with
  the profile.
- Report the time per tile.

**Gate D:** old configurations are bit-identical; the new features are checked; the timing is known.

### Phase E — GA calibration methodology

1. Design the label-free objective and validate it (M2) on the Phase C variants and the anchors. Iterate the design
   until it passes.
   - If no label-free objective passes, fall back to TSTR in the loop with the validated proxy.
   - For that fallback, use common random numbers, enough real tiles and fresh-seed re-evaluation, and explain the
     reason in the report.
2. **Genome.**
   - Ranges (center + log-width) per region or mixture component, warm-started from the best measured configuration
     through `mean`.
   - A box set by physical plausibility.
   - Measured quantities fixed (M4).
3. Run `NatureSelector('genetic', problem, memory=...)` with checkpoints in `Synthetic/files/memory/<block>_<hash>/`.
   Choose the final candidate on fresh seeds between the reference, the best individual and the cloud mean, as the
   user's `Calibration.evaluate()` does.
4. Run a TSTR audit on dev for the 2–3 best candidates plus the reference.

**Gate E:** the GA winner beats the reference on the objective on fresh seeds, and on dev TSTR beyond the noise.

### Phase F — Final validation (the only time the test patches are used)

1. Generate the final dataset into `Dataset/<name>/` with `synthetic.json` and its `Format.ipynb`. Use 220 tiles unless
   an ablation justified another number.
2. Train the reference recipe (full, 100 epochs) with ≥ 2 seeds, one second architecture, and the baselines with the
   same seeds where the budget allows.
3. Evaluate with the **official** `Blocks/Marlim/Analysis.ipynb` on all four patches, through a scratch copy of it run
   with papermill, its constants patched and `cwd=Blocks/Marlim`. It writes its outputs under the `BASE_PATH` you give
   it.
4. Report dev and test separately plus the 4-patch mean, with `F1@1%`, the defect metrics, the figures and the visual
   audit.
5. **Iracema portability.** Run the profile and the calibration with only a configuration change. Show the predictions
   on inline 1375 against the baseline model, with the same visual audit. If compute is short, run at least the profile
   and the calibration.

**Gate F:** sections 2.4 and 14 are satisfied, or an honest report says what was achieved and what blocks further
progress.

### Phase G — Deliver (section 13)

---

## 10. Mechanics, compute and disk

### Environment

- Run with `conda run -n torch-gpu python ...` or `~/anaconda3/envs/torch-gpu/bin/python`. The `base` env has no torch.
- Every path inside a notebook is relative to its own folder, so always pass papermill the right `cwd`.

### Generating a dataset

```python
generator = SyntheticGenerator(seed=SEED)
generator.dataset({'directory': '<dir>', 'seed': SEED, 'regions': {...}})
```

Then save the same options as `synthetic.json`. Examples: the last cells of `Synthetic/Analysis.ipynb` and the section
"GERANDO OS DADOS SINTÉTICOS NOVOS" of the old calibration notebook.

### Formatting

- Copy `Dataset/dataset_74/Format.ipynb` (flat `original/`) or `Dataset/dataset_regions/Format.ipynb`
  (`original/<region>/`).
- It applies `Normalization(scaling)` from `Dataset/index.py` and writes `images/`, `masks/` and `DataBase.csv` (absolute
  paths plus a `scaling` column).
- It also **rewrites `Task/info.json`**.
- With `img_size: null` it ends in `SystemExit`. That is success: check the CSV, not the exit code.
- For private experiments, reproduce the Format with `Normalization` in your own script, so that you never write
  `Task/info.json`.

### Scaling

- The `.dat` tiles are the block's p01/p99 mapped to [0, 1]. The reference recipe uses `'normalize'` (min-max per
  volume).
- `dbrnet` uses convolutions without bias plus GroupNorm, so it is nearly blind to global gain and offset; saturation and
  clipping still matter.
- If you change the scaling, prove it on dev and keep the comparison fair.
- The Marlim notebook only predicts models with `scaling` set to `'normalize'` or `'percentile'`.

### Training

- **Official:** `Task/task.json`, then `cd Task && python index.py`, which writes `Model/Backup/model_N`.
- **Private:** the redirected copy from Phase A.
- The final deliverable includes a `Task/task.json` proposal so the user can reproduce the comparison through the
  official pipeline. Write it to `Task/task.json` only if they agree.

### Costs (Quadro P6000 24 GB, 20 CPUs, 125 GB RAM)

| item | cost |
|---|---|
| GPU caveats | Pascal: FP64 runs at 1/32 of FP32, and AMP doesn't help |
| generator | 2–7 s per tile in default mode, ~0.7 s with `fast`; on CPU (8 threads) 13–15 s per tile, ~50 min for 220 tiles (measured 19/10) |
| `dbrnet` (nf 32, 128³, batch 2) | ≈ 5 min per epoch per 200 training tiles: ≈ 8 h for 100 epochs, ≈ 12 h on `dataset_74_wu` |
| multiscale networks | slower than `dbrnet`; `fault_seg_net` with nf 32 uses ~22 GB of GPU memory, so nothing else fits beside it |
| Marlim prediction | ≈ 0.45 s per tile at 128³: ≈ 7 min per patch, ≈ 30 min for all four |
| old transfer probe | ≈ 6 min per evaluation |

### Disk

- 319 GB were free on 19/10/2026.
- A 220-tile dataset is ≈ 4–5 GB (original plus formatted).
- The official notebook writes ≈ 7.3 GB of masks per patch per model (29 GB for four patches), plus a 0.9 GB
  `predicted.npy` per patch.
- Use the inline-only harness for experiments and the official notebook only for final models.
- Delete experiment datasets and predictions **you** created once their numbers are logged; they can be regenerated from
  `synthetic.json` + seed. Never delete anything you didn't create.

### GPU sharing

- Check `nvidia-smi` and `ps` before starting anything. On 19/10 the user had a campaign running: `Task/index.py` with
  `fault_seg_net` and `nra_sanet` writing into `Model/Backup`, plus a Jupyter kernel of theirs.
- Never stop the user's jobs, and don't touch `Task/` while their jobs run.
- On 19/10 at 18:00 the user's `Task/index.py` campaign was still training: `fault_seg_net` at epoch 3 of 100, ~18 min
  per epoch, with `nra_sanet` queued after it, using ~20 GB of VRAM. It may still occupy the GPU the next day.
  - If it does, start with CPU work: profile, gap table, generator code. The generator also runs on CPU (`device` set
    to `'cpu'` through `set()`), and analysis scripts can use `CUDA_VISIBLE_DEVICES=""`.
  - Before stopping the campaign to free the GPU, ask the user: it is their comparison, and the decision is theirs.
- The generator also runs on the GPU, in float64, so generating while a training runs slows both.
- Ask before starting something that would starve their run, or a single job longer than ~24 h.

### Long jobs

- Launch them with `setsid nohup ... > <log> 2>&1 &`.
- Keep logs and checkpoints under `Synthetic/files/experiments/`, never in `/tmp`: it is wiped at boot, and the user
  sometimes restarts the PC.
- Make everything resumable.

### Pitfalls already paid for

- A notebook that uses cv2 and then forks a pool needs `cv2.setNumThreads(1)` in its first cell, or the children
  deadlock.
- `python3 - <<EOF ... < /dev/null` runs nothing, because stdin is empty.
- Papermill cell output never reaches the stdout of `Task/index.py`.
- `randint` upper bounds are exclusive.
- `a[i:i+randint()] = uniform()` draws the `uniform` first: Python evaluates the right side before the left.
- `np.float64 * float32` gives float64 (NEP 50), which is why `applyGain` casts back to float32.
- The order of `np.random` calls defines every tile.

---

## 11. Coding style (mandatory: "respeitar meu estilo de codar igual faço na pasta Model")

Imitate `Synthetic/index.py`, `Model/Network/index.py`, `Model/1 - Model.ipynb` and the classes of
`Synthetic/Regions/Synthetic/Analysis.ipynb`.

**Classes**

- Use classes with clear responsibilities.
- The main computation is `update()`; helpers are `get<Thing>()`; `info()` returns a dict; `plot()` or `show()` draws;
  `set(options)` overrides attributes.
- Class-level UPPERCASE constants hold the knobs.

**Naming**

- Identifiers in English. Methods and attributes in `camelCase`, constants in `UPPERCASE`.
- No leading underscore, no docstrings, no type hints.

**Comments**

- Exactly one UPPERCASE comment line in Portuguese above every class.
- Elsewhere, comments only where there is a hidden trap: an UPPERCASE line above a non-obvious method, or a short
  trailing comment on a config constant.
- No comment noise.

**Layout**

- Align `=` in consecutive assignment blocks, as the user does.
- Never break a call over several lines.
- Keep methods short.
- No scattered functions, no "parâmetros doidos" (unexplained magic numbers), no DataFrames everywhere.

**Notebooks**

- Imports go in the first cell, and the UPPERCASE configuration in its own cell.
- Each section has a `# TÍTULO` markdown with bullet points explaining it.
- One step per cell, each ending in a visible proof: a print, a table or a figure.
- A class is defined in a cell and used right below it.
- Importing a module never executes work.

**Simplicity**

- The user's "robust" means it doesn't break and doesn't hang. It does not mean ceremonial guarantees.
- Prefer a small surface. Use one class where one is enough.

Illustration of the style only, not a design:

```python
# PERFIL DE UM BLOCO REAL: DESCRITORES POR FAIXA DE z, MEDIDOS DO MESMO JEITO NO REAL E NO SINTÉTICO
class BlockProfile:
    BANDS = ((0, 450), (450, 700), (700, 950), (950, 1200), (1200, 1601))

    def __init__(self, block, nTiles=200, seed=42):
        self.block  = block
        self.nTiles = nTiles
        self.seed   = seed

    def update(self):
        self.table = pd.DataFrame([self.getFeatures(tile) for tile in self.getTiles()])
        return self.table

    # PERÍODO VERTICAL PELO CENTRÓIDE DO ESPECTRO EM z: O EIXO QUE MAIS SEPARA O SINTÉTICO DO MARLIM
    def getPeriod(self, section):
        power = (np.abs(np.fft.rfft(section - section.mean(0), axis=0)) ** 2).mean(1)[1:]
        freqs = np.fft.rfftfreq(len(section))[1:]
        return float(power.sum() / ((freqs * power).sum() + 1e-12))
```

---

## 12. Working discipline

**Journal.** Keep `Synthetic/files/experiments/journal.md` in Portuguese, plus a machine-readable `results.jsonl`.

- For every experiment record the hypothesis, the change, the config/seed/git hash, the result (numbers with their
  noise), the decision and the next step.
- After a context compaction, re-read the journal before acting. It is your memory across a task this long.
- Your context is compacted automatically when needed. Don't wrap up early because of it: the work can continue across
  compactions as long as the journal is current.
- The user's plan has a usage limit, so a session may end mid-task and a new one take over. Keep the journal current
  after every experiment, with what is queued, where its log is and what to do next, so a fresh session can resume
  from it.

**Feedback loop.** After every result, ask:

- Did it move the dev metric beyond the noise, at both thresholds?
- What would falsify it?
- Does it bring the model closer to replacing the geologist, or only to a better number?

Look at the figures. A rising number paired with uglier figures is a warning, not a win.

**Honesty**

- Report negative results with the same care as positive ones.
- Never tune on the test patches, never post-process predictions to improve the metric, never hide a failed check.
- If a check fails, say so and show the output.

**Efficiency**

- Run independent tool calls in parallel.
- Use the GPU around the clock with queued, resumable jobs, but never blindly.
- Before each long run, write in the journal what you expect and why it is worth the hours.

**Talking to the user**

- Write in Portuguese, briefly, at each gate.
- Ask only when the decision is genuinely theirs:
  - structure beyond section 9;
  - deleting their data;
  - changing class defaults;
  - overwriting `Task/` files;
  - fixing their notebooks' pre-existing bugs;
  - jobs that would block their GPU use.
- Each question carries your recommendation. Otherwise, proceed.
- The user is under real pressure to make this work. Report the first concrete numbers as early as possible (the
  anchors, then the first ablations), keep messages short and factual, and never round a result in your favour.

**Memory.** At the end, add or update notes in the project memory: durable measured facts, decisions and failed
approaches, following the memory rules. Don't duplicate what the report or the repo already records.

---

## 13. Deliverables

1. Keep `Synthetic/Regions/` and `Synthetic/Nature/`, moved in Phase 0, runnable.
2. `Synthetic/index.py` with the generator extended, backward compatible, with the proof in the journal.
3. The methodology notebooks in `Synthetic/`, block-agnostic and driven by a block configuration: sampler, profile,
   objective validation, GA calibration, final choice, dataset generation.
4. The Marlim profile and calibration under `Synthetic/files/marlim/` and `Synthetic/files/memory/`, and the final
   dataset `Dataset/<name>/` with `synthetic.json`, `Format.ipynb` and `DataBase.csv`.
5. The final models and their official evaluation (`sticks_report.csv`, figures) on all four patches, with dev and test
   reported separately.
6. The Iracema portability run: profile and calibration, plus qualitative predictions if possible.
7. A ready `Task/task.json` proposal comparing the new dataset against `dataset_74_wu` in the user's parametrization,
   for `dbrnet` and the multiscale networks. Put its content in the report, and write `Task/task.json` only with
   permission.
8. `Synthetic/modifications.md` in Portuguese, in the style of `Documents/Backups/Multiscale/modifications.md`:
   - a summary with the headline numbers;
   - how everything was measured;
   - the methodology, explained so it can be applied to a new block step by step;
   - the generator changes and why;
   - the objective validation;
   - the GA campaign;
   - the final results (dev / test / Iracema);
   - what didn't work;
   - how to reproduce;
   - the files changed.
9. The journal and `results.jsonl` in `Synthetic/files/experiments/`, and the updated memory notes.
10. Suggested (not applied) updates to `CLAUDE.md` and to `Documents/*.md` for the new methodology. `CLAUDE.md` already
    reflects the moves of 19/10/2026; `Documents/1 - Metodologia.md` still cites old paths.

---

## 14. Final self-check before declaring done

- [ ] Training data is 100% synthetic; no network, recipe or evaluation was changed to improve the numbers.
- [ ] Old generator configurations reproduce bit for bit (mask identical, image |Δ| ≤ 1e-8).
- [ ] The GA objective is label-free, and its validation against TSTR (ρ, intervals, pairwise order) is in the report.
- [ ] The test patches were used only in Phase F; dev and test are reported separately, plus the 4-patch mean.
- [ ] The headline comparison uses the user's parametrization, ≥ 2 seeds, both thresholds (0.5 and `F1@1%`), and the
      uncertainty.
- [ ] The gain replicates with a second architecture.
- [ ] Figures on all four patches and on Iracema inline 1375 were inspected and are in the report; the visual audit of
      false positives was done.
- [ ] The procedure reruns from `synthetic.json` + seeds; GA campaigns resume from `Memory`; notebooks run top to
      bottom.
- [ ] The same notebooks ran on Iracema with only a configuration change.
- [ ] The code follows section 11, and no unapproved structure was created.
- [ ] Heavy temporary files you created are deleted; nothing of the user's was deleted or overwritten.
- [ ] The report, journal and memory notes are written, in Portuguese.
- [ ] Nothing is claimed that a number in the report doesn't support.

**Useful references** (optional):

- Wu et al. 2019, FaultSeg3D, *Geophysics* 84(3) (local PDF).
- Wu, Geng, Shi, Pham, Fomel & Caumon 2020, "Building realistic structure models to train convolutional neural networks
  for seismic structural interpretation", *Geophysics*.
- Tobin et al. 2017, domain randomization.
- Ruiz et al. 2019, *Learning to Simulate*.
- Kar et al. 2019, *Meta-Sim*.
- Kynkäänniemi et al. 2019 and Naeem et al. 2020, precision/recall and coverage of distributions.
- Ben-David et al. 2010, the domain-adaptation bound.
- Smith & Winkler 2006, the optimizer's curse.
- Rizzo & Székely 2016, energy distance.
- Hansen, *The CMA Evolution Strategy: A Tutorial*.
