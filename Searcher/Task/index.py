#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task/index.py — Bayesian Optimization Engine (OOP Architecture)

Maximizes `iou_wu` by tuning geological parameters in Synthetic/config.json
using Optuna with a phase-based hybrid sampler (TPE + Random + CMA-ES).

Pipeline per trial:
  1. Optuna proposes geological parameters
  2. Writes config.json + info.json
  3. Executes notebooks: Generate → Format → Train → Predict
  4. Collects iou_wu from Searcher/database/model_*/info.json
  5. Updates CSV history

Resilience: SQLite-backed study auto-recreates if deleted.
Historical data from Searcher/database/ is re-injected on every startup.
"""

import optuna
from optuna.distributions import IntDistribution, FloatDistribution
from optuna.trial import create_trial, TrialState
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
import papermill as pm
from pathlib import Path
import os
import json
import glob
import csv
import shutil
import random
import traceback
from datetime import datetime


class SearchSpace:

    PARAMS = [
        {"name": "layerRange",      "type": "int_range",
         "min_bounds": (20, 100),   "max_bounds": (130, 350)},
        {"name": "layerThickness",  "type": "int_range",
         "min_bounds": (1, 1),      "max_bounds": (2, 5)},

        {"name": "foldCount",       "type": "int_range",
         "min_bounds": (5, 28),     "max_bounds": (28, 60)},
        {"name": "foldSigma",       "type": "int_range",
         "min_bounds": (7, 30),     "max_bounds": (27, 75)},
        {"name": "foldAmplitude",   "type": "int_range",
         "min_bounds": (-50, -24),  "max_bounds": (-5, 22)},
        {"name": "foldDamping",     "type": "float_scalar",
         "bounds": (0.5, 2.5)},
        {"name": "foldBaseShift",   "type": "float_range",
         "min_bounds": (-3.0, 0.5), "max_bounds": (1.0, 7.0)},

        {"name": "shearOffset",     "type": "float_range",
         "min_bounds": (-10.5, -7.5), "max_bounds": (2.5, 8.0)},
        {"name": "shearGradient",   "type": "float_range",
         "min_bounds": (-0.16, -0.09), "max_bounds": (0.00, 0.06)},

        {"name": "faultCount",      "type": "int_range",
         "min_bounds": (1, 6),      "max_bounds": (7, 15)},
        {"name": "faultThrow",      "type": "int_range",
         "min_bounds": (12, 16),    "max_bounds": (28, 60)},
        {"name": "faultDipAngle",   "type": "int_range",
         "min_bounds": (40, 68),    "max_bounds": (74, 84)},
        {"name": "faultRoughness",  "type": "float_scalar",
         "bounds": (3.0, 7.0)},
        {"name": "faultRoughSigma", "type": "float_scalar",
         "bounds": (4.5, 12.0)},
        {"name": "faultDecaySigma", "type": "int_range",
         "min_bounds": (30, 65),    "max_bounds": (45, 160)},
        {"name": "faultZoneWidth",  "type": "float_scalar",
         "bounds": (0.40, 1.50)},
        {"name": "faultThreshold",  "type": "float_scalar",
         "bounds": (0.10, 0.9)},
        {"name": "faultCurveProb",  "type": "float_scalar",
         "bounds": (0.005, 0.7)},
        {"name": "faultCurveMax",   "type": "float_scalar",
         "bounds": (1.0, 15.0)},

        {"name": "waveletFreq",     "type": "int_range",
         "min_bounds": (60, 110),   "max_bounds": (78, 120)},
        {"name": "waveletDuration", "type": "float_scalar",
         "bounds": (0.09, 0.12)},
        {"name": "waveletDt",       "type": "float_scalar",
         "bounds": (0.0005, 0.0030)},

        {"name": "noiseLevel",      "type": "float_range",
         "min_bounds": (0.003, 0.05), "max_bounds": (0.48, 0.70)},
    ]

    def __init__(self):
        self._distributions = None

    @property
    def distributions(self):
        if self._distributions is None:
            self._distributions = self._build_distributions()
        return self._distributions

    def _build_distributions(self):
        dists = {}
        for p in self.PARAMS:
            if p["type"] == "int_range":
                dists[f"{p['name']}_min"] = IntDistribution(*p["min_bounds"])
                dists[f"{p['name']}_max"] = IntDistribution(*p["max_bounds"])
            elif p["type"] == "float_range":
                dists[f"{p['name']}_min"] = FloatDistribution(*p["min_bounds"])
                dists[f"{p['name']}_max"] = FloatDistribution(*p["max_bounds"])
            elif p["type"] == "float_scalar":
                dists[p["name"]] = FloatDistribution(*p["bounds"])
        return dists

    def suggest_config(self, trial):
        config = {}
        for p in self.PARAMS:
            name = p["name"]
            ptype = p["type"]

            if ptype == "int_range":
                v_min = trial.suggest_int(f"{name}_min", *p["min_bounds"])
                v_max = trial.suggest_int(f"{name}_max", *p["max_bounds"])
                if v_min > v_max:
                    v_min, v_max = v_max, v_min
                config[name] = [v_min, v_max]

            elif ptype == "float_range":
                v_min = trial.suggest_float(f"{name}_min", *p["min_bounds"])
                v_max = trial.suggest_float(f"{name}_max", *p["max_bounds"])
                if v_min > v_max:
                    v_min, v_max = v_max, v_min
                config[name] = [v_min, v_max]

            elif ptype == "float_scalar":
                config[name] = trial.suggest_float(name, *p["bounds"])

        return config

    def config_to_flat_params(self, config):
        params = {}
        for p in self.PARAMS:
            name = p["name"]
            value = config.get(name)
            if value is None:
                continue
            if p["type"] in ("int_range", "float_range"):
                params[f"{name}_min"] = value[0]
                params[f"{name}_max"] = value[1]
            elif p["type"] == "float_scalar":
                params[name] = value
        return params

    def clamp_params_to_bounds(self, params):
        clamped = {}
        for key, val in params.items():
            if key in self.distributions:
                dist = self.distributions[key]
                clamped[key] = max(dist.low, min(dist.high, val))
                if isinstance(dist, IntDistribution):
                    clamped[key] = int(round(clamped[key]))
            else:
                clamped[key] = val
        return clamped


class HistoricalDataManager:

    def __init__(self, database_dir, search_space):
        self.database_dir = database_dir
        self.search_space = search_space

    def scan_models(self):
        results = []
        pattern = os.path.join(self.database_dir, "model_*", "info.json")
        for info_path in sorted(glob.glob(pattern)):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append((info_path, data))
            except Exception as e:
                print(f"  ⚠ Could not read {info_path}: {e}")
        return results

    def _is_valid_model(self, data):
        if "config" not in data:
            return False
        iou = data.get("iou_wu")
        if iou is None or not isinstance(iou, (int, float)):
            return False
        if iou <= 0.0:
            return False
        flat = self.search_space.config_to_flat_params(data["config"])
        missing = set(self.search_space.distributions.keys()) - set(flat.keys())
        return len(missing) == 0

    def filter_and_clean(self, delete_failed=False):
        all_models = self.scan_models()
        valid = []
        failed = []

        for info_path, data in all_models:
            if self._is_valid_model(data):
                valid.append(data)
            else:
                model_dir = os.path.dirname(info_path)
                model_id = data.get("model", os.path.basename(model_dir))
                failed.append((model_dir, model_id))

        if failed:
            for model_dir, model_id in failed:
                print(f"  ⚠ Invalid model: {model_id} (path: {model_dir})")
                if delete_failed:
                    try:
                        shutil.rmtree(model_dir)
                        print(f"  🗑 Deleted: {model_dir}")
                    except Exception as e:
                        print(f"  ✘ Failed to delete {model_dir}: {e}")
        else:
            print(f"  ✔ All {len(valid)} models are valid.")

        return valid

    def inject_into_study(self, study, delete_failed=False):
        already_injected = set(study.user_attrs.get("injected_model_ids", []))
        valid_models = self.filter_and_clean(delete_failed=delete_failed)
        injected_count = 0

        for data in valid_models:
            model_id = data.get("model", "unknown")
            if model_id in already_injected:
                continue

            flat_params = self.search_space.config_to_flat_params(data["config"])
            clamped_params = self.search_space.clamp_params_to_bounds(flat_params)

            try:
                frozen_trial = create_trial(
                    params=clamped_params,
                    distributions=self.search_space.distributions,
                    values=[data["iou_wu"]],
                    state=TrialState.COMPLETE,
                )
                study.add_trial(frozen_trial)
                already_injected.add(model_id)
                injected_count += 1
                print(f"  ✔ Injected: {model_id} → iou_wu = {data['iou_wu']:.4f}")
            except Exception as e:
                print(f"  ⚠ Injection failed for {model_id}: {e}")

        study.set_user_attr("injected_model_ids", list(already_injected))

        if injected_count > 0:
            print(f"\n  ▸ {injected_count} historical trial(s) injected.")
        else:
            print(f"\n  ▸ No new historical trials to inject.")


class PipelineExecutor:

    NOTEBOOKS = [
        "../Synthetic/Generate.ipynb",
        "../Dataset/dataset_synthetic/Format.ipynb",
        "../Model/Analysis.ipynb",
        "../Model/Predict.ipynb",
    ]

    def __init__(self, synthetic_config_path, database_dir, training_config, kernel="python3"):
        self.synthetic_config_path = synthetic_config_path
        self.database_dir = database_dir
        self.training_config = training_config
        self.kernel = kernel

    def execute_trial(self, config, trial_number):
        self._write_config(config)
        self._write_info()

        models_before = self._list_model_dirs()

        pipeline_ok = True
        for nb_path in self.NOTEBOOKS:
            if not self._execute_notebook(nb_path):
                pipeline_ok = False
                print(f"\n  ✘ Pipeline halted at {nb_path}")
                break

        if not pipeline_ok:
            return None, None, "FAIL"

        model_id = self._detect_new_model(models_before)
        iou_wu = self._read_result(model_id)

        if iou_wu is not None:
            print(f"\n  ★ iou_wu = {iou_wu:.4f} ({model_id})")
            return iou_wu, model_id, "COMPLETE"

        print(f"\n  ⚠ Result not found for {model_id}")
        return None, model_id, "FAIL"

    def _write_config(self, config):
        os.makedirs(os.path.dirname(self.synthetic_config_path), exist_ok=True)
        with open(self.synthetic_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"  ▸ config.json saved: {self.synthetic_config_path}")

    def _write_info(self):
        with open("info.json", "w", encoding="utf-8") as f:
            json.dump(self.training_config, f, ensure_ascii=False, indent=4)
        print(f"  ▸ info.json saved (fixed training config)")

    def _execute_notebook(self, path):
        p = Path(path)
        dir_path = str(p.parent)
        name = p.stem
        ext = p.suffix

        os.makedirs("logs", exist_ok=True)
        out = os.path.join("logs", f"{name}_out{ext}")

        print(f"  ▶ Running: {name}")
        try:
            pm.execute_notebook(
                str(path), out,
                kernel_name=self.kernel,
                log_output=True,
                progress_bar=True,
                cwd=dir_path,
            )
            print(f"  ✔ {name} completed.")
            return True
        except Exception as e:
            print(f"  ✘ ERROR in {name}: {e}")
            traceback.print_exc()
            return False

    def _list_model_dirs(self):
        return set(
            os.path.basename(d)
            for d in glob.glob(os.path.join(self.database_dir, "model_*"))
        )

    def _detect_new_model(self, models_before):
        models_after = self._list_model_dirs()
        new_models = models_after - models_before

        if new_models:
            return sorted(new_models, key=lambda x: int(x.split("_")[-1]))[-1]

        all_dirs = glob.glob(os.path.join(self.database_dir, "model_*"))
        if all_dirs:
            all_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
            return os.path.basename(all_dirs[-1])

        return None

    def _read_result(self, model_id):
        if model_id is None:
            return None
        info_path = os.path.join(self.database_dir, model_id, "info.json")
        if not os.path.exists(info_path):
            return None
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("iou_wu")
            if val is not None and isinstance(val, (int, float)) and val > 0:
                return float(val)
            return None
        except Exception:
            return None


class StudyManager:

    STUDY_NAME = "falhas_synthetic_hpo_v4"
    DIRECTION = "maximize"

    TRAINING_CONFIG = {
        "network": "unet3d_v2",
        "dataset": "dataset_wu",
        "img_size": None,
        "lr": 0.001,
        "loss": "dice_focal",
        "batch_size": 4,
        "scheduler": "plateau",
        "dropout": 0.1,
        "num_filters": 16,
    }

    PHASE_EXPLORATION_RATES = [
        (30,  0.30),
        (80,  0.20),
        (150, 0.10),
    ]

    N_STARTUP_TRIALS = 15
    N_TRIALS_DEFAULT = 150
    TPE_SEED = 42
    RANDOM_SEED = 123
    CMAES_SEED = 7
    CMAES_ACTIVATION_TRIAL = 60

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.db_path = os.path.join(base_dir, "Searcher", "study.db")
        self.csv_path = os.path.join(base_dir, "Searcher", "historico_otimizacao.csv")
        self.database_dir = os.path.join(base_dir, "Searcher", "database")
        self.synthetic_config_path = os.path.join(base_dir, "Synthetic", "config.json")

        self.search_space = SearchSpace()
        self.history_manager = HistoricalDataManager(self.database_dir, self.search_space)
        self.pipeline = PipelineExecutor(
            synthetic_config_path=self.synthetic_config_path,
            database_dir=self.database_dir,
            training_config=self.TRAINING_CONFIG,
            kernel="python3",
        )

        self.tpe_sampler = None
        self.random_sampler = None
        self.cmaes_sampler = None
        self.study = None

    def _storage_url(self):
        return f"sqlite:///{self.db_path}"

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.database_dir, exist_ok=True)

    def _create_samplers(self):
        self.tpe_sampler = TPESampler(
            seed=self.TPE_SEED,
            multivariate=True,
            n_startup_trials=self.N_STARTUP_TRIALS,
        )
        self.random_sampler = RandomSampler(seed=self.RANDOM_SEED)
        self.cmaes_sampler = CmaEsSampler(
            seed=self.CMAES_SEED,
            n_startup_trials=self.N_STARTUP_TRIALS,
        )

    def _create_or_load_study(self):
        try:
            self.study = optuna.create_study(
                study_name=self.STUDY_NAME,
                storage=self._storage_url(),
                direction=self.DIRECTION,
                sampler=self.tpe_sampler,
                load_if_exists=True,
            )
        except Exception as e:
            print(f"  ⚠ Study load failed ({e}), recreating database...")
            db_file = Path(self.db_path)
            if db_file.exists():
                db_file.unlink()
            self.study = optuna.create_study(
                study_name=self.STUDY_NAME,
                storage=self._storage_url(),
                direction=self.DIRECTION,
                sampler=self.tpe_sampler,
            )

    def initialize(self):
        print("=" * 60)
        print(" 🔬 Bayesian Optimization — Geological Parameters")
        print(f"    Study: {self.STUDY_NAME}")
        print(f"    DB: {self.db_path}")
        print("=" * 60)

        self._ensure_directories()
        self._create_samplers()
        self._create_or_load_study()

        print("\n📂 Scanning Searcher/database/ for historical data...")
        self.history_manager.inject_into_study(self.study, delete_failed=False)

        completed = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        print(f"\n📊 Study state:")
        print(f"   Completed trials: {len(completed)}")
        if completed:
            print(f"   Best iou_wu: {self.study.best_value:.4f}")
            print(f"   Best trial: #{self.study.best_trial.number}")

    def _get_exploration_rate(self, n_completed):
        for threshold, rate in self.PHASE_EXPLORATION_RATES:
            if n_completed < threshold:
                return rate
        return self.PHASE_EXPLORATION_RATES[-1][1]

    def _select_sampler(self, n_completed):
        exploration_rate = self._get_exploration_rate(n_completed)
        cmaes_rate = 0.10 if n_completed >= self.CMAES_ACTIVATION_TRIAL else 0.0

        roll = random.random()

        if roll < exploration_rate:
            self.study.sampler = self.random_sampler
            return "RANDOM"

        if roll < exploration_rate + cmaes_rate:
            self.study.sampler = self.cmaes_sampler
            return "CMA-ES"

        self.study.sampler = self.tpe_sampler
        return "TPE"

    def _objective(self, trial):
        trial_number = trial.number
        print(f"\n{'='*60}")
        print(f" TRIAL {trial_number}")
        print(f"{'='*60}")

        config = self.search_space.suggest_config(trial)

        print(f"\n  Proposed parameters:")
        for key, val in config.items():
            print(f"    {key}: {val}")

        iou_wu, model_id, status = self.pipeline.execute_trial(config, trial_number)

        if iou_wu is None:
            iou_wu = 0.0
            print(f"\n  ✘ Trial {trial_number} failed. Returning iou_wu = 0.0")

        flat_params = self.search_space.config_to_flat_params(config)
        self._update_csv(trial_number, flat_params, iou_wu, status, model_id)

        trial.set_user_attr("model_id", model_id or "N/A")
        trial.set_user_attr("status", status)

        print(f"\n{'='*60}")
        print(f" END TRIAL {trial_number} — iou_wu = {iou_wu:.4f} [{status}]")
        print(f"{'='*60}\n")

        return iou_wu

    def _update_csv(self, trial_number, flat_params, iou_wu, status, model_id):
        file_exists = os.path.exists(self.csv_path)
        sorted_keys = sorted(flat_params.keys())

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["trial", "timestamp", "model_id", "status", "iou_wu"] + sorted_keys
                writer.writerow(header)
            row = [
                trial_number,
                datetime.now().isoformat(),
                model_id or "N/A",
                status,
                f"{iou_wu:.6f}" if iou_wu is not None else "0.000000",
            ]
            row += [flat_params.get(k, "") for k in sorted_keys]
            writer.writerow(row)

        print(f"  ▸ CSV updated: {self.csv_path}")

    def run(self, n_trials=None):
        if n_trials is None:
            n_trials = self.N_TRIALS_DEFAULT

        self.initialize()

        print(f"\n🚀 Starting optimization ({n_trials} trials)...")
        print(f"   Phase-based exploration: adaptive rates\n")

        for i in range(n_trials):
            n_completed = len([
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ])
            sampler_name = self._select_sampler(n_completed)
            exploration_rate = self._get_exploration_rate(n_completed)

            print(f"\n  {'🎲' if sampler_name == 'RANDOM' else '🧬' if sampler_name == 'CMA-ES' else '🧠'} "
                  f"Trial {i+1}/{n_trials} → {sampler_name} "
                  f"(explore={int(exploration_rate*100)}%)")

            try:
                self.study.optimize(self._objective, n_trials=1)
            except KeyboardInterrupt:
                print("\n\n  ⚠ Manual interruption. Saving progress...")
                break
            except Exception as e:
                print(f"\n  ✘ Trial error: {e}")
                traceback.print_exc()
                continue

        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print(" 📋 OPTIMIZATION SUMMARY")
        print("=" * 60)

        completed = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        print(f"   Total trials: {len(self.study.trials)}")
        print(f"   Completed: {len(completed)}")

        if completed:
            print(f"   Best iou_wu:   {self.study.best_value:.4f}")
            print(f"   Best trial:    #{self.study.best_trial.number}")
            print(f"\n   Best parameters:")
            for key, val in sorted(self.study.best_params.items()):
                print(f"     {key}: {val}")

        print(f"\n   History CSV: {self.csv_path}")
        print(f"   Study DB:    {self.db_path}")
        print("=" * 60)


if __name__ == "__main__":
    manager = StudyManager()
    manager.run()