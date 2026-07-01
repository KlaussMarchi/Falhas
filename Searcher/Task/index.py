#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
 Task/index.py — Maestro de Otimização Bayesiana com Optuna
==========================================================================
 Este script substitui a execução linear do pipeline por uma busca
 Bayesiana inteligente utilizando a biblioteca Optuna (TPE Sampler).

 Objetivo: Maximizar a métrica `iou_wu` (IoU no dataset de validação Wu)
 alterando EXCLUSIVAMENTE os parâmetros geológicos do arquivo
 Synthetic/config.json.

 Fluxo por Trial:
   1. Optuna propõe novos parâmetros geológicos
   2. Grava config.json com os parâmetros propostos
   3. Grava info.json com hiperparâmetros de rede (FIXOS)
   4. Executa notebooks via Papermill:
      a) Synthetic/Generate.ipynb  — Gera dados sintéticos
      b) Dataset/dataset_synthetic/Format.ipynb — Pré-processa
      c) Model/Analysis.ipynb — Treina a U-Net 3D
      d) Model/Predict.ipynb — Avalia no dataset_wu
   5. Coleta iou_wu do resultado em Model/Backup
   6. Atualiza CSV de histórico

 Resiliência: O study é persistido em SQLite (Model/Backup/study.db).
 Se o terminal for fechado, basta reiniciar este script.

 Histórico Manual: Resultados anteriores em Model/Backup/model_*
 são injetados automaticamente no study como trials completadas.
==========================================================================
"""

import optuna
from optuna.distributions import IntDistribution, FloatDistribution
from optuna.trial import create_trial, TrialState
import papermill as pm
from pathlib import Path
import os
import json
import glob
import csv
import random
import traceback
from datetime import datetime


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

# Número máximo de trials por sessão de otimização.
# O script pode ser reiniciado e continuará de onde parou.
N_TRIALS = 50

# Direção da otimização: maximizar iou_wu
DIRECTION = "maximize"

# Fração de trials aleatórias para escapar de máximos locais.
# 0.2 = 20% random, 80% TPE Bayesiano. Valor recomendado: 0.15-0.25
EXPLORATION_RATE = 0.20

# Nome do estudo Optuna (identificador único)
STUDY_NAME = "falhas_synthetic_hpo_v3"

# Caminho do banco de dados SQLite para persistência
STUDY_DB_PATH = os.path.abspath("../Model/Backup/study.db")
STUDY_STORAGE = f"sqlite:///{STUDY_DB_PATH}"

# Caminho do CSV de histórico legível por humanos
CSV_HISTORY_PATH = os.path.abspath("../Model/Backup/historico_otimizacao.csv")

# Caminho da pasta de resultados
DATABASE_DIR = os.path.abspath("../Model/Backup")

# Caminho do config.json de geração sintética
SYNTHETIC_CONFIG_PATH = os.path.abspath("../Synthetic/config.json")

# Kernel Jupyter a ser utilizado pelo Papermill
JUPYTER_KERNEL = "python3"


# ============================================================
# CONFIGURAÇÕES FIXAS DE TREINAMENTO (NÃO OTIMIZADAS)
# ============================================================
# Estes valores são escritos em info.json a cada trial e NÃO
# são alterados pelo Optuna. Apenas config.json é otimizado.

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


# ============================================================
# ESPAÇO DE BUSCA — Parâmetros Geológicos (config.json)
# ============================================================
# Cada entrada define um parâmetro do config.json e seus limites.
#
# Tipos:
#   "int_range"    → par [min, max] de inteiros
#   "float_range"  → par [min, max] de floats
#   "float_scalar" → valor float único
#
# RODADA 3 — Bounds recalibrados com base em 35 trials completas.
# Melhor modelo: model_21 (IoU 0.673). Dois clusters de alta
# performance foram identificados e os bounds agora abrangem ambos.
# Correção crítica: bounds anteriores excluíam regiões dos TOP models.

SEARCH_SPACE = [
    # ── Parâmetros de Camadas ──
    # TOP5 usam 68-85; bounds antigos (30,75) excluíam model_21/20
    {"name": "layerRange",      "type": "int_range",
     "min_bounds": (10, 100),   "max_bounds": (130, 350)},
    # TOP10: TODOS usam min=1 (corr -0.49**). Fixar min=1.
    {"name": "layerThickness",  "type": "int_range",
     "min_bounds": (1, 1),      "max_bounds": (2, 5)},

    # ── Parâmetros de Dobras (Folds) ──
    # TOP5 usam 13-25; bounds antigos (3,15) excluíam model_21/20
    {"name": "foldCount",       "type": "int_range",
     "min_bounds": (10, 28),    "max_bounds": (38, 60)},
    # TOP5 usam 16-28; bounds antigos (3,18) excluíam model_21/20
    {"name": "foldSigma",       "type": "int_range",
     "min_bounds": (12, 30),    "max_bounds": (35, 75)},
    # TOP10 usam -34 a -27 (min), -2 a 8 (max); bounds antigos muito largos
    {"name": "foldAmplitude",   "type": "int_range",
     "min_bounds": (-38, -24),  "max_bounds": (-5, 10)},
    # Correlação -0.56*** (menor = melhor); TOP10: 0.65-1.34
    {"name": "foldDamping",     "type": "float_scalar",
     "bounds": (0.5, 1.5)},
    # TOP10: min=-0.79..0.02, max=3.5..4.76; apertar
    {"name": "foldBaseShift",   "type": "float_range",
     "min_bounds": (-1.0, 0.2), "max_bounds": (3.0, 5.0)},

    # ── Parâmetros de Cisalhamento (Shear) ──
    # TOP10: offset_min -9.14..-7.72; apertar range
    {"name": "shearOffset",     "type": "float_range",
     "min_bounds": (-9.5, -7.5), "max_bounds": (2.5, 6.0)},
    # Corr -0.38* no max; TOP10: min=-0.124..-0.094, max=0.001..0.026
    {"name": "shearGradient",   "type": "float_range",
     "min_bounds": (-0.13, -0.09), "max_bounds": (0.00, 0.03)},

    # ── Parâmetros de Falhas (Faults) ──
    # TOP5 usam min=2-6; bounds antigos (4,6) excluíam model_21/20
    {"name": "faultCount",      "type": "int_range",
     "min_bounds": (1, 6),      "max_bounds": (7, 12)},
    # Corr +0.44** no min (maior=melhor); TOP10: min=13-15, max=26-38
    {"name": "faultThrow",      "type": "int_range",
     "min_bounds": (12, 16),    "max_bounds": (28, 42)},
    # MAIOR correlação +0.67*** no min! TOP5: min=56-65, max=75-81
    {"name": "faultDipAngle",   "type": "int_range",
     "min_bounds": (50, 68),    "max_bounds": (74, 82)},
    # TOP5 usam 4.39-5.77; bounds antigos (5.5,7.5) excluíam model_21/20
    {"name": "faultRoughness",  "type": "float_scalar",
     "bounds": (3.0, 7.0)},
    # TOP10: 5.47-9.59; apertar range
    {"name": "faultRoughSigma", "type": "float_scalar",
     "bounds": (4.5, 10.0)},
    # Corr -0.58*** no max (menor=melhor); TOP10: min=32-52, max=49-80
    {"name": "faultDecaySigma", "type": "int_range",
     "min_bounds": (30, 55),    "max_bounds": (45, 85)},
    # TOP10: 0.79-1.16; apertar
    {"name": "faultZoneWidth",  "type": "float_scalar",
     "bounds": (0.40, 1.50)},
    # TOP10: 0.57-0.72; apertar
    {"name": "faultThreshold",  "type": "float_scalar",
     "bounds": (0.10, 0.9)},
    # Corr -0.35* (menor=melhor); TOP5 usam 0.10-0.35; bound (0.30,0.70) excluía!
    {"name": "faultCurveProb",  "type": "float_scalar",
     "bounds": (0.005, 0.5)},
    # Corr +0.36*; TOP10: 4.82-6.70; subir mínimo
    {"name": "faultCurveMax",   "type": "float_scalar",
     "bounds": (4.0, 15.0)},

    # ── Parâmetros de Wavelet ──
    # Corr -0.60*** Spearman no max (menor=melhor); TOP10: min=77-89, max=80-109
    {"name": "waveletFreq",     "type": "int_range",
     "min_bounds": (60, 100),    "max_bounds": (78, 110)},
    # Corr +0.42* (maior=melhor); TOP10: 0.098-0.113
    {"name": "waveletDuration", "type": "float_scalar",
     "bounds": (0.09, 0.12)},
    # TOP10: 0.0017-0.0026; apertar
    {"name": "waveletDt",       "type": "float_scalar",
     "bounds": (0.0005, 0.0030)},

    # ── Parâmetros de Ruído ──
    # Corr +0.45** no max (maior=melhor); TOP10: min=0.015-0.027, max=0.49-0.56
    {"name": "noiseLevel",      "type": "float_range",
     "min_bounds": (0.01, 0.03), "max_bounds": (0.48, 0.65)},
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def build_distributions():
    """
    Constrói o dicionário de distribuições Optuna a partir do
    SEARCH_SPACE. Necessário para injetar trials históricas.
    """
    dists = {}
    for p in SEARCH_SPACE:
        if p["type"] == "int_range":
            dists[f"{p['name']}_min"] = IntDistribution(*p["min_bounds"])
            dists[f"{p['name']}_max"] = IntDistribution(*p["max_bounds"])
        elif p["type"] == "float_range":
            dists[f"{p['name']}_min"] = FloatDistribution(*p["min_bounds"])
            dists[f"{p['name']}_max"] = FloatDistribution(*p["max_bounds"])
        elif p["type"] == "float_scalar":
            dists[p["name"]] = FloatDistribution(*p["bounds"])
    return dists


def suggest_config(trial):
    """
    Gera um dicionário de config.json a partir das sugestões do Optuna.
    Para parâmetros do tipo 'range', garante que min <= max via swap.
    """
    config = {}
    for p in SEARCH_SPACE:
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


def config_to_flat_params(config):
    """
    Converte um dicionário de config.json em um dicionário plano
    compatível com os nomes de parâmetros do Optuna.
    Ex: {"layerRange": [71, 212]} → {"layerRange_min": 71, "layerRange_max": 212}
    """
    params = {}
    for p in SEARCH_SPACE:
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


def clamp_params_to_bounds(params, distributions):
    """
    Ajusta os valores dos parâmetros para ficarem dentro dos limites
    das distribuições. Necessário ao injetar trials históricas cujos
    valores podem estar ligeiramente fora dos bounds definidos.
    """
    clamped = {}
    for key, val in params.items():
        if key in distributions:
            dist = distributions[key]
            low = dist.low
            high = dist.high
            clamped[key] = max(low, min(high, val))
            # Converter para int se a distribuição for inteira
            if isinstance(dist, IntDistribution):
                clamped[key] = int(round(clamped[key]))
        else:
            clamped[key] = val
    return clamped


def execute_notebook(path):
    """
    Executa um notebook Jupyter via Papermill.
    Retorna True se bem-sucedido, False caso contrário.
    """
    p = Path(path)
    dir_path = str(p.parent)
    name = p.stem
    ext = p.suffix

    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", f"{name}_out{ext}")

    print(f"  ▶ Executando: {name}")
    try:
        pm.execute_notebook(
            str(path), out,
            kernel_name=JUPYTER_KERNEL,
            log_output=True,
            progress_bar=True,
            cwd=dir_path,
        )
        print(f"  ✔ {name} concluído com sucesso.")
        return True
    except Exception as e:
        print(f"  ✘ ERRO em {name}: {e}")
        traceback.print_exc()
        return False


def get_all_database_results():
    """
    Lê todos os resultados existentes em Model/Backup/model_*/synthetic.json.
    Retorna uma lista de dicts: [{"model": "model_1", "iou_wu": 0.59, "config": {...}}, ...]
    """
    results = []
    pattern = os.path.join(DATABASE_DIR, "model_*", "synthetic.json")
    for info_path in sorted(glob.glob(pattern)):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "iou_wu" in data and "config" in data:
                results.append(data)
        except Exception as e:
            print(f"  ⚠ Aviso: não foi possível ler {info_path}: {e}")
    return results


def get_latest_model_id():
    """
    Retorna o model_id mais recente em Model/Backup/.
    """
    pattern = os.path.join(DATABASE_DIR, "model_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
    return os.path.basename(dirs[-1])


def read_result(model_id):
    """
    Lê o iou_wu de um resultado específico em Model/Backup.
    Retorna o valor float ou None se não encontrado.
    """
    if model_id is None:
        return None
    info_path = os.path.join(DATABASE_DIR, model_id, "synthetic.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("iou_wu", None)
    except Exception:
        return None


def inject_historical_trials(study, distributions):
    """
    Injeta trials históricas (resultados manuais em Model/Backup)
    no study do Optuna como trials completadas.
    Usa study.user_attrs para rastrear quais model_ids já foram injetados.
    """
    already_injected = set(study.user_attrs.get("injected_model_ids", []))
    results = get_all_database_results()
    injected_count = 0

    for result in results:
        model_id = result.get("model", "unknown")
        if model_id in already_injected:
            continue

        config = result.get("config", {})
        iou_wu = result.get("iou_wu", 0.0)

        # Converter config para parâmetros planos
        flat_params = config_to_flat_params(config)

        # Ajustar valores para dentro dos bounds
        clamped_params = clamp_params_to_bounds(flat_params, distributions)

        # Verificar se todos os parâmetros necessários estão presentes
        missing = set(distributions.keys()) - set(clamped_params.keys())
        if missing:
            print(f"  ⚠ {model_id}: parâmetros faltando ({missing}), pulando injeção.")
            continue

        # Criar e adicionar a trial
        try:
            frozen_trial = create_trial(
                params=clamped_params,
                distributions=distributions,
                values=[iou_wu],
                state=TrialState.COMPLETE,
            )
            study.add_trial(frozen_trial)
            already_injected.add(model_id)
            injected_count += 1
            print(f"  ✔ Trial injetada: {model_id} → iou_wu = {iou_wu:.4f}")
        except Exception as e:
            print(f"  ⚠ Falha ao injetar {model_id}: {e}")

    # Salvar lista de model_ids injetados
    study.set_user_attr("injected_model_ids", list(already_injected))

    if injected_count > 0:
        print(f"\n  ▸ {injected_count} trial(s) históricas injetadas no study.")
    else:
        print(f"\n  ▸ Nenhuma nova trial histórica para injetar.")


def update_csv_history(trial_number, flat_params, iou_wu, status, model_id):
    """
    Atualiza o CSV de histórico com os resultados da trial.
    Cria o arquivo e cabeçalho se não existir.
    """
    file_exists = os.path.exists(CSV_HISTORY_PATH)
    sorted_keys = sorted(flat_params.keys())

    with open(CSV_HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
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

    print(f"  ▸ CSV atualizado: {CSV_HISTORY_PATH}")


# ============================================================
# FUNÇÃO OBJETIVO (chamada pelo Optuna a cada trial)
# ============================================================

def objective(trial):
    """
    Função objetivo do Optuna. Executa o pipeline completo:
    1. Gera config.json com parâmetros propostos
    2. Salva info.json com configuração de treino fixa
    3. Executa os 4 notebooks
    4. Coleta e retorna iou_wu

    Em caso de falha em qualquer notebook, retorna 0.0.
    """
    trial_number = trial.number
    print(f"\n{'='*60}")
    print(f" TRIAL {trial_number}")
    print(f"{'='*60}")

    # 1. Gerar config.json com parâmetros sugeridos pelo Optuna
    config = suggest_config(trial)

    print(f"\n  Parâmetros propostos pelo Optuna:")
    for key, val in config.items():
        print(f"    {key}: {val}")

    # Salvar config.json
    with open(SYNTHETIC_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"\n  ▸ config.json salvo em: {SYNTHETIC_CONFIG_PATH}")

    # 2. Salvar info.json com configuração de treino fixa
    with open("info.json", "w", encoding="utf-8") as f:
        json.dump(TRAINING_CONFIG, f, ensure_ascii=False, indent=4)
    print(f"  ▸ info.json salvo (configuração de treino fixa)")

    # Anotar model_id antes da execução para detectar o novo
    existing_models_before = set(
        os.path.basename(d)
        for d in glob.glob(os.path.join(DATABASE_DIR, "model_*"))
    )

    # 3. Executar notebooks via Papermill
    notebooks = [
        "../Synthetic/Generate.ipynb",
        "../Dataset/dataset_synthetic/Format.ipynb",
        "../Model/Analysis.ipynb",
        "../Model/Predict.ipynb",
    ]

    pipeline_success = True
    for nb_path in notebooks:
        success = execute_notebook(nb_path)
        if not success:
            pipeline_success = False
            print(f"\n  ✘ Pipeline interrompido por falha em {nb_path}")
            break

    # 4. Coletar resultado
    iou_wu = 0.0
    model_id = None
    status = "FAIL"

    if pipeline_success:
        # Encontrar o novo model_id criado
        existing_models_after = set(
            os.path.basename(d)
            for d in glob.glob(os.path.join(DATABASE_DIR, "model_*"))
        )
        new_models = existing_models_after - existing_models_before

        if new_models:
            # Pegar o mais recente
            model_id = sorted(
                new_models,
                key=lambda x: int(x.split("_")[-1])
            )[-1]
        else:
            # Fallback: pegar o último modelo
            model_id = get_latest_model_id()

        result = read_result(model_id)
        if result is not None:
            iou_wu = result
            status = "COMPLETE"
            print(f"\n  ★ iou_wu = {iou_wu:.4f} ({model_id})")
        else:
            print(f"\n  ⚠ Resultado não encontrado para {model_id}")
            iou_wu = 0.0
            status = "FAIL"
    else:
        print(f"\n  ✘ Trial {trial_number} falhou. Retornando iou_wu = 0.0")

    # 5. Atualizar CSV de histórico
    flat_params = config_to_flat_params(config)
    update_csv_history(trial_number, flat_params, iou_wu, status, model_id)

    # 6. Registrar model_id como atributo da trial (para rastreabilidade)
    trial.set_user_attr("model_id", model_id or "N/A")
    trial.set_user_attr("status", status)

    print(f"\n{'='*60}")
    print(f" FIM TRIAL {trial_number} — iou_wu = {iou_wu:.4f} [{status}]")
    print(f"{'='*60}\n")

    return iou_wu


# ============================================================
# PONTO DE ENTRADA PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print(" 🔬 Otimização Bayesiana de Parâmetros Geológicos")
    print("    Objetivo: Maximizar iou_wu via Optuna (TPE Sampler)")
    print(f"    Study: {STUDY_NAME}")
    print(f"    Banco: {STUDY_DB_PATH}")
    print(f"    Trials planejadas: {N_TRIALS}")
    print("=" * 60)

    # Garantir que o diretório do banco existe
    os.makedirs(os.path.dirname(STUDY_DB_PATH), exist_ok=True)
    os.makedirs(DATABASE_DIR, exist_ok=True)

    # Criar sampler híbrido: TPE (Bayesiano) + Random (exploração)
    # A cada trial, há EXPLORATION_RATE de chance de ser 100% aleatória,
    # forçando o Optuna a explorar regiões inesperadas e evitar
    # convergência prematura para máximos locais.
    tpe_sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,     # Modela correlações entre parâmetros
        n_startup_trials=10,   # 10 primeiras trials são aleatórias
    )
    random_sampler = optuna.samplers.RandomSampler(seed=123)

    print(f"\n🎲 Sampler híbrido: {int((1-EXPLORATION_RATE)*100)}% TPE + "
          f"{int(EXPLORATION_RATE*100)}% Random (exploração)")

    # Criar ou carregar o study com persistência SQLite
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        direction=DIRECTION,
        sampler=tpe_sampler,
        load_if_exists=True,  # ← Resiliência: retoma de onde parou
    )

    # Injetar trials históricas de Model/Backup/
    print("\n📂 Verificando histórico em Model/Backup/ ...")
    distributions = build_distributions()
    inject_historical_trials(study, distributions)

    # Exibir estado atual do study
    n_completed = len([
        t for t in study.trials if t.state == TrialState.COMPLETE
    ])
    print(f"\n📊 Estado do Study:")
    print(f"   Trials completadas: {n_completed}")
    if study.best_trial:
        print(f"   Melhor iou_wu: {study.best_value:.4f}")
        print(f"   Melhor trial: #{study.best_trial.number}")

    # Executar otimização com sampler híbrido (TPE + Random)
    # Em vez de study.optimize(), usamos um loop manual que troca
    # o sampler do study antes de cada trial. Isso garante que
    # ~20% das trials sejam puramente aleatórias (exploração),
    # evitando convergência prematura para máximos locais.
    print(f"\n🚀 Iniciando otimização ({N_TRIALS} trials)...")
    print(f"   🎲 Exploração aleatória: {int(EXPLORATION_RATE*100)}% das trials\n")

    for i in range(N_TRIALS):
        # Decidir se esta trial é exploratória (random) ou guiada (TPE)
        is_exploration = random.random() < EXPLORATION_RATE

        if is_exploration:
            study.sampler = random_sampler
            print(f"\n  🎲 Trial {i+1}/{N_TRIALS} → EXPLORAÇÃO ALEATÓRIA")
        else:
            study.sampler = tpe_sampler
            print(f"\n  🧠 Trial {i+1}/{N_TRIALS} → TPE BAYESIANO")

        try:
            study.optimize(objective, n_trials=1)
        except KeyboardInterrupt:
            print("\n\n  ⚠ Interrupção manual detectada. Salvando progresso...")
            break
        except Exception as e:
            print(f"\n  ✘ Erro na trial: {e}")
            traceback.print_exc()
            continue

    # Resumo final
    print("\n" + "=" * 60)
    print(" 📋 RESUMO FINAL DA OTIMIZAÇÃO")
    print("=" * 60)
    print(f"   Total de trials: {len(study.trials)}")
    print(f"   Melhor iou_wu:   {study.best_value:.4f}")
    print(f"   Melhor trial:    #{study.best_trial.number}")
    print(f"\n   Melhores parâmetros:")
    for key, val in sorted(study.best_params.items()):
        print(f"     {key}: {val}")
    print(f"\n   Histórico salvo em: {CSV_HISTORY_PATH}")
    print(f"   Study persistido em: {STUDY_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()