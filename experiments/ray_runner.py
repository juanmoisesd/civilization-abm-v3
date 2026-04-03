"""
ray_runner.py — Runner distribuido con Ray para Civilization-ABM v3.

Reemplaza multiprocessing.Pool de v2 con Ray, que permite:
  - Distribuir en múltiples máquinas
  - Dashboard de monitoreo en tiempo real
  - Mejor manejo de memoria compartida
  - Fácil escalado a 1000s de condiciones

Uso:
    python -m experiments.ray_runner
    python -m experiments.ray_runner --condition baseline
    python -m experiments.ray_runner --jobs 16 --no-ray  # fuerza multiprocessing

Requiere: pip install ray[default]
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

from model.model_v3 import CivilModelV3
from analysis.metrics import summary_statistics

CONFIGS_PATH = Path(__file__).parent / "configs_v3.yaml"
RESULTS_DIR  = Path("results/v3")


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

def _run_single(task: dict) -> dict:
    """Ejecuta una replicación y retorna métricas de resumen."""
    condition = task["condition"]
    rep       = task["rep"]
    seed      = task["seed"]
    steps     = task["steps"]
    params    = task["params"]

    try:
        model = CivilModelV3(seed=seed, **params)
        for _ in range(steps):
            model.step()
            if model.alive_count < 2:
                break

        df    = model.get_model_vars_dataframe()
        stats = summary_statistics(df)
        stats.update({"condition": condition, "rep": rep, "seed": seed,
                      "actual_steps": model._step_count})
        return stats
    except Exception as e:
        return {"condition": condition, "rep": rep, "seed": seed, "error": str(e)}


# Ray remote version
if _RAY_AVAILABLE:
    @ray.remote
    def _run_single_ray(task: dict) -> dict:
        return _run_single(task)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_config(path: Path = CONFIGS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_tasks(config: dict, condition_filter: Optional[str] = None) -> list:
    global_cfg = config["global"]
    steps      = global_cfg["steps"]
    reps       = global_cfg["replications"]
    seed_base  = global_cfg.get("seed_base", 42)

    global_params = {
        "N":                  global_cfg.get("N", 500),
        "landscape_width":    global_cfg.get("landscape_width", 35),
        "landscape_height":   global_cfg.get("landscape_height", 35),
        "growth_rate":        global_cfg.get("growth_rate", 0.5),
        "metabolism":         global_cfg.get("metabolism", 1.0),
        "mutation_rate":      global_cfg.get("mutation_rate", 0.02),
        "evolution_interval": global_cfg.get("evolution_interval", 5),
        "use_memory":         global_cfg.get("use_memory", False),
    }

    overridable = [
        "N", "initial_inequality", "tax_policy", "network_type",
        "enforce_floor", "landscape_peaks", "metabolism",
        "growth_rate", "mutation_rate", "evolution_interval",
        "use_memory",
    ]

    tasks = []
    for cond in config["conditions"]:
        name = cond["name"]
        if condition_filter and name != condition_filter:
            continue
        params = {**global_params}
        for key in overridable:
            if key in cond:
                params[key] = cond[key]
        for rep in range(reps):
            seed = (seed_base * 1000 + hash(name) % 10000 + rep) % (2**31)
            tasks.append({"condition": name, "rep": rep, "seed": seed,
                          "steps": steps, "params": params})
    return tasks


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_with_ray(tasks: list, n_jobs: int) -> list:
    """Ejecuta tareas con Ray."""
    if not ray.is_initialized():
        ray.init(num_cpus=n_jobs, ignore_reinit_error=True)

    futures = [_run_single_ray.remote(t) for t in tasks]

    results = []
    total = len(futures)
    done = 0
    while futures:
        ready, futures = ray.wait(futures, num_returns=min(32, len(futures)))
        batch = ray.get(ready)
        results.extend(batch)
        done += len(batch)
        print(f"\r  Progreso: {done}/{total} ({100*done//total}%)", end="", flush=True)
    print()
    return results


def run_with_multiprocessing(tasks: list, n_jobs: int) -> list:
    """Fallback con multiprocessing estándar."""
    try:
        from tqdm import tqdm
        with mp.Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_single, tasks, chunksize=2),
                total=len(tasks), desc="Simulando (multiprocessing)"
            ))
    except ImportError:
        with mp.Pool(processes=n_jobs) as pool:
            results = list(pool.imap_unordered(_run_single, tasks, chunksize=2))
    return results


def run_all(
    config: dict,
    condition_filter: Optional[str] = None,
    n_jobs: Optional[int] = None,
    force_mp: bool = False,
) -> pd.DataFrame:
    tasks  = build_tasks(config, condition_filter)
    if not tasks:
        print(f"No se encontró condición: {condition_filter}")
        return pd.DataFrame()

    n_jobs = n_jobs or config["global"].get("n_jobs", mp.cpu_count())
    n_jobs = min(n_jobs, mp.cpu_count())

    n_conds = len({t["condition"] for t in tasks})
    print(f"\nCivilization-ABM v3 — Experimentos")
    print(f"  Condiciones: {n_conds}  |  Tareas: {len(tasks)}  |  Workers: {n_jobs}")
    print(f"  Backend: {'Ray' if (_RAY_AVAILABLE and not force_mp) else 'multiprocessing'}")
    print(f"  Inicio: {time.strftime('%H:%M:%S')}\n")

    t0 = time.time()

    if _RAY_AVAILABLE and not force_mp:
        raw = run_with_ray(tasks, n_jobs)
    else:
        raw = run_with_multiprocessing(tasks, n_jobs)

    elapsed = time.time() - t0
    print(f"  Completado en {elapsed:.1f}s  ({elapsed/len(tasks):.2f}s/tarea)")

    return pd.DataFrame(raw)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Civilization-ABM v3")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--jobs",      type=int, default=None)
    parser.add_argument("--no-ray",    action="store_true", dest="no_ray")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_all(config, condition_filter=args.condition,
                      n_jobs=args.jobs, force_mp=args.no_ray)

    if results.empty:
        print("Sin resultados.")
        return

    out = RESULTS_DIR / "all_conditions_v3.csv"
    results.to_csv(out, index=False)
    print(f"\nResultados: {out}  ({len(results)} filas)")

    if "condition" in results.columns and "gini_final" in results.columns:
        summary = results.groupby("condition")["gini_final"].agg(
            ["mean", "std", "min", "max"]).round(3)
        print(f"\nGini final por condición:\n{summary.to_string()}")

    if _RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()

    print(f"\nCompletado. {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
