"""
fitting.py — Calibración del modelo contra datos reales del Banco Mundial (v3).

Objetivo: encontrar la combinación de parámetros del modelo que produce
una distribución de Gini en el rango empírico observado en datos reales.

Método: búsqueda en cuadrícula (grid search) sobre los parámetros clave,
con comparación por distancia cuadrática media (RMSE) al target Gini.

v3: Usa CivilModelV3 en lugar de CivilModelV2.
"""

import itertools
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from calibration.worldbank import world_gini_range
from model.model_v3 import CivilModelV3


# -----------------------------------------------------------------------
# Configuración de la calibración
# -----------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """
    Define el espacio de búsqueda de parámetros y la configuración
    de la calibración.
    """
    # Parámetros a explorar (grid search)
    initial_inequality_values: list = field(default_factory=lambda: [0.5, 0.8, 1.2, 1.5])
    tax_policies: list = field(default_factory=lambda: ["flat", "progressive", None])
    enforce_floor_values: list = field(default_factory=lambda: [False, True])
    landscape_peaks_values: list = field(default_factory=lambda: [2, 3])

    # Fijos durante calibración
    N: int = 500
    steps: int = 200
    replications: int = 10
    n_jobs: int = 4

    # Target empírico (se sobreescribe con datos reales si están disponibles)
    target_gini: float = 0.38   # mediana mundial aproximada
    target_tolerance: float = 0.05


def _run_single(args) -> dict:
    """Ejecuta una replicación y devuelve el Gini final."""
    params, seed, steps = args
    model = CivilModelV3(seed=seed, **params)
    for _ in range(steps):
        model.step()
        if model.alive_count < 2:
            break
    df = model.get_model_vars_dataframe()
    gini_final = float(df["gini"].iloc[-1]) if len(df) > 0 else 0.0
    return {"params": params, "seed": seed, "gini": gini_final}


# -----------------------------------------------------------------------
# Calibración por grid search
# -----------------------------------------------------------------------

def grid_search_calibration(
    config: CalibrationConfig = None,
    use_worldbank: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Realiza una búsqueda en cuadrícula sobre el espacio de parámetros.

    Parámetros
    ----------
    config : CalibrationConfig
        Espacio de búsqueda. Si es None, usa valores por defecto.
    use_worldbank : bool
        Si True, descarga el target Gini desde el Banco Mundial.
    verbose : bool
        Mostrar barra de progreso.

    Retorna
    -------
    pd.DataFrame con todos los resultados, ordenado por RMSE ascendente.
    """
    if config is None:
        config = CalibrationConfig()

    # Actualizar target con datos reales si es posible
    if use_worldbank:
        try:
            wbstats = world_gini_range()
            config.target_gini = wbstats["median"]
            if verbose:
                print(f"Target Gini (Banco Mundial mediana): {config.target_gini:.3f}")
                print(f"  Rango: [{wbstats['min']:.3f}, {wbstats['max']:.3f}]")
        except Exception as e:
            if verbose:
                print(f"No se pudo conectar al Banco Mundial ({e}). Usando target={config.target_gini:.3f}")

    # Generar todas las combinaciones de parámetros
    param_grid = list(itertools.product(
        config.initial_inequality_values,
        config.tax_policies,
        config.enforce_floor_values,
        config.landscape_peaks_values,
    ))

    if verbose:
        print(f"\nGrid search: {len(param_grid)} combinaciones x {config.replications} réplicas")
        print(f"  = {len(param_grid) * config.replications} simulaciones totales")

    # Construir lista de tareas
    tasks = []
    for (ineq, tax, floor, peaks) in param_grid:
        params = {
            "N": config.N,
            "initial_inequality": ineq,
            "tax_policy": tax,
            "enforce_floor": floor,
            "landscape_peaks": peaks,
        }
        for rep in range(config.replications):
            tasks.append((params, rep * 1000 + hash(str(params)) % 1000, config.steps))

    # Ejecutar en paralelo
    results = []
    try:
        with mp.Pool(processes=config.n_jobs) as pool:
            iterator = pool.imap_unordered(_run_single, tasks, chunksize=4)
            if verbose:
                iterator = tqdm(iterator, total=len(tasks), desc="Calibrando")
            for res in iterator:
                results.append(res)
    except Exception:
        # Fallback secuencial
        if verbose:
            print("Fallback a ejecución secuencial.")
        for task in tasks:
            results.append(_run_single(task))

    # Agregar por combinación de parámetros
    rows = []
    for (ineq, tax, floor, peaks) in param_grid:
        matching = [
            r["gini"] for r in results
            if r["params"]["initial_inequality"] == ineq
            and r["params"]["tax_policy"] == tax
            and r["params"]["enforce_floor"] == floor
            and r["params"]["landscape_peaks"] == peaks
        ]
        if not matching:
            continue
        mean_gini = float(np.mean(matching))
        std_gini = float(np.std(matching))
        rmse = float(abs(mean_gini - config.target_gini))
        within_tolerance = rmse <= config.target_tolerance
        rows.append({
            "initial_inequality": ineq,
            "tax_policy": tax,
            "enforce_floor": floor,
            "landscape_peaks": peaks,
            "mean_gini": mean_gini,
            "std_gini": std_gini,
            "target_gini": config.target_gini,
            "rmse": rmse,
            "within_tolerance": within_tolerance,
        })

    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return df


def best_parameters(calibration_df: pd.DataFrame) -> dict:
    """
    Retorna el diccionario de parámetros mejor calibrado.

    Parámetros
    ----------
    calibration_df : pd.DataFrame
        Resultado de grid_search_calibration().

    Retorna
    -------
    dict con los parámetros de la fila con menor RMSE.
    """
    best = calibration_df.iloc[0]
    return {
        "initial_inequality": best["initial_inequality"],
        "tax_policy": best["tax_policy"],
        "enforce_floor": bool(best["enforce_floor"]),
        "landscape_peaks": int(best["landscape_peaks"]),
    }
