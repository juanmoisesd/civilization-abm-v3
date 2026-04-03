"""
metrics.py — Métricas de análisis para Civilization-ABM v2/v3.

Amplía las métricas de v1 con:
  - Theil T y L (descomposición intra/inter clases)
  - Palma ratio
  - Movilidad social (correlación intergeneracional de riqueza)
  - Índice de volatilidad estratégica
  - Resumen de colapsos institucionales

Compatible con CivilModelV3: las columnas del DataFrame pueden variar
(v3 usa snake_case en lugar de CamelCase de v2).
"""

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------
# Métricas de distribución de riqueza
# -----------------------------------------------------------------------

def gini(wealths: np.ndarray) -> float:
    """Coeficiente de Gini. Devuelve 0 si todas las riquezas son iguales."""
    w = np.asarray(wealths, dtype=float)
    w = w[w > 0]
    if len(w) == 0 or w.sum() == 0:
        return 0.0
    sorted_w = np.sort(w)
    n = len(sorted_w)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * sorted_w).sum() / (n * sorted_w.sum()))


def theil_t(wealths: np.ndarray) -> float:
    """
    Índice de Theil T (sensible a desigualdad en la parte alta).
    T = (1/N) sum (x_i / mu) ln(x_i / mu)
    """
    w = np.asarray(wealths, dtype=float)
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    mu = w.mean()
    if mu == 0:
        return 0.0
    ratios = w / mu
    return float(np.mean(ratios * np.log(ratios + 1e-12)))


def theil_l(wealths: np.ndarray) -> float:
    """
    Índice de Theil L / MLD (sensible a desigualdad en la parte baja).
    L = (1/N) sum ln(mu / x_i)
    """
    w = np.asarray(wealths, dtype=float)
    w = w[w > 0]
    if len(w) == 0:
        return 0.0
    mu = w.mean()
    if mu == 0:
        return 0.0
    return float(np.mean(np.log(mu / (w + 1e-12))))


def palma_ratio(wealths: np.ndarray) -> float:
    """
    Ratio de Palma: riqueza del decil superior / riqueza de los cuatro deciles inferiores.
    Captura la desigualdad extrema mejor que el Gini.
    """
    w = np.sort(np.asarray(wealths, dtype=float))
    n = len(w)
    if n == 0:
        return 0.0
    top_10_pct = w[int(n * 0.9):]
    bot_40_pct = w[:int(n * 0.4)]
    if bot_40_pct.sum() == 0:
        return float("inf")
    return float(top_10_pct.sum() / bot_40_pct.sum())


def lorenz_curve(wealths: np.ndarray):
    """
    Curva de Lorenz: (fracción acumulada de población, fracción acumulada de riqueza).
    """
    w = np.sort(np.asarray(wealths, dtype=float))
    n = len(w)
    cum_pop = np.linspace(0, 1, n + 1)
    cum_wealth = np.concatenate([[0], np.cumsum(w) / (w.sum() + 1e-12)])
    return cum_pop, cum_wealth


def top1_share(wealths: np.ndarray) -> float:
    """Fracción de la riqueza total en manos del 1% más rico."""
    w = np.sort(np.asarray(wealths, dtype=float))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    top = w[max(0, int(n * 0.99)):]
    return float(top.sum() / w.sum())


def top10_share(wealths: np.ndarray) -> float:
    """Fracción de la riqueza total en manos del 10% más rico."""
    w = np.sort(np.asarray(wealths, dtype=float))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    top = w[max(0, int(n * 0.90)):]
    return float(top.sum() / w.sum())


# -----------------------------------------------------------------------
# Métricas estratégicas y evolutivas
# -----------------------------------------------------------------------

def strategy_entropy(shares: dict) -> float:
    """Entropía de Shannon de la distribución estratégica."""
    h = 0.0
    for p in shares.values():
        if p > 0:
            h -= p * np.log2(p + 1e-12)
    return float(h)


def strategic_volatility(strategy_series: pd.Series) -> float:
    """
    Fracción de pasos en los que la estrategia dominante cambia.
    Mide la inestabilidad evolutiva del sistema.
    """
    if len(strategy_series) < 2:
        return 0.0
    changes = (strategy_series != strategy_series.shift(1)).sum()
    return float(changes / (len(strategy_series) - 1))


# -----------------------------------------------------------------------
# Métricas de colapso institucional
# -----------------------------------------------------------------------

def collapse_summary(regime_series: pd.Series) -> dict:
    """
    Analiza la serie de regímenes y extrae estadísticas de colapsos.

    Parámetros
    ----------
    regime_series : pd.Series
        Serie de strings 'stable'|'stressed'|'collapsed'|'recovering'

    Retorna
    -------
    dict con:
      n_collapses     — número de episodios de colapso
      total_collapse_steps — pasos totales en estado collapsed
      mean_collapse_duration — duración media de colapsos
      frac_collapsed  — fracción del tiempo en colapso
    """
    regimes = regime_series.values
    n = len(regimes)
    if n == 0:
        return {"n_collapses": 0, "total_collapse_steps": 0,
                "mean_collapse_duration": 0.0, "frac_collapsed": 0.0}

    collapsed = (regimes == "collapsed").astype(int)
    total_steps = int(collapsed.sum())

    # Contar episodios: transiciones 0->1
    n_collapses = int(np.sum(np.diff(np.concatenate([[0], collapsed, [0]])) == 1))

    mean_dur = total_steps / n_collapses if n_collapses > 0 else 0.0

    return {
        "n_collapses":           n_collapses,
        "total_collapse_steps":  total_steps,
        "mean_collapse_duration": float(mean_dur),
        "frac_collapsed":        float(total_steps / n),
    }


# -----------------------------------------------------------------------
# Resumen estadístico completo de una simulación
# Soporta tanto v2 (CamelCase) como v3 (snake_case)
# -----------------------------------------------------------------------

def summary_statistics(model_df: pd.DataFrame) -> dict:
    """
    Genera estadísticas resumidas de una corrida a partir del DataFrame
    del modelo (v2 DataCollector o v3 get_model_vars_dataframe()).

    Parámetros
    ----------
    model_df : pd.DataFrame
        Columnas pueden ser CamelCase (v2) o snake_case (v3).

    Retorna
    -------
    dict con métricas finales y de trayectoria.
    """
    if model_df.empty:
        return {}

    # Mapeo de nombres: intenta columnas v3 primero, luego v2 (puede ser None)
    def _get(df, v3_col, v2_col, default=0):
        if v3_col and v3_col in df.columns:
            return df[v3_col]
        if v2_col and v2_col in df.columns:
            return df[v2_col]
        return pd.Series([default] * len(df))

    gini_series      = _get(model_df, "gini",              "Gini",              0)
    mean_w_series    = _get(model_df, "mean_wealth",        "MeanWealth",        0)
    total_w_series   = _get(model_df, "total_wealth",       "TotalWealth",       0)
    alive_series     = _get(model_df, "alive_count",        "AliveAgents",       0)
    upper_series     = _get(model_df, "upper_frac",         "UpperClass",        0)
    lower_series     = _get(model_df, "lower_frac",         "LowerClass",        0)
    stab_series      = _get(model_df, "stability",          "Stability",         1)
    regime_series    = _get(model_df, "regime",             "Regime",            "stable")
    coop_series      = _get(model_df, "coop_share",         "CoopShare",         0)
    comp_series      = _get(model_df, "comp_share",         "CompShare",         0)
    neutral_series   = _get(model_df, "neutral_share",      "NeutralShare",      0)
    dom_strat_series = _get(model_df, "dominant_strategy",  "DominantStrategy",  "neutral")
    entropy_series   = _get(model_df, "strategy_entropy",   "StrategyEntropy",   0)

    # ── v3 extended metrics (new in CivilModelV3 / StepRecord) ──────────
    theil_series    = _get(model_df, "theil_t",          None,  np.nan)
    palma_series    = _get(model_df, "palma_ratio",      None,  np.nan)
    top1_series     = _get(model_df, "top1_share",       None,  np.nan)
    top10_series    = _get(model_df, "top10_share",      None,  np.nan)
    mob_series      = _get(model_df, "social_mobility",  None,  np.nan)
    mem_series      = _get(model_df, "mean_memory",      None,  np.nan)
    rep_series      = _get(model_df, "mean_reputation",  None,  np.nan)
    lscape_g_series = _get(model_df, "landscape_gini",   None,  np.nan)
    res_series      = _get(model_df, "total_resources",  None,  np.nan)

    final_idx = -1
    result = {
        # ── Métricas finales (Gini + riqueza + estructura) ───────────────
        "gini_final":              float(gini_series.iloc[final_idx]),
        "mean_wealth_final":       float(mean_w_series.iloc[final_idx]),
        "total_wealth_final":      float(total_w_series.iloc[final_idx]),
        "alive_final":             int(alive_series.iloc[final_idx]),
        "upper_frac_final":        float(upper_series.iloc[final_idx]),
        "lower_frac_final":        float(lower_series.iloc[final_idx]),
        "stability_final":         float(stab_series.iloc[final_idx]),
        "regime_final":            str(regime_series.iloc[final_idx]),
        "coop_share_final":        float(coop_series.iloc[final_idx]),
        "comp_share_final":        float(comp_series.iloc[final_idx]),
        "neutral_share_final":     float(neutral_series.iloc[final_idx]),
        "dominant_strategy_final": str(dom_strat_series.iloc[final_idx]),
        "strategy_entropy_final":  float(entropy_series.iloc[final_idx]),

        # ── Métricas de trayectoria ──────────────────────────────────────
        "gini_mean":  float(gini_series.mean()),
        "gini_std":   float(gini_series.std()),
        "gini_max":   float(gini_series.max()),
        "gini_min":   float(gini_series.min()),
        "gini_trend": float(stats.linregress(np.arange(len(gini_series)), gini_series).slope),

        # ── Métricas v3 extendidas: desigualdad multidimensional ─────────
        "theil_t_final":          _safe_float(theil_series,   final_idx),
        "palma_ratio_final":      _safe_float(palma_series,   final_idx),
        "top1_share_final":       _safe_float(top1_series,    final_idx),
        "top10_share_final":      _safe_float(top10_series,   final_idx),
        "theil_t_mean":           _safe_mean(theil_series),
        "palma_ratio_mean":       _safe_mean(palma_series),
        "top1_share_mean":        _safe_mean(top1_series),
        "top10_share_mean":       _safe_mean(top10_series),

        # ── Métricas v3 extendidas: memoria y reputación ─────────────────
        "social_mobility_final":  _safe_float(mob_series,  final_idx),
        "social_mobility_mean":   _safe_mean(mob_series),
        "mean_memory_final":      _safe_float(mem_series,  final_idx),
        "mean_memory_mean":       _safe_mean(mem_series),
        "mean_reputation_final":  _safe_float(rep_series,  final_idx),
        "mean_reputation_mean":   _safe_mean(rep_series),

        # ── Métricas v3 extendidas: paisaje ──────────────────────────────
        "landscape_gini_final":   _safe_float(lscape_g_series, final_idx),
        "total_resources_final":  _safe_float(res_series,      final_idx),
    }

    # Colapsos institucionales
    result.update(collapse_summary(regime_series))

    # Volatilidad estratégica
    result["strategic_volatility"] = strategic_volatility(dom_strat_series)

    return result


# -----------------------------------------------------------------------
# Helpers internos
# -----------------------------------------------------------------------

def _safe_float(series: pd.Series, idx: int) -> float:
    """Extrae float del índice dado; devuelve NaN si la columna no existe o es NaN."""
    try:
        v = series.iloc[idx]
        return float(v) if not pd.isna(v) else float("nan")
    except Exception:
        return float("nan")


def _safe_mean(series: pd.Series) -> float:
    """Media de la serie, ignorando NaN; devuelve NaN si está vacía."""
    try:
        v = series.dropna()
        return float(v.mean()) if len(v) > 0 else float("nan")
    except Exception:
        return float("nan")
