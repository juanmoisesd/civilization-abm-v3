"""
plots.py — Visualizaciones para Civilization-ABM v2/v3.

Genera figuras de calidad de publicación (300 dpi) para Paper 2.
Compatible con CivilModelV3 (usa snake_case column names).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from .metrics import lorenz_curve, gini, top10_share, collapse_summary

# Paleta consistente con v1
PALETTE = {
    "cooperative": "#2196F3",
    "competitive": "#F44336",
    "neutral":     "#9E9E9E",
    "stable":      "#4CAF50",
    "stressed":    "#FF9800",
    "collapsed":   "#F44336",
    "recovering":  "#9C27B0",
}

DPI = 300
FIGSIZE_WIDE = (12, 4)
FIGSIZE_SQUARE = (8, 6)


def _save(fig, path, dpi: int = DPI):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


def _col(df, v3_name, v2_name, default=None):
    """Helper: retorna la columna correcta según versión del modelo."""
    if v3_name in df.columns:
        return df[v3_name]
    if v2_name in df.columns:
        return df[v2_name]
    if default is not None:
        return pd.Series([default] * len(df))
    return None


# -----------------------------------------------------------------------
# Figura 1: Panel base — Gini, riqueza, población viva
# -----------------------------------------------------------------------

def plot_baseline_panel(model_df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Panel de 3 subplots: Gini temporal, riqueza media, agentes vivos."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)

    steps = range(len(model_df))

    gini_col   = _col(model_df, "gini", "Gini", 0)
    wealth_col = _col(model_df, "mean_wealth", "MeanWealth", 0)
    alive_col  = _col(model_df, "alive_count", "AliveAgents", 0)

    axes[0].plot(steps, gini_col, color="#1565C0", linewidth=1.5)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Gini coefficient")
    axes[0].set_title("Wealth Inequality (Gini)")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.38, ls="--", color="gray", alpha=0.6, label="World Bank median")
    axes[0].legend(fontsize=8)

    axes[1].plot(steps, wealth_col, color="#388E3C", linewidth=1.5)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Mean wealth")
    axes[1].set_title("Mean Agent Wealth")

    axes[2].plot(steps, alive_col, color="#6A1B9A", linewidth=1.5)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("N alive agents")
    axes[2].set_title("Population Dynamics")

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 2: Estrategias evolutivas
# -----------------------------------------------------------------------

def plot_strategy_evolution(model_df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Evolución de la distribución estratégica a lo largo del tiempo."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    steps = range(len(model_df))

    strategy_map = [
        ("Cooperative", "coop_share",    "CoopShare"),
        ("Competitive", "comp_share",    "CompShare"),
        ("Neutral",     "neutral_share", "NeutralShare"),
    ]

    for strategy, v3_col, v2_col in strategy_map:
        col = _col(model_df, v3_col, v2_col)
        if col is not None:
            color = PALETTE.get(strategy.lower(), "gray")
            axes[0].plot(steps, col, label=strategy, color=color, linewidth=1.5)

    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Population share")
    axes[0].set_title("Strategy Distribution Over Time")
    axes[0].legend(); axes[0].set_ylim(0, 1)

    entropy_col = _col(model_df, "strategy_entropy", "StrategyEntropy")
    if entropy_col is not None:
        axes[1].plot(steps, entropy_col, color="#795548", linewidth=1.5)
        axes[1].axhline(np.log2(3), ls="--", color="gray", alpha=0.6, label="Max entropy")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Shannon entropy (bits)")
        axes[1].set_title("Strategic Diversity")
        axes[1].legend()

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 3: Colapso institucional
# -----------------------------------------------------------------------

def plot_institutional_collapse(model_df: pd.DataFrame, output_path: str = None) -> plt.Figure:
    """Estabilidad institucional y régimen a lo largo del tiempo."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps = range(len(model_df))

    stab_col   = _col(model_df, "stability", "Stability")
    regime_col = _col(model_df, "regime", "Regime")

    # Panel superior: índice de estabilidad
    if stab_col is not None:
        axes[0].plot(steps, stab_col, color="#1565C0", linewidth=1.5)
        axes[0].axhline(0.65, ls="--", color="#4CAF50", alpha=0.7, label="Stable threshold")
        axes[0].axhline(0.25, ls="--", color="#F44336", alpha=0.7, label="Collapse threshold")
        axes[0].set_ylabel("Stability index")
        axes[0].set_title("Institutional Stability")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8)

    # Panel inferior: régimen como fondo de color
    if regime_col is not None:
        regimes = regime_col.values
        for i, r in enumerate(regimes):
            axes[1].axvspan(i, i + 1, alpha=0.6, color=PALETTE.get(r, "white"))

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Regime")
        axes[1].set_title("Institutional Regime")
        axes[1].set_yticks([])

        # Leyenda manual
        from matplotlib.patches import Patch
        legend_els = [Patch(fc=v, label=k.capitalize()) for k, v in PALETTE.items()
                      if k in ("stable", "stressed", "collapsed", "recovering")]
        axes[1].legend(handles=legend_els, loc="upper right", fontsize=8)

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 4: Curva de Lorenz
# -----------------------------------------------------------------------

def plot_lorenz(
    agent_df: pd.DataFrame,
    steps_to_plot: list = None,
    output_path: str = None,
) -> plt.Figure:
    """
    Curvas de Lorenz en distintos momentos del tiempo.

    agent_df : DataFrame de agentes (agent_reporters) con columna 'Step'.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    all_steps = sorted(agent_df["Step"].unique()) if "Step" in agent_df.columns else []
    if steps_to_plot is None:
        n_steps = len(all_steps)
        steps_to_plot = [
            all_steps[0],
            all_steps[n_steps // 4],
            all_steps[n_steps // 2],
            all_steps[-1],
        ] if n_steps >= 4 else all_steps

    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / max(len(steps_to_plot) - 1, 1))
              for i in range(len(steps_to_plot))]

    for step, color in zip(steps_to_plot, colors):
        subset = agent_df[agent_df["Step"] == step]
        wealths = subset["Wealth"].values
        pop, wealth = lorenz_curve(wealths)
        g = gini(wealths)
        ax.plot(pop, wealth, color=color, linewidth=1.5, label=f"Step {step} (G={g:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect equality")
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative wealth share")
    ax.set_title("Lorenz Curves at Selected Time Steps")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 5: Mapa espacial del paisaje
# -----------------------------------------------------------------------

def plot_landscape_snapshot(model, output_path: str = None) -> plt.Figure:
    """
    Snapshot del paisaje de recursos con posición de los agentes.
    Requiere la instancia del modelo (no el DataFrame).
    Soporta tanto CivilModelV2 (con model.agents) como CivilModelV3 (con arrays).
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    landscape = model.landscape

    # Panel izquierdo: recursos actuales
    im = axes[0].imshow(
        landscape.grid.T, origin="lower", cmap="YlGn",
        vmin=0, vmax=landscape.max_capacity
    )
    plt.colorbar(im, ax=axes[0], label="Resources")
    axes[0].set_title(f"Resource Landscape (Step {model._step_count})")

    # Superponer agentes — compatible con v2 y v3
    strategy_colors = {
        "cooperative": "blue", "competitive": "red", "neutral": "gray",
        0: "blue", 1: "red", 2: "gray",
    }

    if hasattr(model, "agents"):
        # v2: Mesa agents
        alive_agents = [a for a in model.agents if a.alive]
        for agent in alive_agents:
            color = strategy_colors.get(agent.strategy, "black")
            axes[0].scatter(agent.x, agent.y, c=color, s=10, alpha=0.7, zorder=5)
    else:
        # v3: arrays NumPy
        import numpy as np
        alive_mask = model.alive
        xs = model.x[alive_mask]
        ys = model.y[alive_mask]
        strats = model.strategy[alive_mask]
        for s, color in [(0, "blue"), (1, "red"), (2, "gray")]:
            mask = strats == s
            if mask.any():
                axes[0].scatter(xs[mask], ys[mask], c=color, s=10, alpha=0.7, zorder=5)

    # Panel derecho: capacidad máxima del paisaje
    im2 = axes[1].imshow(
        landscape.capacity.T, origin="lower", cmap="YlOrRd",
        vmin=1, vmax=landscape.max_capacity
    )
    plt.colorbar(im2, ax=axes[1], label="Max capacity")
    axes[1].set_title("Landscape Capacity (Fixed)")

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 6: Heatmap de resultados multi-condición
# -----------------------------------------------------------------------

def plot_results_heatmap(
    results_df: pd.DataFrame,
    metric: str = "gini_final",
    row_var: str = "tax_policy",
    col_var: str = "initial_inequality",
    output_path: str = None,
) -> plt.Figure:
    """
    Heatmap de una métrica sobre el espacio de parámetros.

    results_df debe tener columnas: row_var, col_var, metric.
    """
    pivot = results_df.pivot_table(
        index=row_var, columns=col_var, values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn_r",
        linewidths=0.5, cbar_kws={"label": metric.replace("_", " ").title()}
    )
    ax.set_title(f"{metric.replace('_', ' ').title()} by {row_var} x {col_var}")
    ax.set_xlabel(col_var.replace("_", " ").title())
    ax.set_ylabel(row_var.replace("_", " ").title())

    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# -----------------------------------------------------------------------
# Figura 7: Panel de comparación de condiciones
# -----------------------------------------------------------------------

def plot_comparison_panel(
    all_results: pd.DataFrame,
    output_path: str = None,
) -> plt.Figure:
    """
    Boxplots de métricas clave comparando condiciones experimentales.

    all_results : DataFrame con columna 'condition' y métricas por fila.
    """
    metrics = ["gini_final", "stability_final", "strategy_entropy_final", "frac_collapsed"]
    labels = ["Final Gini", "Final Stability", "Strategy Entropy", "Fraction Collapsed"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        if metric not in all_results.columns:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=all_results, x="condition", y=metric, ax=ax,
            palette="Set2", order=sorted(all_results["condition"].unique())
        )
        ax.set_title(label)
        ax.set_xlabel("Condition")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Cross-condition Comparison — Civilization-ABM v3", fontsize=12)
    fig.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig
