"""
rules_v3.py — Reglas institucionales vectorizadas.

Mejoras vs v2:
  - flat_tax, progressive_tax: operaciones NumPy en lugar de loops Python
  - apply_chaos: operación NumPy vectorizada
  - enforce_minimum_wealth: vectorizado
  - InstitutionalSystem: idéntica lógica, gini calculado sobre arrays
"""

import numpy as np


# -----------------------------------------------------------------------
# Reglas fiscales vectorizadas
# -----------------------------------------------------------------------

def flat_tax_vectorized(wealth: np.ndarray, alive: np.ndarray, rate: float = 0.01) -> np.ndarray:
    """
    Impuesto proporcional vectorizado.
    Modifica wealth in-place y retorna el array modificado.
    """
    tax = wealth * rate * alive  # no-op para dead agents (alive=False=0)
    total = tax.sum()
    wealth -= tax
    n_alive = alive.sum()
    if n_alive > 0:
        wealth += (total / n_alive) * alive
    return wealth


def progressive_tax_vectorized(wealth: np.ndarray, alive: np.ndarray, mult: float = 1.0) -> np.ndarray:
    """
    Impuesto progresivo vectorizado por tramos.
    Modifica wealth in-place.
    """
    brackets = [(20, 0.005 * mult), (50, 0.010 * mult), (np.inf, 0.020 * mult)]

    tax = np.zeros_like(wealth)
    for threshold, rate in brackets:
        mask = (wealth <= threshold) & alive & (tax == 0)
        tax[mask] = wealth[mask] * rate

    # Agentes por encima del último umbral
    mask_top = (wealth > 50) & alive & (tax == 0)
    tax[mask_top] = wealth[mask_top] * brackets[-1][1]

    total = tax.sum()
    wealth -= tax
    np.maximum(wealth, 0.0, out=wealth)

    n_alive = alive.sum()
    if n_alive > 0:
        wealth += (total / n_alive) * alive
    return wealth


def reputation_penalty_vectorized(wealth: np.ndarray, reputation: np.ndarray, alive: np.ndarray,
                                   threshold: float = 0.3, penalty: float = 0.1) -> np.ndarray:
    """Sanción económica vectorizada a baja reputación."""
    mask = (reputation < threshold) & alive
    wealth[mask] = np.maximum(0.0, wealth[mask] - penalty)
    return wealth


def enforce_minimum_wealth_vectorized(wealth: np.ndarray, alive: np.ndarray, minimum: float = 1.0) -> np.ndarray:
    """Piso de riqueza mínima vectorizado."""
    deficit_mask = (wealth < minimum) & alive
    if not deficit_mask.any():
        return wealth

    deficit = np.maximum(0.0, minimum - wealth[deficit_mask]).sum()
    rich_mask = (wealth > minimum * 2) & alive

    if rich_mask.sum() == 0:
        return wealth

    contrib_per = deficit / rich_mask.sum()
    wealth[rich_mask] = np.maximum(minimum, wealth[rich_mask] - contrib_per)
    wealth[deficit_mask] = minimum
    return wealth


def apply_chaos_vectorized(wealth: np.ndarray, alive: np.ndarray, chaos_cost: float = 0.5) -> np.ndarray:
    """Pérdida de riqueza durante colapso — vectorizado."""
    wealth[alive] = np.maximum(0.0, wealth[alive] - chaos_cost)
    return wealth


# -----------------------------------------------------------------------
# Sistema institucional (idéntica lógica que v2, trabaja con arrays)
# -----------------------------------------------------------------------

class InstitutionalSystemV3:
    """
    Gestiona la estabilidad institucional — lógica idéntica a v2
    pero opera sobre arrays NumPy en lugar de Mesa AgentSet.
    """

    def __init__(
        self,
        collapse_threshold: float = 0.25,
        stable_threshold: float = 0.65,
        recovery_rate: float = 0.002,
        decay_factor: float = 0.015,
        chaos_cost: float = 0.5,
    ):
        self.collapse_threshold = collapse_threshold
        self.stable_threshold = stable_threshold
        self.recovery_rate = recovery_rate
        self.decay_factor = decay_factor
        self.chaos_cost = chaos_cost

        self.stability = 1.0
        self.regime = "stable"
        self._in_collapse = False

        self.stability_history = []
        self.regime_history = []
        self.collapse_events = []
        self.recovery_events = []

    def _compute_gini(self, wealth: np.ndarray, alive: np.ndarray) -> float:
        w = wealth[alive]
        if len(w) == 0 or w.sum() == 0:
            return 0.0
        w = np.sort(w)
        n = len(w)
        idx = np.arange(1, n + 1)
        return float(((2 * idx - n - 1) * w).sum() / (n * w.sum()))

    def update(self, wealth: np.ndarray, alive: np.ndarray, social_class: np.ndarray, step: int):
        """
        Actualiza estabilidad. Opera sobre arrays NumPy.
        Misma ecuación que v2: S(t+1) = S(t) - delta[G(t) + 0.5*L(t)] + rho_S*1[not collapsed]
        """
        gini = self._compute_gini(wealth, alive)
        n_alive = alive.sum()
        lower_frac = float((social_class[alive] == 0).sum() / n_alive) if n_alive > 0 else 0.0

        deterioration = self.decay_factor * (gini + lower_frac * 0.5)
        self.stability = max(0.0, self.stability - deterioration)

        if self._in_collapse:
            self.stability = min(1.0, self.stability + self.recovery_rate * 0.5)
            if self.stability >= self.collapse_threshold:
                self._in_collapse = False
                self.regime = "recovering"
                self.recovery_events.append(step)
        else:
            self.stability = min(1.0, self.stability + self.recovery_rate * 0.1)

        if self.stability < self.collapse_threshold:
            if not self._in_collapse:
                self._in_collapse = True
                self.collapse_events.append(step)
            self.regime = "collapsed"
        elif self.stability < self.stable_threshold:
            self.regime = "recovering" if self._in_collapse else "stressed"
        else:
            self._in_collapse = False
            self.regime = "stable"

        self.stability_history.append(self.stability)
        self.regime_history.append(self.regime)

    def tax_multiplier(self) -> float:
        return {"stable": 1.0, "stressed": 0.6, "recovering": 0.3, "collapsed": 0.0}.get(self.regime, 1.0)

    def apply_institutions(
        self,
        wealth: np.ndarray,
        alive: np.ndarray,
        reputation: np.ndarray,
        tax_policy: str,
        enforce_floor: bool,
    ) -> np.ndarray:
        """Aplica todas las reglas institucionales sobre los arrays."""
        mult = self.tax_multiplier()

        if mult > 0:
            if tax_policy == "flat":
                flat_tax_vectorized(wealth, alive, rate=0.01 * mult)
            elif tax_policy == "progressive":
                progressive_tax_vectorized(wealth, alive, mult=mult)

        reputation_penalty_vectorized(wealth, reputation, alive)

        if enforce_floor:
            enforce_minimum_wealth_vectorized(wealth, alive)

        if self.regime == "collapsed":
            apply_chaos_vectorized(wealth, alive, self.chaos_cost)

        return wealth
