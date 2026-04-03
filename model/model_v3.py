"""
model_v3.py — CivilModelV3: modelo vectorizado con memoria de agentes.

Mejoras vs v2:
  - Arrays NumPy para todo el estado (sin _agent_map O(N²))
  - Memoria de agentes: buffer circular de 5 interacciones (inspirado en MiroFish)
  - Replicador ponderado por reputación (reciprocidad indirecta)
  - Métricas enriquecidas: Theil T, Palma, top-1%, movilidad social

Referencia memoria:
  Nowak & Sigmund (1998) Evolution of indirect reciprocity by image scoring.
  Santos et al. (2008) Social diversity promotes the emergence of cooperation.
  MiroFish (2025) Swarm intelligence engine with agent long-term memory.
"""

import random as _random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx

from .landscape_v3 import ResourceLandscapeV3
from .rules_v3 import InstitutionalSystemV3

# Estrategias: 0=cooperative, 1=competitive, 2=neutral
STRAT_COOP = 0
STRAT_COMP = 1
STRAT_NEUT = 2
STRAT_NAMES = {0: "cooperative", 1: "competitive", 2: "neutral"}
STRAT_IDX   = {"cooperative": 0, "competitive": 1, "neutral": 2}

MEMORY_SIZE = 5   # últimas N interacciones recordadas por agente


# ------------------------------------------------------------------
# Funciones de métricas de desigualdad
# ------------------------------------------------------------------

def _gini_array(wealth: np.ndarray, alive: np.ndarray) -> float:
    w = wealth[alive]
    if len(w) == 0 or w.sum() == 0:
        return 0.0
    w = np.sort(w)
    n = len(w)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * w).sum() / (n * w.sum()))


def _theil_t(w: np.ndarray) -> float:
    """Índice de Theil T — sensible a desigualdad en la parte alta."""
    if len(w) == 0:
        return 0.0
    mu = w.mean()
    if mu == 0:
        return 0.0
    ratios = w / mu
    return float(np.mean(ratios * np.log(ratios + 1e-12)))


def _palma_ratio(w: np.ndarray) -> float:
    """Ratio de Palma: riqueza del 10% superior / 40% inferior."""
    if len(w) == 0:
        return 0.0
    w_s = np.sort(w)
    n = len(w_s)
    top10 = w_s[int(n * 0.9):].sum()
    bot40 = w_s[:max(1, int(n * 0.4))].sum()
    return float(top10 / (bot40 + 1e-9))


def _top1_share(w: np.ndarray) -> float:
    """Fracción de la riqueza en manos del 1% más rico."""
    if len(w) == 0 or w.sum() == 0:
        return 0.0
    w_s = np.sort(w)
    n = len(w_s)
    top = w_s[max(0, int(n * 0.99)):]
    return float(top.sum() / w_s.sum())


def _top10_share(w: np.ndarray) -> float:
    """Fracción de la riqueza en manos del 10% más rico."""
    if len(w) == 0 or w.sum() == 0:
        return 0.0
    w_s = np.sort(w)
    n = len(w_s)
    top = w_s[max(0, int(n * 0.90)):]
    return float(top.sum() / w_s.sum())


# ------------------------------------------------------------------
# Red social
# ------------------------------------------------------------------

def _build_network(n: int, network_type: str, rng_seed: int = None) -> Optional[nx.Graph]:
    if network_type == "small_world":
        G = nx.watts_strogatz_graph(n, k=4, p=0.1, seed=rng_seed)
    elif network_type == "scale_free":
        G = nx.barabasi_albert_graph(n, m=2, seed=rng_seed)
    elif network_type == "random":
        G = nx.erdos_renyi_graph(n, p=0.05, seed=rng_seed)
    else:
        return None
    return G


def _adj_list_from_graph(G: nx.Graph, n: int) -> list:
    return [np.array(list(G.neighbors(i)), dtype=np.int32) for i in range(n)]


# ------------------------------------------------------------------
# Registro de métricas
# ------------------------------------------------------------------

@dataclass
class StepRecord:
    """Métricas completas de un paso — enriquecidas vs v2."""
    step: int
    # Desigualdad (suite completa)
    gini: float
    theil_t: float
    palma_ratio: float
    top1_share: float
    top10_share: float
    # Riqueza
    mean_wealth: float
    total_wealth: float
    alive_count: int
    # Clases sociales
    upper_frac: float
    lower_frac: float
    social_mobility: float   # NUEVO: fracción que cambió de clase
    # Instituciones
    stability: float
    regime: str
    # Estrategias
    strategy_entropy: float
    dominant_strategy: str
    coop_share: float
    comp_share: float
    neutral_share: float
    # Memoria (NUEVO — inspirado en MiroFish)
    mean_memory: float       # score medio de memoria [-1, 1]
    mean_reputation: float   # reputación media de la población
    # Paisaje
    landscape_gini: float
    total_resources: float


# ------------------------------------------------------------------
# Modelo principal
# ------------------------------------------------------------------

class CivilModelV3:
    """
    Modelo de civilización artificial v3 — vectorizado + memoria de agentes.

    Parámetros nuevos respecto a v2
    --------------------------------
    use_memory : bool
        Activa la memoria de interacciones (buffer circular de 5 pasos).
        Cuando True, el replicador pondera por reputación del modelo
        (reciprocidad indirecta — Nowak & Sigmund, 1998).
    """

    def __init__(
        self,
        N: int = 500,
        initial_inequality: float = 0.8,
        tax_policy: Optional[str] = "progressive",
        network_type: Optional[str] = "small_world",
        enforce_floor: bool = False,
        landscape_width: int = 35,
        landscape_height: int = 35,
        landscape_peaks: int = 3,
        growth_rate: float = 0.5,
        metabolism: float = 1.0,
        mutation_rate: float = 0.02,
        evolution_interval: int = 5,
        use_memory: bool = False,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.tax_policy = tax_policy
        self.enforce_floor = enforce_floor
        self.mutation_rate = mutation_rate
        self.evolution_interval = evolution_interval
        self.use_memory = use_memory
        self._step_count = 0
        self._seed = seed

        rng = np.random.default_rng(seed)
        if seed is not None:
            _random.seed(seed)

        # ── Paisaje ───────────────────────────────────────────────────
        self.landscape = ResourceLandscapeV3(
            width=landscape_width,
            height=landscape_height,
            max_capacity=20,
            growth_rate=growth_rate,
            n_peaks=landscape_peaks,
            seed=seed,
        )
        W, H = landscape_width, landscape_height

        # ── Arrays de estado de agentes ───────────────────────────────
        self.wealth = np.maximum(1.0, rng.lognormal(2.3, initial_inequality, size=N))
        self.metabolism = np.maximum(0.1, rng.lognormal(0.0, 0.6, size=N) * metabolism)
        self.vision = rng.choice([1, 2, 3], size=N, p=[0.50, 0.35, 0.15]).astype(np.int32)
        self.strategy = rng.integers(0, 3, size=N).astype(np.int8)
        self.reputation = np.ones(N, dtype=np.float32)
        self.social_class = np.ones(N, dtype=np.int8)
        self._prev_social_class = np.ones(N, dtype=np.int8)  # para movilidad
        self.alive = np.ones(N, dtype=bool)

        # ── Memoria de agentes (MiroFish-inspired) ────────────────────
        # Buffer circular: outcomes ∈ {-1, 0, +1}
        # -1 = interacción negativa (perdió riqueza)
        # +1 = interacción positiva (ganó o fue cooperador)
        #  0 = neutral
        self.memory = np.zeros((N, MEMORY_SIZE), dtype=np.float32)
        self.memory_ptr = np.zeros(N, dtype=np.int32)

        # ── Posiciones en el paisaje ──────────────────────────────────
        all_cells = [(x, y) for x in range(W) for y in range(H)]
        perm = rng.permutation(len(all_cells))[:N]
        positions = [all_cells[i] for i in perm]

        self.x = np.array([p[0] for p in positions], dtype=np.int32)
        self.y = np.array([p[1] for p in positions], dtype=np.int32)

        for i in range(N):
            self.landscape.place_agent(i, self.x[i], self.y[i])

        # ── Red social ────────────────────────────────────────────────
        self._graph = _build_network(N, network_type, rng_seed=seed)
        if self._graph is not None:
            self._adj = _adj_list_from_graph(self._graph, N)
        else:
            self._adj = None

        # ── Sistema institucional ─────────────────────────────────────
        self.institutions = InstitutionalSystemV3()

        # ── Historial ─────────────────────────────────────────────────
        self.history = []

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def mean_wealth(self) -> float:
        w = self.wealth[self.alive]
        return float(w.mean()) if len(w) > 0 else 0.0

    @property
    def alive_count(self) -> int:
        return int(self.alive.sum())

    # ------------------------------------------------------------------
    # Paso del modelo
    # ------------------------------------------------------------------

    def step(self):
        # 1. Métricas
        self._collect()

        # 2. Paisaje
        self.landscape.grow()

        # 3. Instituciones
        self.institutions.update(self.wealth, self.alive, self.social_class, self._step_count)

        # 4. Movimiento + cosecha
        self._move_and_harvest()

        # 5. Interacción social (con memoria si use_memory=True)
        self._social_interaction()

        # 6. Reglas institucionales
        self.institutions.apply_institutions(
            self.wealth, self.alive, self.reputation,
            self.tax_policy, self.enforce_floor,
        )

        # 7. Limpiar muertos
        self._cleanup_dead()

        # 8. Evolución estratégica
        if self._step_count % self.evolution_interval == 0:
            self._replicator_dynamics()

        # 9. Rewiring de red
        if self._graph is not None and self._step_count % 10 == 0:
            self._rewire_network()

        # 10. Clases sociales
        self._update_social_classes()

        self._step_count += 1

    # ------------------------------------------------------------------
    # Movimiento y cosecha
    # ------------------------------------------------------------------

    def _move_and_harvest(self):
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return

        perm = np.random.permutation(len(alive_idx))
        alive_idx = alive_idx[perm]

        xs = self.x[alive_idx]
        ys = self.y[alive_idx]
        visions = self.vision[alive_idx]

        new_xs, new_ys = self.landscape.best_neighbors_batch(xs, ys, visions)

        for k, i in enumerate(alive_idx):
            nx_, ny_ = int(new_xs[k]), int(new_ys[k])
            ox, oy = int(self.x[i]), int(self.y[i])
            if (nx_, ny_) != (ox, oy):
                self.landscape.move_agent(i, ox, oy, nx_, ny_)
                self.x[i], self.y[i] = nx_, ny_
            harvested = self.landscape.harvest(self.x[i], self.y[i])
            self.wealth[i] += harvested
            self.wealth[i] -= self.metabolism[i]
            if self.wealth[i] <= 0:
                self.wealth[i] = 0.0
                self.alive[i] = False

    # ------------------------------------------------------------------
    # Interacción social
    # ------------------------------------------------------------------

    def _social_interaction(self):
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) < 2:
            return

        for i in alive_idx:
            if self._adj is not None:
                nbr_ids = self._adj[i]
                alive_nbrs = nbr_ids[self.alive[nbr_ids]] if len(nbr_ids) > 0 else np.array([], dtype=np.int32)
            else:
                candidates = alive_idx[alive_idx != i]
                alive_nbrs = candidates

            if len(alive_nbrs) == 0:
                continue

            j = int(alive_nbrs[np.random.randint(len(alive_nbrs))])
            self._interact_pair(i, j)

    def _interact_pair(self, i: int, j: int):
        """
        Interacción pairwise con registro de memoria.

        La memoria registra el outcome económico de la interacción:
          +1 si el agente ganó o ayudó (cooperativo)
          -1 si el agente fue explotado
           0 si fue neutral
        """
        transfer = min(1.0, self.wealth[i] * 0.05)
        outcome_i = 0.0

        if self.strategy[i] == STRAT_COOP:
            if self.wealth[i] > self.wealth[j]:
                amount = min(transfer, self.wealth[i])
                self.wealth[i] -= amount
                self.wealth[j] += amount
                self.reputation[i] = min(2.0, self.reputation[i] + 0.02)
                outcome_i = +1.0  # cooperó exitosamente

        elif self.strategy[i] == STRAT_COMP:
            if self.wealth[i] < self.wealth[j]:
                extracted = min(transfer, self.wealth[j])
                self.wealth[j] -= extracted
                self.wealth[i] += extracted
                self.reputation[j] = max(0.0, self.reputation[j] - 0.05)
                outcome_i = +1.0   # extrajo riqueza
                # El agente j fue explotado — memoria negativa para j
                if self.use_memory:
                    slot_j = int(self.memory_ptr[j]) % MEMORY_SIZE
                    self.memory[j, slot_j] = -1.0
                    self.memory_ptr[j] += 1
            # Si no pudo extraer, outcome neutro

        else:  # neutral
            if np.random.random() < 0.5 and self.wealth[i] > self.wealth[j]:
                amount = min(transfer * 0.5, self.wealth[i])
                self.wealth[i] -= amount
                self.wealth[j] += amount
                outcome_i = 0.5

        # Registrar memoria del agente i
        if self.use_memory:
            slot_i = int(self.memory_ptr[i]) % MEMORY_SIZE
            self.memory[i, slot_i] = outcome_i
            self.memory_ptr[i] += 1

            # Actualizar reputación desde memoria (score medio de últimas interacciones)
            mem_score = float(self.memory[i].mean())  # ∈ [-1, 1]
            # Reputación ajustada: la memoria influye gradualmente
            self.reputation[i] = float(np.clip(
                self.reputation[i] + 0.01 * mem_score, 0.0, 2.0
            ))

    # ------------------------------------------------------------------
    # Dinámica de replicador (con ponderación por reputación si use_memory)
    # ------------------------------------------------------------------

    def _replicator_dynamics(self):
        """
        Replicador estándar (use_memory=False):
          p_copy = (w_b* - w_a) / (w_max + w_a + ε)

        Replicador ponderado por reputación (use_memory=True):
          p_copy_adj = p_copy × rep_factor(best_nbr)
          donde rep_factor = 0.5 + 0.5 × reputation / 2.0 ∈ [0.25, 1.0]

        Justificación: en sistemas con memoria, los agentes imitan
        preferentemente a quienes tienen alta reputación — no solo a
        los más ricos. Esto es reciprocidad indirecta (Nowak & Sigmund 1998).
        """
        alive_idx = np.where(self.alive)[0]
        mutation_mask = np.random.random(len(alive_idx)) < self.mutation_rate

        for k, i in enumerate(alive_idx):
            if mutation_mask[k]:
                self.strategy[i] = np.random.randint(3)
                continue

            if self._adj is None:
                continue

            nbr_ids = self._adj[i]
            if len(nbr_ids) == 0:
                continue
            alive_nbrs = nbr_ids[self.alive[nbr_ids]]
            if len(alive_nbrs) == 0:
                continue

            nbr_wealth = self.wealth[alive_nbrs]
            best_k = int(np.argmax(nbr_wealth))
            best_nbr = alive_nbrs[best_k]

            if self.wealth[best_nbr] <= self.wealth[i]:
                continue

            max_w = float(nbr_wealth.max())
            p_copy = (self.wealth[best_nbr] - self.wealth[i]) / (max_w + self.wealth[i] + 1e-9)
            p_copy = min(float(p_copy), 1.0)

            # Ponderación por reputación (solo si use_memory=True)
            if self.use_memory:
                rep = float(self.reputation[best_nbr])
                rep_factor = 0.5 + 0.5 * (rep / 2.0)  # ∈ [0.25, 1.0]
                p_copy = min(p_copy * rep_factor, 1.0)

            if np.random.random() < p_copy:
                self.strategy[i] = self.strategy[best_nbr]

    # ------------------------------------------------------------------
    # Network rewiring (reputation-based, igual que antes)
    # ------------------------------------------------------------------

    def _rewire_network(self):
        if self._graph is None:
            return
        alive_idx = list(np.where(self.alive)[0])
        if len(alive_idx) < 4:
            return

        for i in alive_idx:
            if np.random.random() < 0.01:
                nbrs = list(self._graph.neighbors(i))
                if not nbrs:
                    continue
                worst = min(nbrs, key=lambda n: self.reputation[n])
                if self._graph.has_edge(i, worst):
                    self._graph.remove_edge(i, worst)
                candidates = [a for a in alive_idx if a != i and not self._graph.has_edge(i, a)]
                if candidates:
                    new_nbr = _random.choice(candidates)
                    self._graph.add_edge(i, new_nbr)

        for i in list(np.where(self.alive)[0]):
            self._adj[i] = np.array(list(self._graph.neighbors(i)), dtype=np.int32)

    # ------------------------------------------------------------------
    # Limpieza y actualización
    # ------------------------------------------------------------------

    def _cleanup_dead(self):
        dead = np.where(~self.alive)[0]
        for i in dead:
            self.landscape.remove_agent(i, int(self.x[i]), int(self.y[i]))

    def _update_social_classes(self):
        self._prev_social_class = self.social_class.copy()
        mw = self.mean_wealth
        if mw == 0:
            return
        self.social_class[self.alive & (self.wealth < mw * 0.5)] = 0
        self.social_class[self.alive & (self.wealth >= mw * 0.5) & (self.wealth < mw * 1.5)] = 1
        self.social_class[self.alive & (self.wealth >= mw * 1.5)] = 2

    # ------------------------------------------------------------------
    # Recolección de métricas enriquecidas
    # ------------------------------------------------------------------

    def _collect(self):
        alive = self.alive
        n_alive = int(alive.sum())
        if n_alive == 0:
            return

        w_alive = self.wealth[alive]
        gini    = _gini_array(self.wealth, alive)
        theil   = _theil_t(w_alive)
        palma   = _palma_ratio(w_alive)
        top1    = _top1_share(w_alive)
        top10   = _top10_share(w_alive)

        strat_counts = np.bincount(self.strategy[alive].astype(np.int64), minlength=3)
        strat_fracs  = strat_counts / n_alive
        h = 0.0
        for p in strat_fracs:
            if p > 0:
                h -= p * np.log2(p)
        dom_strat = STRAT_NAMES[int(np.argmax(strat_counts))]

        sc = self.social_class[alive]
        lower_f = float((sc == 0).sum() / n_alive)
        upper_f = float((sc == 2).sum() / n_alive)

        # Movilidad social: fracción de agentes vivos que cambiaron de clase
        changed = (self.social_class[alive] != self._prev_social_class[alive]).sum()
        mobility = float(changed / n_alive)

        # Memoria: score medio ∈ [-1, 1] (0 si memoria desactivada)
        if self.use_memory:
            mean_mem = float(self.memory[alive].mean())
        else:
            mean_mem = 0.0

        mean_rep = float(self.reputation[alive].mean())

        self.history.append(StepRecord(
            step=self._step_count,
            gini=gini,
            theil_t=theil,
            palma_ratio=palma,
            top1_share=top1,
            top10_share=top10,
            mean_wealth=float(w_alive.mean()),
            total_wealth=float(w_alive.sum()),
            alive_count=n_alive,
            upper_frac=upper_f,
            lower_frac=lower_f,
            social_mobility=mobility,
            stability=self.institutions.stability,
            regime=self.institutions.regime,
            strategy_entropy=h,
            dominant_strategy=dom_strat,
            coop_share=float(strat_fracs[0]),
            comp_share=float(strat_fracs[1]),
            neutral_share=float(strat_fracs[2]),
            mean_memory=mean_mem,
            mean_reputation=mean_rep,
            landscape_gini=self.landscape.gini_landscape(),
            total_resources=self.landscape.total_resources(),
        ))

    def get_model_vars_dataframe(self):
        import pandas as pd
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([r.__dict__ for r in self.history]).set_index("step")
