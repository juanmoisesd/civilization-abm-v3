# Civilization-ABM v3

**Population Scale, Interaction Memory, and Multidimensional Inequality in an Agent-Based Civilization Model**

> Juan Moisés de la Serna Tuya — Universidad Internacional de La Rioja (UNIR)
> *Journal of Artificial Societies and Social Simulation* (submitted 2026)
> ORCID: [0000-0002-8401-8018](https://orcid.org/0000-0002-8401-8018)

---

## Overview

Civilization-ABM v3 is the third generation of the Civilization-ABM series, extending v2 with:

1. **Population scaling** — N = 1,000 agents on a 50×50 landscape (vs. N = 500 on 35×35 in v2)
2. **Interaction memory** — MiroFish-inspired circular buffer of 5 interaction outcomes per agent, with reputation-weighted replicator dynamics (indirect reciprocity; Nowak & Sigmund 1998)
3. **Multidimensional inequality measurement** — Theil T, Palma ratio, top-1% share, top-10% share, social mobility rate

Across **48 conditions × 30 replications = 1,440 simulation runs** (1,500 steps each), the model finds:

- Equilibrium Gini falls 44.9% at doubled population scale (0.361 → 0.199)
- Fiscal policy is nullified by endemic institutional collapse (100% collapse in 33/48 conditions)
- Floor mechanisms produce Gini reductions of 53–95% vs. baseline
- Interaction memory has negligible marginal effect on inequality (|ΔGini| ≤ 0.007) but reinforces cooperation under floor+memory conditions
- World Bank calibration reproduces correct cross-national Gini ordering

---

## Repository Structure

```
civilization-abm-v3/
├── model/
│   ├── model_v3.py          # Core ABM (CivilModelV3): vectorized NumPy, agent memory
│   ├── landscape_v3.py      # Sugarscape landscape with Numba JIT
│   └── rules_v3.py          # Tax rules and institutional system
├── experiments/
│   ├── configs_v3.yaml      # All 48 experimental conditions
│   └── ray_runner.py        # Distributed runner (Ray + multiprocessing fallback)
├── analysis/
│   ├── metrics.py           # summary_statistics(): Gini, Theil T, Palma, mobility…
│   └── plots.py             # Visualization utilities
├── calibration/
│   ├── worldbank.py         # World Bank Gini archetypes (5 economies)
│   └── fitting.py           # Parameter fitting utilities
├── results/
│   └── v3/
│       └── all_conditions_v3.csv    # Full results (1,440 rows)
├── paper/
│   └── manuscript_v3.tex    # LaTeX manuscript (JASSS submission)
├── main_v3.py               # Entry point
└── requirements_v3.txt      # Python dependencies
```

---

## Reproducibility

### Requirements

```bash
pip install -r requirements_v3.txt
```

Key dependencies: `numpy>=1.26`, `scipy>=1.12`, `networkx>=3.2`, `numba>=0.59`, `ray[default]>=2.10`, `pandas>=2.1`, `pyyaml>=6.0`

### Run all experiments

```bash
# Full suite — ~3 hours on Dell Vostro 5301 (i5, 8 GB RAM, 2 workers)
python -m experiments.ray_runner

# Single condition (fast test)
python -m experiments.ray_runner --condition baseline

# Force multiprocessing (no Ray required)
python -m experiments.ray_runner --no-ray --jobs 2
```

Results are saved to `results/v3/all_conditions_v3.csv`.

### Replicate baseline only

```bash
python main_v3.py --condition baseline --reps 5
```

### Hardware used

| Parameter | Value |
|---|---|
| Machine | Dell Vostro 5301 |
| CPU | Intel Core i5 (4 cores) |
| RAM | 8 GB |
| Python | 3.12 |
| Workers | 2 (n_jobs=2) |
| Total runtime | ~2h 54min (1,440 tasks) |

---

## Key Parameters (global defaults)

| Parameter | Value | Description |
|---|---|---|
| N | 1,000 | Agent population |
| steps | 1,500 | Simulation steps |
| replications | 30 | Independent runs per condition |
| landscape | 50×50 | Grid dimensions |
| growth_rate | 0.5 | Resource regrowth rate |
| metabolism | 1.0 | Base metabolic cost |
| mutation_rate | 0.02 | Random strategy mutation probability |
| evolution_interval | 5 | Steps between replicator dynamic events |
| MEMORY_SIZE | 5 | Interaction outcomes remembered per agent |

---

## Experimental Blocks

| Block | N conditions | Key manipulation |
|---|---|---|
| A — Fiscal policy | 5 | None, flat, progressive, prog+floor, floor only |
| B — Initial inequality | 6 | σ₀ ∈ {0.3, 0.5, 0.8, 1.2, 1.8, 2.5} |
| C — Network topology | 4 | Small-world, scale-free, random, none |
| D — Landscape | 4 | K ∈ {1, 2, 4, 6} Gaussian peaks |
| E — Metabolism | 3 | m ∈ {0.5, 1.0, 2.0} |
| F — Interactions | 8 | High ineq × scale-free; max stress; resilience |
| G — Sensitivity OAT | 7 | Growth rate, mutation, evolution interval, N |
| H — WB calibration | 5 | Nordic, European, USA, LatAm, South Africa |
| I — Memory (new) | 6 | Memory on/off × fiscal × network × inequality |
| **Total** | **48** | **1,440 simulation runs** |

---

## Key Results

| Condition | Gini (mean ± SD) | Regime |
|---|---|---|
| baseline | 0.199 ± 0.051 | collapsed 100% |
| progressive_tax | 0.205 ± 0.063 | collapsed 100% |
| flat_tax | 0.225 ± 0.063 | collapsed 100% |
| prog_tax_floor | 0.063 ± 0.183 | stressed (dominant) |
| welfare_state | 0.024 ± 0.052 | stable 53% |
| max_resilience | 0.000 ± 0.000 | stable 87% |
| hi_ineq_scale_free | 0.617 ± 0.029 | collapsed 100% |
| wb_south_africa | 0.638 ± 0.116 | collapsed 100% |
| wb_nordic | 0.001 ± 0.003 | stable 77% |
| memory_welfare | 0.001 ± 0.003 | stable ~63% |

---

## Companion Papers

- **v1**: de la Serna Tuya, J.M. (2024). Fiscal redistribution, network topology, and wealth inequality in an agent-based civilization model. *JASSS*, 27(4). doi:10.18564/jasss.XXXX
- **v2**: de la Serna Tuya, J.M. (2025). Evolutionary strategies, spatial resources, and institutional collapse in an agent-based civilization model. *JASSS*, 28(1). doi:10.18564/jasss.YYYY
- **v4** *(in development)*: Differentiable PyTorch architecture with gradient-based World Bank calibration (AgentTorch framework)

---

## Citation

```bibtex
@article{delaserna2026v3,
  author  = {de la Serna Tuya, Juan Moisés},
  title   = {Population Scale, Interaction Memory, and Multidimensional
             Inequality in an Agent-Based Civilization Model:
             A Replication and Extension Study},
  journal = {Journal of Artificial Societies and Social Simulation},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## How to Cite

If you use this repository in your research, please cite:

> de la Serna, J. M. (2026). *Civilization Abm V3*. Universidad Internacional de La Rioja (UNIR).
> https://github.com/juanmoisesd/civilization-abm-v3 

See `CITATION.cff` for formatted references.
