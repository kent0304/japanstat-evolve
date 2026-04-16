# JapanStat-Evolve: Automated Optimization of Table Q&A Agents via ShinkaEvolve

A proof-of-concept that uses **[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve)** to evolutionarily optimize the scaffold (prompts, pipeline structure, and parameters) of an LLM agent for Japanese government statistics (e-Stat) table Q&A.

| | Dev Set (20 Q) | Test Set (20 Q) |
|---|---|---|
| **Baseline** | 26.1 | 35.0 |
| **Best Evolved** | **74.0** | **69.75** |
| **Improvement** | +184% | +99% |

## Overview

A 3-stage LLM agent (Router → Analyst → Verifier) is evolved over 130 generations with ShinkaEvolve, significantly improving its score on a Japanese prefectural statistics Q&A benchmark. ShinkaEvolve optimizes the regions enclosed by `EVOLVE-BLOCK` markers in `initial.py` (prompts, parameters, and flow control) through LLM-based mutations.

![Pipeline](figures/pipeline.png)

**What this repository enables:**

1. **Run the baseline agent** -- Try the 3-stage pipeline with `initial.py`
2. **Run evolutionary optimization** -- Launch automatic evolution with `run_evo.py`
3. **Evaluate and compare** -- Compare baseline vs. best evolved program with `evaluate.py`


## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Gemini API key or OpenAI API key

### Setup

```bash
git clone https://github.com/kent0304/japanstat-evolve.git
cd japanstat-evolve

# Install dependencies
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env and set your API keys
```

### Evaluate the baseline

```bash
uv run python evaluate.py
```

### Evaluate the best evolved program

```bash
uv run python evaluate.py --program_path results/best_evolved/main.py
```

### Evaluate on the test set

```bash
TABLE_QA_GROUP=group_test uv run python evaluate.py --program_path results/best_evolved/main.py
```

### Run evolution with ShinkaEvolve

```bash
uv run python run_evo.py --config_path evo_config.yaml
```

> **Note:** Running evolution incurs Gemini/OpenAI API costs (~$29 for the full experiment). Adjust `num_generations` and `max_api_costs` in `evo_config.yaml` to control the budget.

## Benchmark: JapanStat-Bench

### Data Source

Three statistical tables sourced from e-Stat (Portal Site of Official Statistics of Japan):

| Table | File | Columns | Content |
|---|---|---|---|
| Population | `population_2017-2022.csv` | 53 | Total population, age distribution, birth/death rates |
| Finance | `finance_2017-2022.csv` | 25 | Fiscal capacity index, revenue/expenditure, local taxes |
| Labor | `labor_2017-2022.csv` | 127 | Job openings, wages, minimum wage |

Each table contains 96 rows (47 prefectures + national aggregate) with data from 2017 and 2022.

### Problem Design

40 questions (20 train + 20 test) across 5 themes:

- Labor market supply & demand (6 Q)
- Wage structure (10 Q)
- Fiscal-labor-population cross-analysis (8 Q)
- Population dynamics (6 Q)
- Time-series changes 2017 → 2022 (10 Q)

Domain-specific traps (unit conversion, year-dependent column names, national record handling, etc.) are intentionally embedded.

### Scoring

Deterministic scoring (entity matching + numeric comparison) without LLM-as-judge, ensuring reproducible evaluation.

## Repository Structure

```
japanstat-evolve/
├── initial.py              # Baseline 3-stage agent (seed for evolution)
├── run_evo.py              # ShinkaEvolve launcher
├── evaluate.py             # Evaluation harness
├── scoring.py              # Deterministic scoring
├── utils.py                # LLM query utilities
├── evo_config.yaml       # Evolution config
├── pyproject.toml          # Dependencies (uv)
├── data/
│   ├── schema.md           # Column dictionary (205 columns)
│   ├── qa.json             # Full 40-question master set
│   ├── population_2017-2022.csv
│   ├── finance_2017-2022.csv
│   ├── labor_2017-2022.csv
│   ├── group_train/        # Training split (20 Q)
│   └── group_test/         # Test split (20 Q)
├── results/
│   ├── best_evolved/       # Best evolved program (score 74.0)
│   ├── eval_test_evolved/  # Test set results (evolved)
│   ├── eval_test_baseline/ # Test set results (baseline)
│   └── figures/            # Analysis figures
├── figures/
│   └── pipeline.png        # Architecture diagram
└── LICENSE                 # Apache 2.0
```

## Evolution Config

Key parameters in `evo_config.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `num_generations` | 130 | Number of generations |
| `num_islands` | 2 (→ max 10) | Island model (dynamic expansion) |
| `patch_types` | diff:70%, full:20%, cross:10% | Mutation types |
| `llm_models` | gemini-2.5-flash-lite, gemini-2.5-flash, gpt-5-nano | Proposal LLMs |
| `max_api_costs` | $15 | Proposal LLM cost cap |
| `evolve_prompts` | true | Prompt co-evolution |

## Data Attribution

The CSV files in `data/` are derived from statistics published on **e-Stat** (https://www.e-stat.go.jp/), the Portal Site of Official Statistics of Japan. Used in accordance with the terms of use for government statistics.

## Citation

This project uses [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

```bibtex
@software{shinkaevolve2025,
  title={ShinkaEvolve: LLM-Driven Evolutionary Optimization},
  author={Sakana AI},
  year={2025},
  url={https://github.com/SakanaAI/ShinkaEvolve}
}
```

## License

[Apache License 2.0](LICENSE)
