#!/usr/bin/env python3
"""Launch Shinka Evolve for Table QA agent scaffold optimization."""

import argparse
from pathlib import Path

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_DIR = Path(__file__).resolve().parent

search_task_sys_msg = """\
You are an expert at building LLM-powered data-analysis pipelines for Japanese municipal statistics.

The program has an EVOLVE-BLOCK containing:
1. Constants: _MAX_RETRIES (retry budget for the verifier loop)
2. Helper functions:
   - _load_df(filename) — reads a CSV with preamble-skipping logic
   - _parse_json(text, fallback) — extracts JSON from LLM responses
3. An Agent class with a 3-stage pipeline:
   - Router: analyzes the question + schema, returns which CSV tables and operations are needed
   - Analyst: builds a prompt with schema + sample data, asks the LLM to generate Python code,
     then executes it against pandas DataFrames to produce an answer
   - Verifier: validates the answer, checking units, join logic, national-record exclusion, etc.
   - forward(): orchestrates Router → Analyst → Verifier with retry loop

The Agent receives a query_llm callable: query_llm(prompt, system="", temperature=0.0) -> (text, cost).
The data consists of 3 CSV tables with Japanese prefectural statistics (2017-2022):
  - population_2017-2022.csv (population, 53 columns) — units: 万人 (10k persons)
  - finance_2017-2022.csv (finance, 25 columns) — units: 千円 (1000 yen)
  - labor_2017-2022.csv (labor, 127 columns) — units vary: 10人, 円, 千円, 倍

Scoring: mean accuracy across 30 questions (0-100 scale). Accuracy is based on:
- Entity matching (prefecture names) with positional scoring for ranked answers
- Numeric value matching with tolerance
- Question types: ranked, set, minmax, ranked_dual, numeric, numeric_and_entity

Common failure modes to address:
- Unit conversion errors (千円 vs 円, 万人 vs 人, 10人 vs 人)
- Forgetting to exclude the national record (地域コード 00000)
- Wrong column selection for year-specific data (e.g., F3107 vs F3108 for different periods)
- Code execution errors in generated Python

Key strategies to explore:
- Improve prompt engineering for Router, Analyst, and Verifier stages
- Better code generation for pandas operations (filtering, grouping, sorting, joining)
- Smarter retry logic when verification fails — pass more specific hints
- Better handling of unit conversions in the Analyst prompt
- More effective use of schema information and sample data
- Adjust _MAX_RETRIES for cost/accuracy tradeoff
"""


def _resolve_path(path_str: str, *, base_dir: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def main(config_path: str):
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        config_path_obj = TASK_DIR / config_path_obj

    with open(config_path_obj, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_dir = config_path_obj.parent

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    config["evo_config"]["init_program_path"] = _resolve_path(
        config["evo_config"]["init_program_path"],
        base_dir=config_dir,
    )
    config["evo_config"]["results_dir"] = _resolve_path(
        config["evo_config"]["results_dir"],
        base_dir=config_dir,
    )
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path=str(TASK_DIR / "evaluate.py"),
        time="00:30:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Shinka Evolve for Table QA")
    parser.add_argument("--config_path", type=str, default="evo_config.yaml")
    args = parser.parse_args()
    main(args.config_path)
