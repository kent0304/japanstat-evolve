"""Shinka Evolve evaluator for Table QA agent scaffold.

Dynamically imports the evolved program, runs it on group_a questions,
scores with deterministic entity/numeric matching, and writes
metrics.json + correct.json for Shinka consumption.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
_QA_GROUP = os.environ.get("TABLE_QA_GROUP", "group_train")
QA_PATH = TASK_DIR / "data" / _QA_GROUP / "qa.json"

# Number of times each question is evaluated; final score is the average.
# Reduces variance from LLM stochasticity at the cost of evaluation time.
NUM_RUNS = 2

# Shared cost log across all evaluations in a run.
# Delete this file manually when starting a fresh experiment.
COST_LOG_PATH = TASK_DIR / "results" / "cost_log.jsonl"


def _log_and_print_cost(eval_cost: float, results_dir: str) -> None:
    """Atomically append eval cost, then print this eval's cost + cumulative total."""
    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({"cost": eval_cost, "results_dir": results_dir, "time": time.time()})
    with open(COST_LOG_PATH, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(entry + "\n")
        f.seek(0)
        lines = [l for l in f.readlines() if l.strip()]
        fcntl.flock(f, fcntl.LOCK_UN)
    cumulative = sum(json.loads(l)["cost"] for l in lines)
    n_evals = len(lines)
    print(f"\n{'─'*50}")
    print(f"  評価コスト (今回):  ${eval_cost:.4f}")
    print(f"  累計コスト ({n_evals}回目): ${cumulative:.4f}")
    print(f"{'─'*50}")

# Ensure task directory is on sys.path so evolved modules can
# resolve `from utils import ...` even when loaded from a temp dir.
sys.path.insert(0, str(TASK_DIR))

from scoring import score_answer, _load_ground_truth  # noqa: E402
from utils import create_call_limited_query_llm, query_llm  # noqa: E402


def evaluate(program_path: str, results_dir: str) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    start_t = time.time()

    # Point DATA_DIR in the evolved module to the real data directory
    os.environ["TABLE_QA_DATA_DIR"] = str(TASK_DIR / "data")

    # ── 1. Load evolved module ──────────────────────────────────────
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Catch load-time errors (SyntaxError, NameError, ImportError, etc.)
        # and write metrics.json so text_feedback reaches the next proposal LLM.
        elapsed = time.time() - start_t
        error_msg = f"{type(e).__name__}: {e}"
        # Attach a code snippet around the error line so the proposal LLM has
        # enough context to understand and fix the problem, not just the symptom.
        snippet = ""
        if isinstance(e, (SyntaxError, IndentationError)) and e.lineno:
            try:
                with open(program_path, encoding="utf-8") as _f:
                    src_lines = _f.readlines()
                lo = max(0, e.lineno - 4)
                hi = min(len(src_lines), e.lineno + 3)
                numbered = "".join(
                    f"{'>>>' if i + 1 == e.lineno else '   '} {i + 1:4d} | {src_lines[i]}"
                    for i in range(lo, hi)
                )
                snippet = f"\n\nCode around the error (line {e.lineno}):\n```python\n{numbered}```"
            except Exception:
                pass
        metrics = {
            "combined_score": 0.0,
            "runtime": elapsed,
            "public": {"mean_accuracy": 0, "questions_correct": 0, "questions_total": 0, "pass_rate": 0, "num_runs_per_question": NUM_RUNS, "total_llm_cost_usd": 0.0},
            "private": {"per_question": []},
            "text_feedback": f"Program failed to load. Fix this error before anything else: {error_msg}{snippet}",
        }
        metrics_file = os.path.join(results_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        correct_file = os.path.join(results_dir, "correct.json")
        with open(correct_file, "w", encoding="utf-8") as f:
            json.dump({"correct": False, "error": error_msg}, f, indent=2)
        print(f"Load error: {error_msg}")
        raise

    # ── 2. Load questions and ground truth ──────────────────────────
    with open(QA_PATH, encoding="utf-8") as f:
        qa = json.load(f)
    gt_data = _load_ground_truth()

    # ── 3. Run each question ────────────────────────────────────────
    results = []
    total_cost = 0.0
    error = ""
    correct_flag = True

    try:
        for item in qa:
            qid = item["id"]
            entry: dict = {"id": qid, "question": item["question"]}

            run_scores: list[float] = []
            run_cost = 0.0
            last_answer = None

            for run_idx in range(NUM_RUNS):
                try:
                    limited_qlm = create_call_limited_query_llm(query_llm, 10)
                    answer, cost = module.run(
                        question=item["question"],
                        query_llm_fn=limited_qlm,
                    )
                    run_cost += cost or 0
                    last_answer = answer
                except Exception as e:
                    answer = None
                    if run_idx == 0:
                        entry["error"] = str(e)

                sc = score_answer(qid, answer or "", gt_data)
                run_scores.append(sc["combined_score"])
                print(f"  [{qid}] run={run_idx+1}/{NUM_RUNS} score={sc['combined_score']:.3f}")

            avg_score = sum(run_scores) / len(run_scores)
            total_cost += run_cost
            entry["answer"] = last_answer
            entry["cost"] = run_cost
            entry["score"] = avg_score
            entry["correct"] = avg_score > 0
            entry["run_scores"] = run_scores
            entry["text_feedback"] = score_answer(qid, last_answer or "", gt_data)["text_feedback"]
            results.append(entry)

            print(f"  [{qid}] avg_score={avg_score:.3f} runs={run_scores}")
    except Exception as e:
        error = str(e)
        correct_flag = False

    # ── 4. Aggregate fitness (use pre-averaged per-question scores) ──
    scores = [r["score"] for r in results]
    n = len(scores)
    mean_score = sum(scores) / n if n > 0 else 0.0
    n_correct = sum(1 for s in scores if s > 0)
    fitness = {
        "combined_score": round(mean_score * 100, 2),
        "public": {
            "mean_accuracy": round(mean_score, 4),
            "questions_correct": n_correct,
            "questions_total": n,
            "pass_rate": round(n_correct / n, 3) if n > 0 else 0.0,
            "num_runs_per_question": NUM_RUNS,
        },
        "private": {
            "per_question": [
                {"id": r["id"], "avg_score": r["score"], "run_scores": r.get("run_scores", [])}
                for r in results
            ],
        },
        "text_feedback": (
            f"{n_correct}/{n} questions answered correctly (avg over {NUM_RUNS} runs). "
            f"Mean accuracy: {mean_score:.1%}."
        ),
    } if results else {
        "combined_score": 0.0,
        "public": {"mean_accuracy": 0, "questions_correct": 0, "questions_total": 0, "pass_rate": 0, "num_runs_per_question": NUM_RUNS},
        "private": {"per_question": []},
        "text_feedback": "No results.",
    }

    elapsed = time.time() - start_t

    # ── 5. Build metrics dict ───────────────────────────────────────
    metrics = {
        "combined_score": fitness["combined_score"],
        "runtime": elapsed,
        "public": {
            **fitness["public"],
            "total_llm_cost_usd": round(total_cost, 6),
        },
        "private": fitness["private"],
        "text_feedback": fitness["text_feedback"],
    }

    # ── 6. Write output files ───────────────────────────────────────
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_file}")

    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w", encoding="utf-8") as f:
        json.dump({"correct": correct_flag, "error": error}, f, indent=2)
    print(f"Correct saved to {correct_file}")

    # ── 7. Print summary ────────────────────────────────────────────
    _log_and_print_cost(total_cost, results_dir)
    print(f"\ncombined_score: {metrics['combined_score']}")
    print(f"pass_rate:      {fitness['public']['pass_rate']}")
    print(f"correct:        {fitness['public']['questions_correct']}/{fitness['public']['questions_total']}")
    print(f"LLM cost:       ${total_cost:.4f}")
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Completed after {hours}h {minutes}m {seconds}s")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table QA evaluator for Shinka Evolve")
    parser.add_argument(
        "--program_path", type=str, default=str(TASK_DIR / "initial.py"),
        help="Path to the evolved program",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to save metrics.json and correct.json",
    )
    args = parser.parse_args()
    evaluate(args.program_path, args.results_dir)
