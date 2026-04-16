"""Deterministic scoring for Table QA answers.

DAB-style per-question validation with partial credit for continuous scores.
No LLM calls — purely regex + string matching.
"""

import json
import os
import re
from typing import Any

# 47 prefectures for entity extraction
PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

# Short-name → full-name mapping (e.g. "東京" → "東京都")
_SHORT_TO_FULL: dict[str, str] = {}
for _p in PREFECTURES:
    if _p.endswith(("都", "府")):
        _SHORT_TO_FULL[_p[:-1]] = _p  # 東京 → 東京都
    elif _p.endswith("県"):
        _SHORT_TO_FULL[_p[:-1]] = _p  # 青森 → 青森県
    elif _p == "北海道":
        pass  # 北海道 is already unique

# Build regex: match full names first, then short names (2+ chars)
_pref_pattern = "|".join(
    sorted(list(PREFECTURES) + list(_SHORT_TO_FULL.keys()), key=len, reverse=True)
)
_PREF_RE = re.compile(f"({_pref_pattern})")

# Numeric extraction: integers, decimals, negative numbers
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _load_ground_truth(gt_path: str | None = None) -> dict:
    if gt_path is None:
        _group = os.environ.get("TABLE_QA_GROUP", "group_train")
        gt_path = os.path.join(
            os.path.dirname(__file__), "data", _group, "ground_truth.json"
        )
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


def extract_prefectures(text: str) -> list[str]:
    """Extract prefecture names from text, preserving order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _PREF_RE.finditer(text):
        name = m.group()
        full = _SHORT_TO_FULL.get(name, name)
        if full not in seen:
            seen.add(full)
            result.append(full)
    return result


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    results = []
    for m in _NUM_RE.findall(text):
        try:
            results.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return results


# ── Scoring functions ────────────────────────────────────────────────────────


def _score_ranked(pred_entities: list[str], gt: dict) -> dict:
    """Score a ranked (TOP-N) answer. Returns entity_score in [0, 1]."""
    gt_entities = gt["entities"]
    n = len(gt_entities)
    if n == 0:
        return {"entity_score": 1.0, "detail": "no entities to check"}

    # Set-level: what fraction of expected entities appear at all
    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    recall = len(pred_set & gt_set) / n
    precision = len(pred_set & gt_set) / len(pred_set) if pred_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    # Order score: fraction of entities in exactly the correct position
    positional_hits = 0
    for i, gt_e in enumerate(gt_entities):
        if i < len(pred_entities) and pred_entities[i] == gt_e:
            positional_hits += 1
    positional_score = positional_hits / n

    # Combined: weight order more than just set membership
    entity_score = 0.5 * positional_score + 0.5 * f1

    return {
        "entity_score": entity_score,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "positional_score": positional_score,
        "matched": sorted(pred_set & gt_set),
        "missing": sorted(gt_set - pred_set),
        "extra": sorted(pred_set - gt_set),
    }


def _score_set(pred_entities: list[str], gt: dict) -> dict:
    """Score a set-matching answer (order doesn't matter). Returns entity_score in [0, 1]."""
    gt_entities = gt["entities"]
    n = len(gt_entities)
    if n == 0:
        return {"entity_score": 1.0, "detail": "no entities to check"}

    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    recall = len(pred_set & gt_set) / n
    precision = len(pred_set & gt_set) / len(pred_set) if pred_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    return {
        "entity_score": f1,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matched": sorted(pred_set & gt_set),
        "missing": sorted(gt_set - pred_set),
        "extra": sorted(pred_set - gt_set),
    }


def _score_minmax(pred_entities: list[str], gt: dict) -> dict:
    """Score a min/max answer."""
    pred_set = set(pred_entities)
    checks = []
    total = 0

    for key in ("max_entities", "min_entities"):
        if key not in gt:
            continue
        expected = set(gt[key])
        total += len(expected)
        hit = len(pred_set & expected)
        checks.append({"key": key, "expected": sorted(expected), "hit": hit})

    if total == 0:
        return {"entity_score": 1.0, "detail": "no entities to check"}

    hits = sum(c["hit"] for c in checks)
    return {
        "entity_score": hits / total,
        "checks": checks,
    }


def _score_ranked_dual(pred_entities: list[str], gt: dict) -> dict:
    """Score a dual-ranked answer (TOP-N and BOTTOM-N)."""
    pred_set = set(pred_entities)
    top_expected = set(gt.get("top_entities", []))
    bottom_expected = set(gt.get("bottom_entities", []))

    top_hits = len(pred_set & top_expected)
    bottom_hits = len(pred_set & bottom_expected)
    total = len(top_expected) + len(bottom_expected)
    if total == 0:
        return {"entity_score": 1.0}

    return {
        "entity_score": (top_hits + bottom_hits) / total,
        "top_matched": sorted(pred_set & top_expected),
        "top_missing": sorted(top_expected - pred_set),
        "bottom_matched": sorted(pred_set & bottom_expected),
        "bottom_missing": sorted(bottom_expected - pred_set),
    }


def _score_unanswerable(pred_text: str, gt: dict) -> dict:
    """Score detection of unanswerable questions."""
    keywords = gt.get("unanswerable_keywords", [])
    text_lower = pred_text.lower()
    detected = any(kw.lower() in text_lower for kw in keywords)

    result = {
        "entity_score": 1.0 if detected else 0.0,
        "detected_unanswerable": detected,
    }

    # Some unanswerable questions offer partial credit for fallback answers
    if not detected and "partial_credit" in gt:
        pc = gt["partial_credit"]
        pred_entities = extract_prefectures(pred_text)
        if pc["type"] == "minmax":
            sub = _score_minmax(pred_entities, pc)
            result["entity_score"] = sub["entity_score"] * 0.5  # half credit
            result["partial_credit_detail"] = sub

    return result


def _score_numeric(pred_text: str, gt: dict) -> dict:
    """Score numeric-only answers."""
    answers = gt.get("numeric_answers", [])
    if not answers:
        return {"numeric_score": 1.0}

    pred_nums = extract_numbers(pred_text)
    hits = 0
    details = []

    for ans in answers:
        gt_val = ans["value"]
        tol = ans.get("tolerance_abs", 0)
        matched = any(abs(n - gt_val) <= tol for n in pred_nums)
        if matched:
            hits += 1
        details.append({
            "label": ans.get("label", ""),
            "expected": gt_val,
            "matched": matched,
        })

    return {
        "numeric_score": hits / len(answers),
        "numeric_details": details,
    }


def _score_numeric_and_entity(
    pred_text: str, pred_entities: list[str], gt: dict
) -> dict:
    """Score answers that combine numeric values and entity names."""
    num_result = _score_numeric(pred_text, gt)

    gt_ents = gt.get("entities", [])
    if gt_ents:
        pred_set = set(pred_entities)
        gt_set = set(gt_ents)
        ent_recall = len(pred_set & gt_set) / len(gt_set)
    else:
        ent_recall = 1.0  # no entities to check

    # Combine numeric and entity scores
    num_w = len(gt.get("numeric_answers", []))
    ent_w = len(gt_ents)
    total_w = num_w + ent_w
    if total_w == 0:
        combined = 1.0
    else:
        combined = (
            num_result["numeric_score"] * num_w + ent_recall * ent_w
        ) / total_w

    return {
        "entity_score": combined,
        "numeric_score": num_result["numeric_score"],
        "entity_recall": ent_recall,
        "numeric_details": num_result.get("numeric_details", []),
    }


def _score_ranked_and_minmax(pred_entities: list[str], gt: dict) -> dict:
    """Score answers with ranked TOP-N + separate min entity."""
    pred_set = set(pred_entities)

    # TOP part
    top_expected = gt.get("top_entities", [])
    top_hits = len(pred_set & set(top_expected))
    top_n = len(top_expected)

    # MIN part
    min_expected = set(gt.get("min_entities", []))
    min_hits = len(pred_set & min_expected)
    min_n = len(min_expected)

    total = top_n + min_n
    if total == 0:
        return {"entity_score": 1.0}

    return {
        "entity_score": (top_hits + min_hits) / total,
        "top_matched": sorted(pred_set & set(top_expected)),
        "top_missing": sorted(set(top_expected) - pred_set),
        "min_matched": sorted(pred_set & min_expected),
        "min_missing": sorted(min_expected - pred_set),
    }


# ── Main scoring entry point ────────────────────────────────────────────────


def score_answer(qid: str, pred_text: str, gt_data: dict | None = None) -> dict:
    """Score a single answer against ground truth.

    Returns:
        dict with keys:
            combined_score (float): 0.0–1.0, higher is better
            correct (bool): whether the answer passes minimum threshold
            public (dict): metrics visible to the LLM for mutation
            private (dict): detailed breakdown for debugging
            text_feedback (str): actionable improvement hint
    """
    if gt_data is None:
        gt_data = _load_ground_truth()

    gt = gt_data.get(qid)
    if gt is None:
        return {
            "combined_score": 0.0,
            "correct": False,
            "public": {},
            "private": {},
            "text_feedback": f"No ground truth found for {qid}",
        }

    # Handle empty/None answers
    if not pred_text or pred_text.strip() == "":
        return {
            "combined_score": 0.0,
            "correct": False,
            "public": {"entity_score": 0.0},
            "private": {},
            "text_feedback": "Empty answer",
        }

    qtype = gt["type"]
    pred_entities = extract_prefectures(pred_text)

    # Dispatch by question type
    if qtype == "ranked":
        detail = _score_ranked(pred_entities, gt)
    elif qtype == "set":
        detail = _score_set(pred_entities, gt)
    elif qtype == "minmax":
        detail = _score_minmax(pred_entities, gt)
    elif qtype == "ranked_dual":
        detail = _score_ranked_dual(pred_entities, gt)
    elif qtype == "unanswerable":
        detail = _score_unanswerable(pred_text, gt)
    elif qtype == "numeric":
        detail = _score_numeric(pred_text, gt)
        detail["entity_score"] = detail.pop("numeric_score")
    elif qtype == "numeric_and_entity":
        detail = _score_numeric_and_entity(pred_text, pred_entities, gt)
    elif qtype == "ranked_and_minmax":
        detail = _score_ranked_and_minmax(pred_entities, gt)
    else:
        return {
            "combined_score": 0.0,
            "correct": False,
            "public": {},
            "private": {"error": f"Unknown question type: {qtype}"},
            "text_feedback": f"Unknown question type: {qtype}",
        }

    entity_score = detail.get("entity_score", 0.0)

    # Execution error detection: penalize but don't zero out if partial info exists
    has_exec_error = "[実行エラー]" in pred_text or "Traceback" in pred_text
    error_penalty = 0.5 if has_exec_error else 1.0

    combined_score = entity_score * error_penalty

    # correct gate: lenient — any non-zero score passes
    correct = combined_score > 0.0

    # Build public metrics (visible to LLM)
    public: dict[str, Any] = {
        "accuracy": round(entity_score, 3),
        "question_type": qtype,
    }
    if has_exec_error:
        public["execution_error"] = True
    if "recall" in detail:
        public["recall"] = round(detail["recall"], 3)
        public["precision"] = round(detail["precision"], 3)
    if "numeric_score" in detail:
        public["numeric_accuracy"] = round(detail["numeric_score"], 3)

    # Build text feedback
    feedback_parts = []
    if has_exec_error:
        feedback_parts.append("Code execution error detected — score halved.")
    if "missing" in detail and detail["missing"]:
        feedback_parts.append(
            f"Missing entities: {', '.join(detail['missing'][:5])}"
        )
    if "extra" in detail and detail["extra"]:
        feedback_parts.append(
            f"Extra (wrong) entities: {', '.join(detail['extra'][:5])}"
        )
    if "top_missing" in detail and detail["top_missing"]:
        feedback_parts.append(
            f"Missing from TOP: {', '.join(detail['top_missing'][:5])}"
        )
    if "bottom_missing" in detail and detail["bottom_missing"]:
        feedback_parts.append(
            f"Missing from BOTTOM: {', '.join(detail['bottom_missing'][:5])}"
        )
    if "min_missing" in detail and detail["min_missing"]:
        feedback_parts.append(
            f"Missing min entity: {', '.join(detail['min_missing'])}"
        )
    if qtype == "unanswerable" and not detail.get("detected_unanswerable"):
        feedback_parts.append(
            "This question's data is unavailable (***) — "
            "answer should detect and report this."
        )
    if "numeric_details" in detail:
        for nd in detail["numeric_details"]:
            if not nd["matched"]:
                feedback_parts.append(
                    f"Numeric mismatch for '{nd['label']}': "
                    f"expected {nd['expected']}"
                )
    if not feedback_parts:
        feedback_parts.append("Correct answer.")

    text_feedback = " | ".join(feedback_parts)

    return {
        "combined_score": round(combined_score, 4),
        "correct": correct,
        "public": public,
        "private": detail,
        "text_feedback": text_feedback,
    }


def score_all(
    results: list[dict],
    gt_path: str | None = None,
) -> dict:
    """Score all answers and return aggregate fitness.

    Args:
        results: list of {"id": "Q1", "answer": "..."} dicts
        gt_path: optional path to ground_truth.json

    Returns:
        dict following ShinkaEvolve fitness structure
    """
    gt_data = _load_ground_truth(gt_path)
    per_question: list[dict] = []
    scores: list[float] = []

    for r in results:
        qid = r["id"]
        answer = r.get("answer") or ""
        result = score_answer(qid, answer, gt_data)
        per_question.append({"id": qid, **result})
        scores.append(result["combined_score"])

    n = len(scores)
    mean_score = sum(scores) / n if n > 0 else 0.0
    n_correct = sum(1 for s in scores if s > 0)

    return {
        "combined_score": round(mean_score * 100, 2),  # 0–100 scale
        "correct": mean_score > 0.0,
        "public": {
            "mean_accuracy": round(mean_score, 4),
            "questions_correct": n_correct,
            "questions_total": n,
            "pass_rate": round(n_correct / n, 3) if n > 0 else 0.0,
        },
        "private": {
            "per_question": per_question,
        },
        "text_feedback": (
            f"{n_correct}/{n} questions answered correctly. "
            f"Mean accuracy: {mean_score:.1%}."
        ),
    }


# ── CLI for quick testing ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    infer_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "infer.json"
    )
    with open(infer_path, encoding="utf-8") as f:
        results = json.load(f)

    fitness = score_all(results)
    print(f"\n=== Fitness Summary ===")
    print(f"combined_score: {fitness['combined_score']}")
    print(f"pass_rate: {fitness['public']['pass_rate']}")
    print(f"questions_correct: {fitness['public']['questions_correct']}/{fitness['public']['questions_total']}")
    print(f"text_feedback: {fitness['text_feedback']}")
    print()

    for pq in fitness["private"]["per_question"]:
        mark = "OK" if pq["correct"] else "NG"
        print(f"  [{mark}] {pq['id']}: score={pq['combined_score']:.3f}  {pq['text_feedback'][:80]}")
