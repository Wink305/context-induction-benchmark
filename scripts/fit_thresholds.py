"""Fit the scorer's content gates on a human-labelled calibration set.

Why this script exists
----------------------
``scoring.py`` decides that a predicted constraint "has the right content" by
comparing character bigrams against a threshold that was **hand-picked** so
that a short paraphrase of a constructed unit survives. That threshold was
never checked against human labels, because no human gold existed. The
consequence is a scorer that accepts a semantically wrong prediction as long
as enough characters overlap, and rejects a correct paraphrase that happens
to be worded differently.

This script closes that loop *when human labels exist*:

1. read the human gold (the output of ``parse-sheets``) and one model's
   predictions for the same dialogues;
2. sweep a grid of ``text_min`` / ``anchor_coverage_min`` values;
3. pick the threshold that maximises recall subject to a precision floor;
4. report **out-of-fold** sensitivity and specificity (5-fold, split by
   dialogue) — in-sample numbers would only measure how well the grid fits
   the calibration set;
5. optionally write ``configs/calibration.json`` with the provenance the
   repository's gate requires.

Hard rules this script will not break:

* It never reads ``candidate_truth.jsonl``. Tuning thresholds against
  generator truth is the self-certifying loop the calibration file exists to
  prevent.
* It refuses to write a "calibrated" config without a third-party adjudicator
  name — two annotators and a dispute have to exist before a number is
  allowed to look calibrated.

Usage::

    python scripts/fit_thresholds.py \
        --gold data/annotation/gold_v1.jsonl \
        --predictions runs/<run>/predictions.jsonl \
        --tasks data/pilot_v0/tasks.jsonl \
        --adjudicator "名字" \
        --write configs/calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from context_benchmark.scoring import (  # noqa: E402
    _dice, _ngrams, anchor_coverage, greedy_match,
)
from context_benchmark.semantic import semantic_conflict  # noqa: E402

DEFAULT_TEXT_GRID = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
DEFAULT_ANCHOR_GRID = (0.0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_MIN_PRECISION = 0.95
DEFAULT_FOLDS = 5


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gold_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    units = row.get("annotations")
    if not isinstance(units, list):
        units = row.get("constraints")
    return units if isinstance(units, list) else []


def prediction_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = row.get("output")
    if not isinstance(output, dict):
        output = {"constraints": row.get("constraints", [])}
    units = output.get("constraints")
    return units if isinstance(units, list) else []


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    turns: dict[int, dict[str, Any]],
    text_min: float,
    anchor_min: float,
) -> dict[str, Any]:
    """Score one (gold, prediction) pair under one threshold setting.

    A match counts as a true positive only when the text gate passes, the
    semantic guard does not fire, and type and scope agree. A match that
    fails any of those is a false positive: the scorer would have credited a
    wrong answer.
    """
    matches, unmatched_gold, unmatched_predictions = greedy_match(
        gold, predictions, turns=turns, text_min=text_min, anchor_min=anchor_min
    )
    tp = 0
    fp = 0
    for gi, pi, _score in matches:
        gold_item = gold[gi]
        prediction = predictions[pi]
        text_sim = _dice(_ngrams(gold_item.get("text", "")), _ngrams(prediction.get("text", "")))
        conflict = semantic_conflict(gold_item.get("text", ""), prediction.get("text", ""))
        content_ok = (
            text_sim >= text_min
            and anchor_coverage(
                prediction.get("text", ""), turns, prediction.get("evidence_turns", [])
            ) >= anchor_min
            and not conflict["conflicted"]
        )
        type_ok = prediction.get("type") == gold_item.get("type")
        scope_ok = prediction.get("scope") == gold_item.get("scope")
        if content_ok and type_ok and scope_ok:
            tp += 1
        else:
            fp += 1
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / len(gold) if gold else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else (0.0 if (tp + fp) and gold else None)
    )
    return {
        "text_min": text_min,
        "anchor_coverage_min": anchor_min,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_n": len(gold),
        "matched_n": len(matches),
        "unmatched_gold_n": len(unmatched_gold),
        "unmatched_prediction_n": len(unmatched_predictions),
    }


def choose_threshold(
    grid: list[dict[str, Any]], min_precision: float = DEFAULT_MIN_PRECISION,
) -> tuple[dict[str, Any], str, bool]:
    """Maximise recall subject to a precision floor; fall back to F1.

    The boolean is ``True`` when the floor could not be met anywhere in the
    grid — in that case the chosen threshold is *not* safe to report and the
    caller must say so.
    """
    usable = [row for row in grid if row["precision"] is not None]
    if not usable:
        raise ValueError("the grid produced no scorable row")
    feasible = [row for row in usable if row["precision"] >= min_precision]
    if feasible:
        best = max(feasible, key=lambda row: (row["recall"], row["f1"], -row["text_min"]))
        return best, f"max recall subject to precision >= {min_precision}", False
    best = max(usable, key=lambda row: (row["f1"], row["recall"] or 0.0))
    return best, f"no threshold reached precision >= {min_precision}; F1 maximised instead", True


def fold_of(dialogue_id: str, folds: int) -> int:
    digest = hashlib.sha256(dialogue_id.encode("utf-8")).digest()
    return digest[0] % folds


def sweep(
    pairs: list[dict[str, Any]],
    text_grid: Iterable[float],
    anchor_grid: Iterable[float],
) -> list[dict[str, Any]]:
    rows = []
    for text_min in text_grid:
        for anchor_min in anchor_grid:
            merged = {
                "tp": 0, "fp": 0, "fn": 0, "gold_n": 0,
                "matched_n": 0, "unmatched_gold_n": 0, "unmatched_prediction_n": 0,
            }
            precision_parts: list[tuple[int, int]] = []
            for pair in pairs:
                result = evaluate(
                    pair["gold"], pair["predictions"], pair["turns"], text_min, anchor_min
                )
                merged["tp"] += result["tp"]
                merged["fp"] += result["fp"]
                merged["fn"] += result["fn"]
                merged["gold_n"] += result["gold_n"]
                merged["matched_n"] += result["matched_n"]
                merged["unmatched_gold_n"] += result["unmatched_gold_n"]
                merged["unmatched_prediction_n"] += result["unmatched_prediction_n"]
                precision_parts.append((result["tp"], result["fp"]))
            tp = merged["tp"]
            fp = merged["fp"]
            merged["precision"] = tp / (tp + fp) if (tp + fp) else None
            merged["recall"] = tp / merged["gold_n"] if merged["gold_n"] else None
            merged["f1"] = (
                2 * merged["precision"] * merged["recall"] / (merged["precision"] + merged["recall"])
                if merged["precision"] is not None and merged["recall"] is not None
                and (merged["precision"] + merged["recall"])
                else (0.0 if (tp + fp) and merged["gold_n"] else None)
            )
            merged["text_min"] = text_min
            merged["anchor_coverage_min"] = anchor_min
            rows.append(merged)
    return rows


def cross_validate(
    pairs: list[dict[str, Any]],
    text_grid: Iterable[float],
    anchor_grid: Iterable[float],
    min_precision: float,
    folds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose on the training folds, evaluate on the held-out fold."""
    grid = list(text_grid)
    anchors = list(anchor_grid)
    fold_rows = []
    for index in range(folds):
        train = [pair for pair in pairs if pair["fold"] != index]
        test = [pair for pair in pairs if pair["fold"] == index]
        if not train or not test:
            continue
        chosen, rule, degraded = choose_threshold(sweep(train, grid, anchors), min_precision)
        held_out = sweep(test, [chosen["text_min"]], [chosen["anchor_coverage_min"]])[0]
        held_out["fold"] = index
        held_out["chosen_text_min"] = chosen["text_min"]
        held_out["chosen_anchor_coverage_min"] = chosen["anchor_coverage_min"]
        held_out["rule"] = rule
        held_out["degraded"] = degraded
        fold_rows.append(held_out)
    tp = sum(row["tp"] for row in fold_rows)
    fp = sum(row["fp"] for row in fold_rows)
    fn = sum(row["fn"] for row in fold_rows)
    gold_n = sum(row["gold_n"] for row in fold_rows)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / gold_n if gold_n else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    summary = {
        "out_of_fold": {
            "tp": tp, "fp": fp, "fn": fn, "gold_n": gold_n,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "folds": fold_rows,
        "folds_n": len(fold_rows),
        "degraded_any_fold": any(row["degraded"] for row in fold_rows),
    }
    return fold_rows, summary


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def build_config(
    *,
    gold_path: Path,
    chosen: dict[str, Any],
    summary: dict[str, Any],
    annotators: list[str],
    adjudicator: str,
    folds: int,
    min_precision: float,
) -> dict[str, Any]:
    out_of_fold = summary["out_of_fold"]
    precision = out_of_fold["precision"]
    recall = out_of_fold["recall"]
    try:
        relative = str(gold_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        relative = str(gold_path).replace("\\", "/")
    return {
        "schema_version": 1,
        "state": "calibrated",
        "thresholds": {
            "text_min": chosen["text_min"],
            "anchor_coverage_min": chosen["anchor_coverage_min"],
            "evidence_iou_min": None,
            "slot_match_required": False,
        },
        "provenance": {
            "calibration_set": relative,
            "calibration_set_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            "annotators": annotators,
            "adjudicator": adjudicator,
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample_n": out_of_fold["gold_n"],
            "sensitivity": recall,
            "specificity": precision,
            "split": f"{folds}_fold_by_dialogue_out_of_fold",
        },
        "threshold_sources": {
            "text_min": "fitted by scripts/fit_thresholds.py on the human calibration set",
            "anchor_coverage_min": "fitted by scripts/fit_thresholds.py on the human calibration set",
            "evidence_iou_min": "not implemented yet — the scorer does not read this key",
            "slot_match_required": "not implemented yet — the scorer does not read this key",
        },
        "selection_rule": (
            f"max recall subject to precision >= {min_precision}; "
            f"out-of-fold over {folds} dialogue folds"
        ),
        "note": (
            "sensitivity is the recall over human-labelled gold units; specificity is "
            "the precision of the same judgement (the share of credited predictions "
            "that are genuinely correct), not a classical true-negative rate — this "
            "task has no independent negative set. Both are out-of-fold."
        ),
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", required=True, type=Path, help="human gold jsonl (parse-sheets output)")
    parser.add_argument("--predictions", required=True, type=Path, help="model predictions jsonl")
    parser.add_argument("--tasks", required=True, type=Path, help="tasks.jsonl (for the dialogue turns)")
    parser.add_argument("--write", type=Path, default=None, help="write the calibrated config here")
    parser.add_argument("--adjudicator", default="", help="third-party adjudicator name")
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--text-grid", type=float, nargs="*", default=list(DEFAULT_TEXT_GRID))
    parser.add_argument("--anchor-grid", type=float, nargs="*", default=list(DEFAULT_ANCHOR_GRID))
    args = parser.parse_args(argv)

    if "candidate_truth" in args.gold.name:
        print("refusing to fit against a generator-truth file: calibration needs human labels")
        return 2

    tasks = {row["dialogue_id"]: row for row in read_jsonl(args.tasks)}
    gold_rows = {row["dialogue_id"]: row for row in read_jsonl(args.gold)}
    prediction_rows = {row["dialogue_id"]: row for row in read_jsonl(args.predictions)}

    pairs = []
    annotators: set[str] = set()
    for dialogue_id, gold_row in gold_rows.items():
        task = tasks.get(dialogue_id)
        if task is None:
            print(f"skipping {dialogue_id}: no turns in the tasks file")
            continue
        turns = {turn["turn"]: turn for turn in task["turns"]}
        units = gold_units(gold_row)
        if not units:
            continue
        annotator = gold_row.get("annotator")
        if isinstance(annotator, str) and annotator:
            annotators.add(annotator)
        prediction_row = prediction_rows.get(dialogue_id, {"output": {"constraints": []}})
        pairs.append({
            "dialogue_id": dialogue_id,
            "gold": units,
            "predictions": prediction_units(prediction_row),
            "turns": turns,
            "fold": fold_of(dialogue_id, args.folds),
        })

    if not pairs:
        print("no dialogue with human-labelled units was found")
        return 2

    grid = sweep(pairs, args.text_grid, args.anchor_grid)
    chosen, rule, degraded = choose_threshold(grid, args.min_precision)
    _, summary = cross_validate(pairs, args.text_grid, args.anchor_grid, args.min_precision, args.folds)

    report = {
        "gold_dialogue_n": len(pairs),
        "gold_unit_n": sum(pair["gold"].__len__() for pair in pairs),
        "annotators": sorted(annotators),
        "selection_rule": rule,
        "degraded": degraded,
        "in_sample_chosen": chosen,
        "out_of_fold": summary["out_of_fold"],
        "grid": grid,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write:
        if not args.adjudicator:
            print(
                "refusing to write a calibrated config without --adjudicator: "
                "a third-party adjudicator must exist before a threshold may be "
                "described as calibrated"
            )
            return 2
        if summary["degraded_any_fold"]:
            print(
                "refusing to write: no threshold reached the precision floor in at "
                "least one fold — the calibration set is too small or too noisy"
            )
            return 2
        config = build_config(
            gold_path=args.gold,
            chosen=chosen,
            summary=summary,
            annotators=sorted(annotators),
            adjudicator=args.adjudicator,
            folds=args.folds,
            min_precision=args.min_precision,
        )
        write_config(args.write, config)
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
