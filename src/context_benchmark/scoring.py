"""Deterministic A/C scoring with explicit numerators and denominators.

The evidence/text matcher is an *uncalibrated engineering baseline*, not a
validated semantic judge. P0 must calibrate or replace it before any model claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .ontology import ACTIVE_TYPES, TYPE_TO_CODE
from .validation import public_constraint, validate_output

MATCHER_VERSION = "evidence_char_ngram_greedy_v0_uncalibrated"


def ratio(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (numerator / denominator) if denominator else None,
    }


def _normal(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def _ngrams(text: str, n: int = 2) -> set[str]:
    normal = _normal(text)
    if not normal:
        return set()
    if len(normal) <= n:
        return {normal}
    return {normal[index:index + n] for index in range(len(normal) - n + 1)}


def _dice(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def unit_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    text = _dice(_ngrams(left.get("text", "")), _ngrams(right.get("text", "")))
    evidence = _dice(set(left.get("evidence_turns", [])), set(right.get("evidence_turns", [])))
    return 0.55 * text + 0.45 * evidence


def greedy_match(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float = 0.32
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = []
    for gi, gold_item in enumerate(gold):
        for pi, prediction in enumerate(predictions):
            score = unit_similarity(gold_item, prediction)
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_gold: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for score, gi, pi in candidates:
        if gi not in used_gold and pi not in used_predictions:
            used_gold.add(gi)
            used_predictions.add(pi)
            matches.append((gi, pi, score))
    unmatched_gold = [index for index in range(len(gold)) if index not in used_gold]
    unmatched_predictions = [index for index in range(len(predictions)) if index not in used_predictions]
    return matches, unmatched_gold, unmatched_predictions


def _prf(tp: int, predicted_n: int, gold_n: int) -> dict[str, Any]:
    fp = predicted_n - tp
    fn = gold_n - tp
    precision = tp / predicted_n if predicted_n else None
    recall = tp / gold_n if gold_n else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if predicted_n and gold_n else None)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": ratio(tp, predicted_n),
        "recall": ratio(tp, gold_n),
        "f1": {"value": f1, "predicted_n": predicted_n, "gold_n": gold_n},
    }


def score_dialogue(dialogue: dict[str, Any], prediction_row: dict[str, Any]) -> dict[str, Any]:
    gold = dialogue["candidate_truth"]
    output = prediction_row.get("output", {"constraints": prediction_row.get("constraints", [])})
    validation = validate_output(output, dialogue["turns"])
    raw_predictions = output.get("constraints", []) if isinstance(output, dict) else []
    if not isinstance(raw_predictions, list):
        raw_predictions = []
    # Invalid JSON/schema/semantics fail this dialogue's A/C metrics. Raw output
    # remains visible in validation and unmatched-prediction accounting.
    predictions = raw_predictions if validation.valid else []
    matches, unmatched_gold, unmatched_predictions = greedy_match(gold, predictions)
    if not validation.valid:
        unmatched_predictions = list(range(len(raw_predictions)))
    match_by_gold = {gi: (pi, score) for gi, pi, score in matches}

    presence_gold = [index for index, item in enumerate(gold) if item["type"] in ACTIVE_TYPES]
    presence_correct = 0
    forget_violations = 0
    forget_gold_n = 0
    expired_reuse = 0
    superseded_total = 0
    superseded_correct_type = 0
    supersession_links_total = 0
    supersession_links_correct = 0
    tp_by_type = Counter()
    pred_by_type = Counter(item.get("type") for item in predictions)
    gold_by_type = Counter(item["type"] for item in gold)

    for gi, item in enumerate(gold):
        pi_score = match_by_gold.get(gi)
        matched_prediction = predictions[pi_score[0]] if pi_score else None
        correct_type = bool(matched_prediction and matched_prediction.get("type") == item["type"])
        correct_scope = bool(matched_prediction and matched_prediction.get("scope") == item["scope"])
        if correct_type:
            tp_by_type[item["type"]] += 1
        if item["type"] in ACTIVE_TYPES and correct_type and correct_scope:
            presence_correct += 1
        if item["type"] in {"one_off", "superseded"}:
            forget_gold_n += 1
            if matched_prediction and matched_prediction.get("type") in ACTIVE_TYPES:
                forget_violations += 1
        if item["type"] == "superseded":
            superseded_total += 1
            if correct_type:
                superseded_correct_type += 1
            if matched_prediction and matched_prediction.get("type") in ACTIVE_TYPES:
                expired_reuse += 1
            if item.get("superseded_by"):
                supersession_links_total += 1
                if matched_prediction and matched_prediction.get("superseded_by") == item["superseded_by"]:
                    supersession_links_correct += 1

    distractor_matches, _, _ = greedy_match(dialogue.get("distractors", []), raw_predictions, threshold=0.35)
    distractor_n = len(dialogue.get("distractors", []))
    distractor_violations = len({gi for gi, _, _ in distractor_matches})
    forget_n = forget_gold_n + distractor_n
    forget_violations += distractor_violations
    mpa = ratio(presence_correct, len(presence_gold))
    faa = ratio(forget_n - forget_violations, forget_n)
    lam = forget_n / (len(presence_gold) + forget_n) if len(presence_gold) + forget_n else 0.0
    fama = None
    if mpa["value"] is not None or faa["value"] is not None:
        fama = max(0.0, (mpa["value"] or 0.0) - lam * (1.0 - (faa["value"] or 0.0)))

    per_type = {
        ctype: _prf(tp_by_type[ctype], pred_by_type[ctype], gold_by_type[ctype])
        for ctype in TYPE_TO_CODE
    }
    correct_type_n = sum(tp_by_type.values())
    return {
        "dialogue_id": dialogue["dialogue_id"],
        "valid_output": validation.valid,
        "validation": validation.to_dict(),
        "matcher_version": MATCHER_VERSION,
        "match_n": len(matches),
        "unmatched_gold_n": len(unmatched_gold),
        "unmatched_prediction_n": len(unmatched_predictions),
        "predicted_n": len(predictions),
        "raw_predicted_n": len(raw_predictions),
        "gold_n": len(gold),
        "must_include": mpa,
        "must_exclude_absence": faa,
        "fama": {"value": fama, "lambda": lam, "dialogue_n": 1 if fama is not None else 0},
        "per_type": per_type,
        "classification": _prf(correct_type_n, len(predictions), len(gold)),
        "overgeneralization": ratio(forget_violations - distractor_violations, forget_gold_n),
        "fmr": ratio(distractor_n - distractor_violations, distractor_n),
        "unmatched_prediction_rate": ratio(len(unmatched_predictions), len(raw_predictions)),
        "empty_output": {
            "is_empty_dialogue": not gold,
            "correct": bool(not gold and not predictions and validation.valid),
            "n": 1 if not gold else 0,
        },
        "dimension_c": {
            "superseded_type_accuracy": ratio(superseded_correct_type, superseded_total),
            "expired_reuse_rate": ratio(expired_reuse, superseded_total),
            "supersession_link_accuracy": ratio(supersession_links_correct, supersession_links_total),
        },
    }


def _sum_ratio(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return ratio(sum(row["numerator"] for row in materialized), sum(row["denominator"] for row in materialized))


def score_dataset(dialogues: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {row["dialogue_id"]: row for row in prediction_rows}
    missing = [dialogue["dialogue_id"] for dialogue in dialogues if dialogue["dialogue_id"] not in predictions]
    rows = []
    for dialogue in dialogues:
        row = predictions.get(dialogue["dialogue_id"], {"dialogue_id": dialogue["dialogue_id"], "output": None})
        rows.append(score_dialogue(dialogue, row))
    aggregate: dict[str, Any] = {
        "dialogue_n": len(rows),
        "missing_prediction_dialogues": missing,
        "invalid_output_rate": ratio(sum(not row["valid_output"] for row in rows), len(rows)),
        "must_include": _sum_ratio(row["must_include"] for row in rows),
        "must_exclude_absence": _sum_ratio(row["must_exclude_absence"] for row in rows),
        "overgeneralization": _sum_ratio(row["overgeneralization"] for row in rows),
        "fmr": _sum_ratio(row["fmr"] for row in rows),
        "unmatched_prediction_rate": _sum_ratio(row["unmatched_prediction_rate"] for row in rows),
        "empty_output_accuracy": ratio(sum(row["empty_output"]["correct"] for row in rows), sum(row["empty_output"]["n"] for row in rows)),
        "dimension_c": {
            key: _sum_ratio(row["dimension_c"][key] for row in rows)
            for key in ("superseded_type_accuracy", "expired_reuse_rate", "supersession_link_accuracy")
        },
        "per_type": {},
    }
    fama_values = [row["fama"]["value"] for row in rows if row["fama"]["value"] is not None]
    aggregate["fama_macro"] = ratio(sum(fama_values), len(fama_values))
    for ctype in TYPE_TO_CODE:
        tp = sum(row["per_type"][ctype]["tp"] for row in rows)
        predicted_n = sum(row["per_type"][ctype]["precision"]["denominator"] for row in rows)
        gold_n = sum(row["per_type"][ctype]["recall"]["denominator"] for row in rows)
        aggregate["per_type"][ctype] = _prf(tp, predicted_n, gold_n)
    return {
        "metric_version": "a-c-v0",
        "matcher_version": MATCHER_VERSION,
        "matcher_calibrated": False,
        "scientific_status": "engineering_baseline_not_a_model_result",
        "per_dialogue": rows,
        "aggregate": aggregate,
    }


def judge_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Two-class gold calibration plus unextractable rate (SEQUOR-style)."""
    gold_yes = [row for row in rows if row["gold"] == "yes"]
    gold_no = [row for row in rows if row["gold"] == "no"]
    errors_yes = sum(row.get("predicted") != "yes" for row in gold_yes)
    errors_no = sum(row.get("predicted") != "no" for row in gold_no)
    unextractable = sum(row.get("predicted") not in {"yes", "no"} for row in rows)
    paired = [row for row in rows if row.get("predicted") in {"yes", "no"}]
    from .agreement import cohen_kappa
    human_machine = cohen_kappa([row["gold"] for row in paired], [row["predicted"] for row in paired])
    return {
        "gold_yes_error_rate": ratio(errors_yes, len(gold_yes)),
        "gold_no_error_rate": ratio(errors_no, len(gold_no)),
        "overall_error_rate": ratio(errors_yes + errors_no, len(rows)),
        "unextractable_rate": ratio(unextractable, len(rows)),
        "human_machine_kappa_scorable_only": human_machine,
        "n": len(rows),
        "pass_kappa_0_60": human_machine["kappa"] is not None and human_machine["kappa"] >= 0.60,
        "warning": "Overall accuracy must not hide asymmetric yes/no errors; unextractable rows are excluded only from kappa and remain in their own denominator.",
    }


def score_counterfactual(probe_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score executor choices for true/none/fake arms.

    Each row needs probe_id, arm, compliant (0/1), and related (bool). The fake
    arm is a plausibility control; a positive fake delta is a validity warning.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in probe_results:
        grouped[row["probe_id"]][row["arm"]].append(row)
    deltas = []
    fake_deltas = []
    negative_transfer = []
    false_trigger = []
    for probe_id, arms in grouped.items():
        if not {"true", "none", "fake"} <= set(arms):
            continue
        means = {
            arm: sum(int(item["compliant"]) for item in items) / len(items)
            for arm, items in arms.items()
        }
        deltas.append(means["true"] - means["none"])
        fake_deltas.append(means["fake"] - means["none"])
        negative_transfer.append(int(means["true"] < means["none"]))
        related = arms["true"][0].get("related", True)
        false_trigger.append(int(not related and means["true"] < means["none"]))
    return {
        "probe_n": len(deltas),
        "true_delta_mean": ratio(sum(deltas), len(deltas)),
        "fake_delta_mean": ratio(sum(fake_deltas), len(fake_deltas)),
        "negative_transfer_rate": ratio(sum(negative_transfer), len(negative_transfer)),
        "false_trigger_rate": ratio(sum(false_trigger), len(false_trigger)),
        "validity_warning_fake_positive": bool(fake_deltas and sum(fake_deltas) / len(fake_deltas) > 0),
        "denominators": {"true_delta": len(deltas), "fake_delta": len(fake_deltas)},
    }
