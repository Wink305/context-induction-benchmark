"""Deterministic A/C scoring with explicit numerators and denominators.

The evidence/text matcher is an *uncalibrated engineering baseline*, not a
validated semantic judge. P0 must calibrate or replace it before any model claim.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .ontology import ACTIVE_TYPES, TYPE_TO_CODE
from .semantic import extract_slot_values, semantic_conflict
from .validation import public_constraint, validate_output

MATCHER_VERSION = "evidence_char_ngram_greedy_v3_text_gated_semantic_guard"
# These gates are UNCALIBRATED engineering heuristics. They were chosen so that
# a hand-written short paraphrase of a constructed unit survives while pure
# garbage (unrelated text with perfect evidence) does not. No human annotation
# exists behind either number, and the pilot candidate truth is not gold, so
# these thresholds must not be described as "calibrated on human prechecks".
# P1 must fit them on an independent, human-labelled calibration set and report
# sensitivity before any model score is reported. See design doc §5 and §11.2.
TEXT_MATCH_MIN_DEFAULT = 0.10
# The predicted text must also be recoverable from the user turns it cites.
# Coverage = |bigrams(text) ∩ bigrams(cited turns)| / |bigrams(text)|.
ANCHOR_COVERAGE_MIN_DEFAULT = 0.08


def _load_thresholds() -> dict[str, Any]:
    """Return the effective gates and where they came from.

    A calibration nobody reads is a decoration. When
    ``configs/calibration.json`` exists and declares ``state: "calibrated"``,
    its numbers are the ones in force; otherwise the built-in defaults are.
    The file is read once at import time and the result is reported in every
    score so a score can never be read without knowing which thresholds
    produced it.
    """
    defaults = {
        "text_min": TEXT_MATCH_MIN_DEFAULT,
        "anchor_coverage_min": ANCHOR_COVERAGE_MIN_DEFAULT,
        "evidence_iou_min": None,
        "slot_match_required": False,
    }
    path = Path(__file__).resolve().parents[2] / "configs" / "calibration.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**defaults, "state": "unavailable", "source": "built_in_default"}
    state = config.get("state", "uncalibrated")
    if state != "calibrated":
        return {**defaults, "state": state, "source": "built_in_default"}
    thresholds = config.get("thresholds") or {}
    try:
        text_min = float(thresholds["text_min"])
        anchor_min = float(thresholds["anchor_coverage_min"])
    except (KeyError, TypeError, ValueError):
        return {
            **defaults, "state": state, "source": "built_in_default",
            "warning": "the calibration config declares 'calibrated' but has no usable thresholds",
        }
    evidence_iou_min = thresholds.get("evidence_iou_min")
    if evidence_iou_min is not None:
        try:
            evidence_iou_min = float(evidence_iou_min)
        except (TypeError, ValueError):
            evidence_iou_min = None
    return {
        "text_min": text_min,
        "anchor_coverage_min": anchor_min,
        "evidence_iou_min": evidence_iou_min,
        "slot_match_required": bool(thresholds.get("slot_match_required", False)),
        "state": state,
        "source": "configs/calibration.json",
    }


_THRESHOLDS = _load_thresholds()
TEXT_MATCH_MIN = _THRESHOLDS["text_min"]
ANCHOR_COVERAGE_MIN = _THRESHOLDS["anchor_coverage_min"]
EVIDENCE_IOU_MIN = _THRESHOLDS["evidence_iou_min"]
SLOT_MATCH_REQUIRED = _THRESHOLDS["slot_match_required"]
CONTENT_TEXT_MIN = TEXT_MATCH_MIN
THRESHOLD_SOURCE = {"state": _THRESHOLDS["state"], "source": _THRESHOLDS["source"]}


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


def anchor_coverage(text: str, turns: dict[int, dict[str, Any]], evidence: list[int]) -> float:
    """Fraction of the predicted text's bigrams that appear in its cited turns."""
    joined = " ".join(str(turns[turn]["text"]) for turn in evidence if turn in turns)
    predicted_ngrams = _ngrams(text)
    if not predicted_ngrams:
        return 0.0
    return len(predicted_ngrams & _ngrams(joined)) / len(predicted_ngrams)


def unit_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    text = _dice(_ngrams(left.get("text", "")), _ngrams(right.get("text", "")))
    evidence = _dice(set(left.get("evidence_turns", [])), set(right.get("evidence_turns", [])))
    return 0.55 * text + 0.45 * evidence


def greedy_match(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float = 0.32,
    *, turns: dict[int, dict[str, Any]] | None = None,
    text_min: float | None = None, anchor_min: float | None = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Match predictions against gold units.

    ``text_min`` and ``anchor_min`` default to the module-level gates and are
    overridable so that ``scripts/fit_thresholds.py`` can evaluate a grid of
    thresholds without mutating the process-wide constants. The default
    behaviour is unchanged.
    """
    effective_text_min = TEXT_MATCH_MIN if text_min is None else text_min
    effective_anchor_min = ANCHOR_COVERAGE_MIN if anchor_min is None else anchor_min
    candidates = []
    for gi, gold_item in enumerate(gold):
        for pi, prediction in enumerate(predictions):
            text_sim = _dice(_ngrams(gold_item.get("text", "")), _ngrams(prediction.get("text", "")))
            # Hard text gate: a perfect evidence overlap must not be enough to
            # match a prediction whose text has nothing to do with the gold.
            if text_sim < effective_text_min:
                continue
            if turns is not None and anchor_coverage(
                prediction.get("text", ""), turns, prediction.get("evidence_turns", [])
            ) < effective_anchor_min:
                continue
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
    turn_map = {turn["turn"]: turn for turn in dialogue["turns"]}
    # Invalid JSON/schema/semantics fail this dialogue's A/C metrics. Raw output
    # remains visible in validation and unmatched-prediction accounting.
    predictions = raw_predictions if validation.valid else []
    matches, unmatched_gold, unmatched_predictions = greedy_match(gold, predictions, turns=turn_map)
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
    content_ok_n = 0
    semantic_conflict_n = 0
    pred_by_type = Counter(item.get("type") for item in predictions)
    gold_by_type = Counter(item["type"] for item in gold)

    for gi, item in enumerate(gold):
        pi_score = match_by_gold.get(gi)
        matched_prediction = predictions[pi_score[0]] if pi_score else None
        content_ok = False
        if matched_prediction:
            text_sim = _dice(_ngrams(item["text"]), _ngrams(matched_prediction.get("text", "")))
            # Semantic guard: a prediction that quotes the gold sentence and
            # then tells the reader not to follow it ("不要遵守本条", "改为完全
            # 相反的操作"), or that names the opposite value of the same slot
            # ("只用邮件" where the user said "只用短信"), is not a correct
            # constraint no matter how much text it shares.
            conflict = semantic_conflict(item["text"], matched_prediction.get("text", ""))
            content_ok = (
                text_sim >= CONTENT_TEXT_MIN
                and anchor_coverage(
                    matched_prediction.get("text", ""), turn_map,
                    matched_prediction.get("evidence_turns", []),
                ) >= ANCHOR_COVERAGE_MIN
                and not conflict["conflicted"]
            )
            if conflict["conflicted"]:
                semantic_conflict_n += 1
        content_ok_n += int(content_ok)
        correct_type = bool(matched_prediction and content_ok and matched_prediction.get("type") == item["type"])
        correct_scope = bool(matched_prediction and content_ok and matched_prediction.get("scope") == item["scope"])
        if correct_type:
            tp_by_type[item["type"]] += 1
        if item["type"] in ACTIVE_TYPES and correct_type and correct_scope:
            presence_correct += 1
        if item["type"] in {"one_off", "superseded"}:
            forget_gold_n += 1
            if matched_prediction and content_ok and matched_prediction.get("type") in ACTIVE_TYPES:
                forget_violations += 1
        if item["type"] == "superseded":
            superseded_total += 1
            if correct_type:
                superseded_correct_type += 1
            if matched_prediction and content_ok and matched_prediction.get("type") in ACTIVE_TYPES:
                expired_reuse += 1
            if item.get("superseded_by"):
                supersession_links_total += 1
                if matched_prediction and matched_prediction.get("superseded_by") == item["superseded_by"]:
                    supersession_links_correct += 1

    # An invalid output fails A/C uniformly. Both the gold-side forget set and
    # the distractor side must use the same (empty) prediction list; otherwise
    # invalid output zeroed forget violations but still counted distractor
    # violations, producing opposite effects in FAA and overgeneralization.
    valid_predictions = raw_predictions if validation.valid else []
    distractor_matches, _, _ = greedy_match(
        dialogue.get("distractors", []), valid_predictions, threshold=0.35
    )
    distractor_n = len(dialogue.get("distractors", []))
    distractor_violations = len({gi for gi, _, _ in distractor_matches})
    forget_n = forget_gold_n + distractor_n
    forget_violations += distractor_violations
    mpa = ratio(presence_correct, len(presence_gold))
    faa = ratio(forget_n - forget_violations, forget_n) if validation.valid else ratio(0, forget_n)
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
        "content_gate": {
            "text_min": CONTENT_TEXT_MIN,
            "anchor_coverage_min": ANCHOR_COVERAGE_MIN,
            "content_ok_n": content_ok_n,
            "semantic_conflict_n": semantic_conflict_n,
            "semantic_guard": "slot_opposition_and_meta_override_v1",
            "semantic_guard_calibrated": False,
        },
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
    # Fixed denominator: every dialogue counts, including those whose mpa/faa
    # are undefined (empty dialogues). Undefined FAMA is 0 for the macro mean —
    # a model that produces no scorable output cannot improve the average by
    # disappearing from the denominator. The excluded_n is disclosed below.
    fama_values = [row["fama"]["value"] if row["fama"]["value"] is not None else 0.0 for row in rows]
    fama_undefined_n = sum(row["fama"]["value"] is None for row in rows)
    aggregate["fama_macro"] = ratio(sum(fama_values), len(rows))
    aggregate["fama_macro"]["undefined_scored_as_zero_n"] = fama_undefined_n
    aggregate["fama_macro"]["denominator_policy"] = "all_dialogues"
    for ctype in TYPE_TO_CODE:
        tp = sum(row["per_type"][ctype]["tp"] for row in rows)
        predicted_n = sum(row["per_type"][ctype]["precision"]["denominator"] for row in rows)
        gold_n = sum(row["per_type"][ctype]["recall"]["denominator"] for row in rows)
        aggregate["per_type"][ctype] = _prf(tp, predicted_n, gold_n)
    return {
        "metric_version": "a-c-v1-text-gated",
        "matcher_version": MATCHER_VERSION,
        "matcher_calibrated": False,
        "scoring_thresholds": {
            "text_min": TEXT_MATCH_MIN,
            "anchor_coverage_min": ANCHOR_COVERAGE_MIN,
            "source": THRESHOLD_SOURCE.get("source"),
            "calibration_state": THRESHOLD_SOURCE.get("state"),
        },
        "capabilities": score_capabilities(dialogues, prediction_rows),
        "scientific_status": "engineering_baseline_not_a_model_result",
        "per_dialogue": rows,
        "aggregate": aggregate,
    }


def judge_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Two-class gold calibration plus unextractable rate (SEQUOR-style).

    ``unextractable`` is a third label, not a missing value. The gate kappa
    includes it, because a judge that abstains on the hardest rows must pay for
    those abstentions; otherwise "I can't tell" is rewarded with a higher score.
    A scorable-only kappa is still reported for comparison.
    """
    gold_yes = [row for row in rows if row["gold"] == "yes"]
    gold_no = [row for row in rows if row["gold"] == "no"]
    errors_yes = sum(row.get("predicted") != "yes" for row in gold_yes)
    errors_no = sum(row.get("predicted") != "no" for row in gold_no)
    unextractable = sum(row.get("predicted") not in {"yes", "no"} for row in rows)
    from .agreement import cohen_kappa

    all_gold = [row["gold"] for row in rows]
    all_pred = [row.get("predicted", "unextractable") for row in rows]
    human_machine = cohen_kappa(all_gold, all_pred)
    paired = [row for row in rows if row.get("predicted") in {"yes", "no"}]
    scorable_only = cohen_kappa([row["gold"] for row in paired], [row["predicted"] for row in paired])
    return {
        "gold_yes_error_rate": ratio(errors_yes, len(gold_yes)),
        "gold_no_error_rate": ratio(errors_no, len(gold_no)),
        "overall_error_rate": ratio(errors_yes + errors_no, len(rows)),
        "unextractable_rate": ratio(unextractable, len(rows)),
        "human_machine_kappa_includes_unextractable": human_machine,
        "human_machine_kappa_scorable_only": scorable_only,
        "n": len(rows),
        "pass_kappa_0_60": human_machine["kappa"] is not None and human_machine["kappa"] >= 0.60,
        "warning": (
            "The gate uses kappa over all rows with unextractable as a third "
            "label; scorable-only kappa is reported for comparison but never "
            "used to pass the gate. Overall accuracy must not hide asymmetric "
            "yes/no errors."
        ),
    }


def score_counterfactual(probe_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score executor choices for true/none/fake arms.

    Each row needs probe_id, arm and compliant (0/1). ``related`` marks whether
    the injection is relevant to the new situation. A false trigger is the
    model following an *unrelated* constraint (true > none); negative transfer
    on a related probe is a different failure and is reported separately.
    Probes without an explicit ``related`` flag are excluded from this metric
    rather than silently counted as related.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in probe_results:
        grouped[row["probe_id"]][row["arm"]].append(row)
    deltas = []
    fake_deltas = []
    negative_transfer = []
    false_trigger = []
    unclassified_related_n = 0
    unscorable_probes_n = 0
    semantics_seen: set[str] = set()
    for probe_id, arms in grouped.items():
        if not {"true", "none", "fake"} <= set(arms):
            continue
        # Only provider-scorable rows may enter a mean. An unscorable row
        # (missing induced target, API error, invalid choice) is NOT a wrong
        # answer; counting compliant=0 for it fabricated failure data points.
        scorable_arms = {
            arm: [item for item in items if item.get("scorable", True)]
            for arm, items in arms.items()
        }
        if any(not rows for rows in scorable_arms.values()):
            unscorable_probes_n += 1
            continue
        means = {
            arm: sum(int(item["compliant"]) for item in items) / len(items)
            for arm, items in scorable_arms.items()
        }
        deltas.append(means["true"] - means["none"])
        fake_deltas.append(means["fake"] - means["none"])
        negative_transfer.append(int(means["true"] < means["none"]))
        related = arms["true"][0].get("related")
        # What does compliant=1 mean for this probe? Two conventions exist and
        # they invert the false-trigger direction:
        #   follows_injection — compliant=1 means the behaviour applied the
        #       injected rule; a false trigger is then true > none.
        #   task_correct — compliant=1 means the behaviour was correct for the
        #       task; applying an irrelevant rule makes it wrong, so a false
        #       trigger is then true < none.
        # Mixing the two silently produced a false-trigger rate of exactly 0
        # in one direction. The convention is read from the probe row.
        semantics = arms["true"][0].get("compliant_semantics") or "follows_injection"
        semantics_seen.add(semantics)
        if related is False:
            if semantics == "task_correct":
                false_trigger.append(int(means["true"] < means["none"]))
            else:
                false_trigger.append(int(means["true"] > means["none"]))
        elif related is None:
            unclassified_related_n += 1
    return {
        "probe_n": len(deltas),
        "true_delta_mean": ratio(sum(deltas), len(deltas)),
        "fake_delta_mean": ratio(sum(fake_deltas), len(fake_deltas)),
        "negative_transfer_rate": ratio(sum(negative_transfer), len(negative_transfer)),
        "false_trigger_rate": ratio(sum(false_trigger), len(false_trigger)),
        "validity_warning_fake_positive": bool(fake_deltas and sum(fake_deltas) / len(fake_deltas) > 0),
        "compliant_semantics_seen": sorted(semantics_seen),
        "denominators": {
            "true_delta": len(deltas),
            "fake_delta": len(fake_deltas),
            "false_trigger": len(false_trigger),
            "unclassified_related": unclassified_related_n,
            "unscorable_probes_excluded": unscorable_probes_n,
        },
    }


# ===========================================================================
# Capability layer: retention / pruning / trap marking.
#
# The A/C scorer above answers "does this prediction look like the gold
# constraint?". These three answer three different questions, each with its
# own denominator and its own failure mode:
#
#   retention — were the durable constraints kept *whole*, including the ones
#               buried at the end of a long dialogue?
#   pruning   — was the stuff that must not survive actually removed, without
#               the model emptying its memory to score well on it?
#   traps     — did the model notice the planted traps, name them, and still
#               handle the affected constraints correctly?
#
# They are reported as a profile and never folded into one number. Every
# threshold comes from configs/calibration.json; while that file is
# "uncalibrated" the two gates that were always in force apply and the two
# P1 gates stay switched off, so this layer cannot quietly become stricter
# than the scorer it sits on top of.
# ===========================================================================

CAPABILITY_SCORER_VERSION = "capabilities-v1-uncalibrated"
DEPTH_BUCKETS = ("early", "mid", "late")
PRUNABLE_TYPES = {"one_off", "superseded"}


def evidence_iou(gold_turns: Iterable[int], predicted_turns: Iterable[int]) -> float:
    """Intersection over union of the two evidence sets.

    The A/C scorer never checks that a prediction points at the *right*
    turns: a prediction that copies the gold sentence but cites an unrelated
    turn still scores full marks. This is the number that exposes that.
    """
    gold = set(gold_turns)
    predicted = set(predicted_turns)
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    return len(gold & predicted) / len(gold | predicted)


def _depth_bucket(turn: int, total_turns: int) -> str:
    """Where in the dialogue a constraint's last evidence sits."""
    if total_turns <= 0:
        return "early"
    position = turn / total_turns
    if position < 1 / 3:
        return "early"
    if position < 2 / 3:
        return "mid"
    return "late"


def _slot_values_agree(gold_text: str, predicted_text: str) -> tuple[bool, bool]:
    """Do the controlled slot values agree, and is the check applicable?

    Returns ``(agrees, applicable)``. A gold sentence with no controlled
    slot (nothing to compare) is not a failure; it is simply not applicable.
    """
    gold_slots = extract_slot_values(gold_text)
    if not gold_slots:
        return True, False
    predicted_slots = extract_slot_values(predicted_text)
    for family, values in gold_slots.items():
        if not (values & predicted_slots.get(family, set())):
            return False, True
    return True, True


def _content_verdict(
    gold_item: dict[str, Any],
    prediction: dict[str, Any],
    turns: dict[int, dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """One (gold, prediction) pair, judged against every content gate."""
    text_sim = _dice(_ngrams(gold_item.get("text", "")), _ngrams(prediction.get("text", "")))
    anchor = anchor_coverage(
        prediction.get("text", ""), turns, prediction.get("evidence_turns", [])
    )
    iou = evidence_iou(
        gold_item.get("evidence_turns", []) or [], prediction.get("evidence_turns", []) or []
    )
    conflict = semantic_conflict(gold_item.get("text", ""), prediction.get("text", ""))
    slot_ok, slot_applicable = _slot_values_agree(
        gold_item.get("text", ""), prediction.get("text", "")
    )
    content_ok = (
        text_sim >= thresholds["text_min"]
        and anchor >= thresholds["anchor_coverage_min"]
        and not conflict["conflicted"]
        and slot_ok
    )
    if thresholds.get("evidence_iou_min") is not None:
        content_ok = content_ok and iou >= thresholds["evidence_iou_min"]
    return {
        "text_sim": text_sim,
        "anchor_coverage": anchor,
        "evidence_iou": iou,
        "conflict": conflict,
        "slot_ok": slot_ok,
        "slot_applicable": slot_applicable,
        "content_ok": content_ok,
    }


def _capability_match(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    turns: dict[int, dict[str, Any]],
    thresholds: dict[str, Any],
) -> list[tuple[int, int, float]]:
    """One-to-one matching that applies the content gates and nothing else.

    The A/C matcher also demands a combined similarity of 0.32, which is a
    statement about *which* unit is which. The capability layer asks a
    different question — "is this constraint present at all?" — so it
    matches on the gates alone and lets the content verdict decide.
    """
    pairs: list[tuple[float, int, int]] = []
    for gi, gold_item in enumerate(gold):
        for pi, prediction in enumerate(predictions):
            text_sim = _dice(
                _ngrams(gold_item.get("text", "")), _ngrams(prediction.get("text", ""))
            )
            if text_sim < thresholds["text_min"]:
                continue
            anchor = anchor_coverage(
                prediction.get("text", ""), turns, prediction.get("evidence_turns", [])
            )
            if anchor < thresholds["anchor_coverage_min"]:
                continue
            iou = evidence_iou(
                gold_item.get("evidence_turns", []) or [],
                prediction.get("evidence_turns", []) or [],
            )
            pairs.append((text_sim + 0.25 * iou, gi, pi))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_gold: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gi, pi in pairs:
        if gi in used_gold or pi in used_predictions:
            continue
        used_gold.add(gi)
        used_predictions.add(pi)
        matches.append((gi, pi, score))
    return matches


def _prediction_output(prediction_row: dict[str, Any]) -> dict[str, Any]:
    output = prediction_row.get("output")
    if isinstance(output, dict):
        return output
    return {"constraints": prediction_row.get("constraints", [])}


def _default_error_kind(item: dict[str, Any]) -> str:
    kind = item.get("error_kind")
    if isinstance(kind, str) and kind:
        return kind
    if item.get("type") == "one_off":
        return "one_off_as_standing"
    if item.get("type") == "superseded":
        return "expired_reuse"
    return "fabricated"


def score_retention(
    dialogue: dict[str, Any],
    prediction_row: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Were the durable constraints kept — every one of them, and intact?

    Three numbers, because they fail differently:

    * ``unit_recall`` — the average share of durable constraints kept;
    * ``full_set_exact`` — whether *every* durable constraint in this
      dialogue survived. A model that keeps nine out of ten looks fine on
      the average and terrible here;
    * ``retention_by_depth`` — whether the constraints that sat at the end of
      a long dialogue were kept as well as the ones at the start.
    """
    gates = thresholds or _THRESHOLDS
    gold = dialogue.get("candidate_truth", []) or []
    turns_list = dialogue.get("turns", []) or []
    turn_map = {turn["turn"]: turn for turn in turns_list}
    output = _prediction_output(prediction_row)
    validation = validate_output(output, turns_list)
    predictions = output.get("constraints", []) if validation.valid else []

    matches = _capability_match(gold, predictions, turn_map, gates)
    match_by_gold = {gi: pi for gi, pi, _ in matches}
    active_indices = [
        index for index, item in enumerate(gold) if item.get("type") in ACTIVE_TYPES
    ]
    depth = {bucket: {"gold_n": 0, "retained_n": 0} for bucket in DEPTH_BUCKETS}
    retained_n = 0
    ious: list[float] = []
    slot_applicable_n = 0
    slot_ok_n = 0

    for gi in active_indices:
        item = gold[gi]
        last_turn = max(item.get("evidence_turns") or [0])
        bucket = _depth_bucket(last_turn, len(turn_map))
        depth[bucket]["gold_n"] += 1
        pi = match_by_gold.get(gi)
        if pi is None:
            continue
        prediction = predictions[pi]
        verdict = _content_verdict(item, prediction, turn_map, gates)
        ious.append(verdict["evidence_iou"])
        if verdict["slot_applicable"]:
            slot_applicable_n += 1
            slot_ok_n += int(verdict["slot_ok"])
        if (
            verdict["content_ok"]
            and prediction.get("type") == item.get("type")
            and prediction.get("scope") == item.get("scope")
        ):
            retained_n += 1
            depth[bucket]["retained_n"] += 1

    active_n = len(active_indices)
    return {
        "dialogue_id": dialogue.get("dialogue_id"),
        "valid_output": validation.valid,
        "active_gold_n": active_n,
        "retained_n": retained_n,
        "unit_recall": ratio(retained_n, active_n),
        "full_set_exact": 1 if (active_n and retained_n == active_n) else 0,
        "full_set_exact_n": 1 if active_n else 0,
        "retention_by_depth": {
            bucket: {
                "gold_n": depth[bucket]["gold_n"],
                "retained_n": depth[bucket]["retained_n"],
                "recall": ratio(depth[bucket]["retained_n"], depth[bucket]["gold_n"]),
            }
            for bucket in DEPTH_BUCKETS
        },
        "evidence_iou_mean": (sum(ious) / len(ious)) if ious else None,
        "slot_value_exact": ratio(slot_ok_n, slot_applicable_n),
    }


def _retained_as_active(
    item: dict[str, Any],
    predictions: list[dict[str, Any]],
    turn_map: dict[int, dict[str, Any]],
    gates: dict[str, Any],
) -> bool:
    """Did this non-durable item survive as an active constraint?"""
    for _gi, pi, _score in _capability_match([item], predictions, turn_map, gates):
        prediction = predictions[pi]
        verdict = _content_verdict(item, prediction, turn_map, gates)
        if verdict["content_ok"] and prediction.get("type") in ACTIVE_TYPES:
            return True
    return False


def score_pruning(
    dialogue: dict[str, Any],
    prediction_row: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Was the stuff that must not survive actually removed?

    ``prune_recall`` alone would be gameable: a model that outputs nothing
    prunes perfectly. ``over_prune_rate`` is therefore always reported next
    to it, and ``degenerate_all_delete`` names the failure outright.
    """
    gates = thresholds or _THRESHOLDS
    gold = dialogue.get("candidate_truth", []) or []
    distractors = dialogue.get("distractors", []) or []
    turns_list = dialogue.get("turns", []) or []
    turn_map = {turn["turn"]: turn for turn in turns_list}
    output = _prediction_output(prediction_row)
    validation = validate_output(output, turns_list)
    predictions = output.get("constraints", []) if validation.valid else []

    by_kind: dict[str, dict[str, int]] = {}
    wrongly_retained_n = 0

    for item in gold:
        if item.get("type") not in PRUNABLE_TYPES:
            continue
        kind = _default_error_kind(item)
        bucket = by_kind.setdefault(kind, {"n": 0, "wrongly_retained_n": 0})
        bucket["n"] += 1
        if _retained_as_active(item, predictions, turn_map, gates):
            bucket["wrongly_retained_n"] += 1
            wrongly_retained_n += 1

    distractor_matches = _capability_match(distractors, predictions, turn_map, gates)
    for gi, pi, _score in distractor_matches:
        distractor = distractors[gi]
        kind = _default_error_kind(distractor)
        bucket = by_kind.setdefault(kind, {"n": 0, "wrongly_retained_n": 0})
        bucket["n"] += 1
        verdict = _content_verdict(distractor, predictions[pi], turn_map, gates)
        if verdict["content_ok"] and predictions[pi].get("type") in ACTIVE_TYPES:
            bucket["wrongly_retained_n"] += 1
            wrongly_retained_n += 1

    prunable_n = sum(bucket["n"] for bucket in by_kind.values())
    retention = score_retention(dialogue, prediction_row, gates)
    if not validation.valid:
        # An output that does not satisfy the contract has not pruned
        # anything correctly and has not retained anything either. Leaving
        # these at zero would make a broken answer look like a model that
        # deleted exactly the right things.
        wrongly_retained_n = prunable_n
        over_pruned_n = retention["active_gold_n"]
        for bucket in by_kind.values():
            bucket["wrongly_retained_n"] = bucket["n"]
    else:
        over_pruned_n = retention["active_gold_n"] - retention["retained_n"]
    prune_recall = ratio(prunable_n - wrongly_retained_n, prunable_n)
    degenerate = bool(
        prune_recall["value"] is not None
        and prune_recall["value"] >= 0.9
        and retention["unit_recall"]["value"] is not None
        and retention["unit_recall"]["value"] <= 0.5
    )
    return {
        "dialogue_id": dialogue.get("dialogue_id"),
        "valid_output": validation.valid,
        "prunable_n": prunable_n,
        "wrongly_retained_n": wrongly_retained_n,
        "prune_recall": prune_recall,
        "prune_recall_by_kind": {
            kind: {
                "n": bucket["n"],
                "wrongly_retained_n": bucket["wrongly_retained_n"],
                "recall": ratio(bucket["n"] - bucket["wrongly_retained_n"], bucket["n"]),
            }
            for kind, bucket in sorted(by_kind.items())
        },
        "false_retain_rate": ratio(wrongly_retained_n, prunable_n),
        "over_prune_rate": ratio(over_pruned_n, retention["active_gold_n"]),
        "degenerate_all_delete": degenerate,
    }


def score_traps(
    dialogue: dict[str, Any],
    prediction_row: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Did the model notice the planted traps, name them, and handle them?

    Marking a trap is not the same as dealing with it, so both are scored:
    ``trap_flag_recall``/``trap_flag_precision`` for the naming, and
    ``trap_disposition_accuracy`` for the handling. A prediction without
    ``trap_flags`` is reported as **not evaluated** — never as zero, because
    "the model did not use the new field" and "the model failed to find the
    trap" are different facts.
    """
    gates = thresholds or _THRESHOLDS
    gold_traps = dialogue.get("traps") or []
    turns_list = dialogue.get("turns", []) or []
    turn_map = {turn["turn"]: turn for turn in turns_list}
    gold = dialogue.get("candidate_truth", []) or []
    output = _prediction_output(prediction_row)
    flags = output.get("trap_flags") or []

    base: dict[str, Any] = {
        "dialogue_id": dialogue.get("dialogue_id"),
        "trap_gold_n": len(gold_traps),
        "trap_flag_n": len(flags),
        "not_evaluated": False,
        "not_evaluated_reason": None,
    }
    if not gold_traps:
        base.update({
            "not_evaluated": True,
            "not_evaluated_reason": "the dialogue carries no trap gold",
            "trap_flag_recall": None,
            "trap_flag_precision": None,
            "trap_disposition_accuracy": None,
            "meta_override_resistance": None,
            "missed_and_followed_n": 0,
        })
        return base
    if not flags:
        base.update({
            "not_evaluated": True,
            "not_evaluated_reason": "the prediction carries no trap_flags",
            "trap_flag_recall": None,
            "trap_flag_precision": None,
            "trap_disposition_accuracy": None,
            "meta_override_resistance": None,
            "missed_and_followed_n": 0,
        })
        return base

    predictions_for_disposition = output.get("constraints", []) or []

    def _constraint_retained(constraint_id: str) -> bool:
        item = next((entry for entry in gold if entry.get("id") == constraint_id), None)
        if item is None:
            return False
        return _retained_as_active(item, predictions_for_disposition, turn_map, gates)

    matched_gold: dict[int, int] = {}
    true_positive_flags: set[int] = set()
    for gi, trap in enumerate(gold_traps):
        for fi, flag in enumerate(flags):
            if fi in true_positive_flags:
                continue
            if flag.get("kind") != trap.get("kind"):
                continue
            if not (
                set(flag.get("target_turns") or []) & set(trap.get("target_turns") or [])
            ):
                continue
            matched_gold[gi] = fi
            true_positive_flags.add(fi)
            break

    disposition_ok_n = 0
    missed_and_followed_n = 0
    for gi, trap in enumerate(gold_traps):
        affected = trap.get("affected_constraint_ids") or []
        disposition = trap.get("expected_disposition")
        handled = False
        if disposition == "retain_original":
            handled = bool(affected) and all(_constraint_retained(cid) for cid in affected)
        elif disposition == "drop":
            handled = all(not _constraint_retained(cid) for cid in affected)
        elif disposition == "mark_conflict":
            handled = any(
                flag.get("kind") == "conflicting_values"
                and flag.get("constraint_ref") in affected
                for flag in flags
            )
        if gi in matched_gold:
            if handled:
                disposition_ok_n += 1
        elif not handled:
            missed_and_followed_n += 1

    meta_traps = [trap for trap in gold_traps if trap.get("kind") == "meta_override"]
    meta_ok_n = 0
    for trap in meta_traps:
        affected = trap.get("affected_constraint_ids") or []
        if affected and all(_constraint_retained(cid) for cid in affected):
            meta_ok_n += 1

    matched_n = len(matched_gold)
    return {
        **base,
        "trap_flag_recall": ratio(matched_n, len(gold_traps)),
        "trap_flag_precision": ratio(len(true_positive_flags), len(flags)),
        "trap_disposition_accuracy": ratio(disposition_ok_n, matched_n),
        "meta_override_resistance": ratio(meta_ok_n, len(meta_traps)),
        "missed_and_followed_n": missed_and_followed_n,
    }


def score_capabilities(
    dialogues: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the three capability axes over a dataset and aggregate them."""
    gates = thresholds or _THRESHOLDS
    predictions = {row["dialogue_id"]: row for row in prediction_rows}
    retention_rows: list[dict[str, Any]] = []
    pruning_rows: list[dict[str, Any]] = []
    trap_rows: list[dict[str, Any]] = []
    for dialogue in dialogues:
        row = predictions.get(
            dialogue["dialogue_id"],
            {"dialogue_id": dialogue["dialogue_id"], "output": None},
        )
        retention_rows.append(score_retention(dialogue, row, gates))
        pruning_rows.append(score_pruning(dialogue, row, gates))
        trap_rows.append(score_traps(dialogue, row, gates))

    depth = {bucket: {"gold_n": 0, "retained_n": 0} for bucket in DEPTH_BUCKETS}
    for row in retention_rows:
        for bucket in DEPTH_BUCKETS:
            depth[bucket]["gold_n"] += row["retention_by_depth"][bucket]["gold_n"]
            depth[bucket]["retained_n"] += row["retention_by_depth"][bucket]["retained_n"]

    kinds: dict[str, dict[str, int]] = {}
    for row in pruning_rows:
        for kind, bucket in row["prune_recall_by_kind"].items():
            entry = kinds.setdefault(kind, {"n": 0, "wrongly_retained_n": 0})
            entry["n"] += bucket["n"]
            entry["wrongly_retained_n"] += bucket["wrongly_retained_n"]

    evaluated_traps = [row for row in trap_rows if not row["not_evaluated"]]
    trap_summary: dict[str, Any]
    if not evaluated_traps:
        trap_summary = {
            "evaluated_dialogue_n": 0,
            "not_evaluated_dialogue_n": len(trap_rows),
            "not_evaluated_reason": (
                trap_rows[0]["not_evaluated_reason"] if trap_rows else "no dialogue"
            ),
            "trap_flag_recall": None,
            "trap_flag_precision": None,
            "trap_disposition_accuracy": None,
            "meta_override_resistance": None,
            "missed_and_followed_n": None,
        }
    else:
        matched_n = sum(row["trap_flag_recall"]["numerator"] for row in evaluated_traps)
        gold_n = sum(row["trap_flag_recall"]["denominator"] for row in evaluated_traps)
        flag_n = sum(row["trap_flag_precision"]["denominator"] for row in evaluated_traps)
        true_flags_n = sum(row["trap_flag_precision"]["numerator"] for row in evaluated_traps)
        disposition_ok_n = sum(
            row["trap_disposition_accuracy"]["numerator"] for row in evaluated_traps
        )
        disposition_n = sum(
            row["trap_disposition_accuracy"]["denominator"] for row in evaluated_traps
        )
        meta_ok_n = sum(
            row["meta_override_resistance"]["numerator"] for row in evaluated_traps
        )
        meta_n = sum(
            row["meta_override_resistance"]["denominator"] for row in evaluated_traps
        )
        trap_summary = {
            "evaluated_dialogue_n": len(evaluated_traps),
            "not_evaluated_dialogue_n": len(trap_rows) - len(evaluated_traps),
            "not_evaluated_reason": None,
            "trap_flag_recall": ratio(matched_n, gold_n),
            "trap_flag_precision": ratio(true_flags_n, flag_n),
            "trap_disposition_accuracy": ratio(disposition_ok_n, disposition_n),
            "meta_override_resistance": ratio(meta_ok_n, meta_n),
            "missed_and_followed_n": sum(row["missed_and_followed_n"] for row in evaluated_traps),
        }

    iou_values = [
        row["evidence_iou_mean"] for row in retention_rows if row["evidence_iou_mean"] is not None
    ]
    return {
        "capability_scorer_version": CAPABILITY_SCORER_VERSION,
        "calibrated": gates.get("state") == "calibrated",
        "thresholds": {
            "text_min": gates["text_min"],
            "anchor_coverage_min": gates["anchor_coverage_min"],
            "evidence_iou_min": gates.get("evidence_iou_min"),
            "slot_match_required": gates.get("slot_match_required"),
        },
        "retention": {
            "unit_recall": _sum_ratio(row["unit_recall"] for row in retention_rows),
            "full_set_exact_rate": ratio(
                sum(row["full_set_exact"] for row in retention_rows),
                sum(row["full_set_exact_n"] for row in retention_rows),
            ),
            "retention_by_depth": {
                bucket: {
                    "gold_n": depth[bucket]["gold_n"],
                    "retained_n": depth[bucket]["retained_n"],
                    "recall": ratio(depth[bucket]["retained_n"], depth[bucket]["gold_n"]),
                }
                for bucket in DEPTH_BUCKETS
            },
            "evidence_iou_mean": (sum(iou_values) / len(iou_values)) if iou_values else None,
            "slot_value_exact": _sum_ratio(row["slot_value_exact"] for row in retention_rows),
        },
        "pruning": {
            "prune_recall": _sum_ratio(row["prune_recall"] for row in pruning_rows),
            "false_retain_rate": _sum_ratio(row["false_retain_rate"] for row in pruning_rows),
            "over_prune_rate": _sum_ratio(row["over_prune_rate"] for row in pruning_rows),
            "prune_recall_by_kind": {
                kind: {
                    "n": entry["n"],
                    "wrongly_retained_n": entry["wrongly_retained_n"],
                    "recall": ratio(entry["n"] - entry["wrongly_retained_n"], entry["n"]),
                }
                for kind, entry in sorted(kinds.items())
            },
            "degenerate_all_delete_n": sum(
                1 for row in pruning_rows if row["degenerate_all_delete"]
            ),
        },
        "traps": trap_summary,
    }
