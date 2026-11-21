"""Deterministic A/C scoring with explicit numerators and denominators.

The evidence/text matcher is an *uncalibrated engineering baseline*, not a
validated semantic judge. P0 must calibrate or replace it before any model claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .ontology import ACTIVE_TYPES, TYPE_TO_CODE
from .semantic import semantic_conflict
from .validation import public_constraint, validate_output

MATCHER_VERSION = "evidence_char_ngram_greedy_v3_text_gated_semantic_guard"
# These gates are UNCALIBRATED engineering heuristics. They were chosen so that
# a hand-written short paraphrase of a constructed unit survives while pure
# garbage (unrelated text with perfect evidence) does not. No human annotation
# exists behind either number, and the pilot candidate truth is not gold, so
# these thresholds must not be described as "calibrated on human prechecks".
# P1 must fit them on an independent, human-labelled calibration set and report
# sensitivity before any model score is reported. See design doc §5 and §11.2.
TEXT_MATCH_MIN = 0.10
# The predicted text must also be recoverable from the user turns it cites.
# Coverage = |bigrams(text) ∩ bigrams(cited turns)| / |bigrams(text)|.
ANCHOR_COVERAGE_MIN = 0.08
CONTENT_TEXT_MIN = TEXT_MATCH_MIN


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
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = []
    for gi, gold_item in enumerate(gold):
        for pi, prediction in enumerate(predictions):
            text_sim = _dice(_ngrams(gold_item.get("text", "")), _ngrams(prediction.get("text", "")))
            # Hard text gate: a perfect evidence overlap must not be enough to
            # match a prediction whose text has nothing to do with the gold.
            if text_sim < TEXT_MATCH_MIN:
                continue
            if turns is not None and anchor_coverage(
                prediction.get("text", ""), turns, prediction.get("evidence_turns", [])
            ) < ANCHOR_COVERAGE_MIN:
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
