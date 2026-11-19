"""P0 annotation agreement, cluster-bootstrap CI and adjudication queue."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .ontology import TYPE_TO_CODE
from .scoring import greedy_match, ratio


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    if len(labels_a) != len(labels_b):
        raise ValueError("label arrays must have equal length")
    n = len(labels_a)
    if not n:
        return {"kappa": None, "observed_agreement": None, "expected_agreement": None, "n": 0}
    labels = sorted(set(labels_a) | set(labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = sum(counts_a[label] * counts_b[label] for label in labels) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"kappa": kappa, "observed_agreement": observed, "expected_agreement": expected, "n": n}


def gwet_ac1(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    n = len(labels_a)
    if not n:
        return {"ac1": None, "n": 0}
    labels = sorted(set(labels_a) | set(labels_b))
    q = len(labels)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    if q <= 1:
        return {"ac1": None, "n": n}
    expected = 0.0
    for label in labels:
        pi = (labels_a.count(label) + labels_b.count(label)) / (2 * n)
        expected += pi * (1 - pi)
    expected /= q - 1
    return {"ac1": (observed - expected) / (1 - expected) if expected < 1 else None, "n": n}


ABSENT = "__absent__"  # synthetic label for a unit one annotator extracted and the other did not


def _flatten_aligned(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], threshold: float = 0.32
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_b = {row["dialogue_id"]: row for row in rows_b}
    aligned = []
    adjudication = []
    # These carry every unit, including extraction disagreements. They are the
    # only labels allowed into the agreement gate: computing kappa on matched
    # units alone lets an annotator erase a disagreement by paraphrasing a unit
    # past the matcher and watching it vanish from the denominator.
    labels_a: list[str] = []
    labels_b: list[str] = []
    matched_n = 0
    unmatched_a_n = 0
    unmatched_b_n = 0
    per_dialogue_labels: dict[str, list[int]] = defaultdict(list)
    for row_a in rows_a:
        dialogue_id = row_a["dialogue_id"]
        row_b = by_b.get(dialogue_id, {"annotations": []})
        annotations_a = row_a.get("annotations", [])
        annotations_b = row_b.get("annotations", [])
        matches, unmatched_a, unmatched_b = greedy_match(annotations_a, annotations_b, threshold)
        for ai, bi, similarity in matches:
            a = annotations_a[ai]
            b = annotations_b[bi]
            pair = {"dialogue_id": dialogue_id, "a": a, "b": b, "similarity": similarity}
            aligned.append(pair)
            per_dialogue_labels[dialogue_id].append(len(labels_a))
            labels_a.append(a.get("type"))
            labels_b.append(b.get("type"))
            matched_n += 1
            reasons = []
            if a.get("type") != b.get("type"):
                reasons.append("type_mismatch")
            if a.get("valid_from") != b.get("valid_from") or a.get("valid_to") != b.get("valid_to"):
                reasons.append("interval_mismatch")
            if reasons:
                adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": reasons, "a_index": ai, "b_index": bi, "annotator_a": a, "annotator_b": b, "decision": None, "reviewer": None, "status": "pending"})
        for index in unmatched_a:
            per_dialogue_labels[dialogue_id].append(len(labels_a))
            labels_a.append(annotations_a[index].get("type"))
            labels_b.append(ABSENT)
            unmatched_a_n += 1
            adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": ["unmatched_a"], "a_index": index, "b_index": None, "annotator_a": annotations_a[index], "annotator_b": None, "decision": None, "reviewer": None, "status": "pending"})
        for index in unmatched_b:
            per_dialogue_labels[dialogue_id].append(len(labels_a))
            labels_a.append(ABSENT)
            labels_b.append(annotations_b[index].get("type"))
            unmatched_b_n += 1
            adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": ["unmatched_b"], "a_index": None, "b_index": index, "annotator_a": None, "annotator_b": annotations_b[index], "decision": None, "reviewer": None, "status": "pending"})
    extraction = {
        "matched_n": matched_n,
        "unmatched_a_n": unmatched_a_n,
        "unmatched_b_n": unmatched_b_n,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "per_dialogue_label_indices": dict(per_dialogue_labels),
    }
    return aligned, adjudication, extraction


def _interval_match(a: dict[str, Any], b: dict[str, Any], tolerance: int = 2) -> bool:
    if a.get("valid_to") is None or b.get("valid_to") is None:
        end_match = a.get("valid_to") == b.get("valid_to")
    else:
        end_match = abs(a["valid_to"] - b["valid_to"]) <= tolerance
    return abs(a.get("valid_from", -10**9) - b.get("valid_from", 10**9)) <= tolerance and end_match


# Single authority for the human-agreement gate. Design record §5 / Landis &
# Koch: kappa >= 0.61 is "substantial"; we require the 95% CI lower bound to
# clear 0.60 and the point estimate to clear 0.70. AC1 is a different scale and
# gets its own (stricter) threshold; never reuse the kappa number for it.
KAPPA_POINT_MIN = 0.70
KAPPA_CI_LOWER_MIN = 0.60
AC1_POINT_MIN = 0.75
EXTRACTION_MATCH_MIN = 0.95


def agreement_report(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, bootstrap_n: int = 1000, seed: int = 20261115
) -> dict[str, Any]:
    aligned, adjudication, extraction = _flatten_aligned(rows_a, rows_b)
    labels_a = extraction["labels_a"]
    labels_b = extraction["labels_b"]
    # Kappa is computed on ALL units, with extraction disagreements represented
    # as ABSENT. This closes the loophole where a paraphrase that fails the
    # matcher removed a real disagreement from the denominator.
    overall = cohen_kappa(labels_a, labels_b)
    matched_only = cohen_kappa(
        [pair["a"]["type"] for pair in aligned], [pair["b"]["type"] for pair in aligned]
    )

    per_dialogue_labels = extraction["per_dialogue_label_indices"]
    dialogue_ids = sorted(per_dialogue_labels)
    rng = random.Random(seed)
    boot = []
    boot_skipped = 0
    for _ in range(bootstrap_n):
        sampled_indices = []
        for _ in dialogue_ids:
            sampled = rng.choice(dialogue_ids)
            sampled_indices.extend(per_dialogue_labels[sampled])
        value = cohen_kappa(
            [labels_a[i] for i in sampled_indices], [labels_b[i] for i in sampled_indices]
        )["kappa"]
        if value is None:
            boot_skipped += 1
        else:
            boot.append(value)
    boot.sort()
    ci = None
    # Require a usable number of non-degenerate resamples before quoting a CI.
    # A tiny pilot bootstrap (e.g. 20) still gets an interval, but we disclose
    # how many resamples were dropped so no one mistakes it for a stable estimate.
    min_boot = min(30, max(5, int(0.5 * bootstrap_n)))
    if len(boot) >= min_boot:
        ci = [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]
    ci_lower = ci[0] if ci else None

    per_type = {}
    for ctype in list(TYPE_TO_CODE) + [ABSENT]:
        binary_a = [str(label == ctype) for label in labels_a]
        binary_b = [str(label == ctype) for label in labels_b]
        kappa = cohen_kappa(binary_a, binary_b)
        ac1 = gwet_ac1(binary_a, binary_b)
        n = len(binary_a)
        positive_n = sum(label == ctype for label in labels_a)
        positive_b = sum(label == ctype for label in labels_b)
        rare = max(positive_n, positive_b) < 50
        # AC1 and kappa are not on the same scale; each gets its own threshold.
        threshold = AC1_POINT_MIN if rare else KAPPA_POINT_MIN
        metric_value = ac1["ac1"] if rare else kappa["kappa"]
        # If neither annotator ever used this class, kappa/AC1 are undefined by
        # construction (expected agreement == 1). That is trivial agreement, not
        # a failure — otherwise a clean run could never pass the per-type gate.
        both_absent = positive_n == 0 and positive_b == 0
        per_type[ctype] = {
            **kappa,
            "gwet_ac1": ac1["ac1"],
            "positive_n_annotator_a": positive_n,
            "positive_n_annotator_b": positive_b,
            "prevalence_annotator_a": positive_n / n if n else None,
            "rare_class_fallback": rare,
            "both_annotators_never_used_class": both_absent,
            "threshold_used": threshold,
            "threshold_metric": "gwet_ac1" if rare else "cohen_kappa",
            "pass_gate": True if both_absent else (metric_value is not None and metric_value >= threshold),
        }

    interval_n = sum(_interval_match(pair["a"], pair["b"]) for pair in aligned)
    boundary = {}
    for rule in (f"R{i}" for i in range(1, 9)):
        relevant = [pair for pair in aligned if rule in pair["a"].get("rule_tags", []) or rule in pair["b"].get("rule_tags", [])]
        agree = sum((rule in pair["a"].get("rule_tags", [])) == (rule in pair["b"].get("rule_tags", [])) for pair in relevant)
        boundary[rule] = ratio(agree, len(relevant))
    boundary_all = ratio(sum(item["numerator"] for item in boundary.values()), sum(item["denominator"] for item in boundary.values()))

    total_extracted = extraction["matched_n"] + extraction["unmatched_a_n"] + extraction["unmatched_b_n"]
    extraction_match = ratio(extraction["matched_n"], total_extracted)
    extraction_rate = extraction_match["value"]
    checks = {
        "overall_kappa_gte_0_70": overall["kappa"] is not None and overall["kappa"] >= KAPPA_POINT_MIN,
        "overall_ci_lower_gte_0_60": ci_lower is not None and ci_lower >= KAPPA_CI_LOWER_MIN,
        "per_type_pass": all(item["pass_gate"] for item in per_type.values()),
        "interval_gte_0_80": interval_n / len(aligned) >= 0.80 if aligned else False,
        "boundary_gte_0_75": boundary_all["value"] is not None and boundary_all["value"] >= 0.75,
        "extraction_match_gte_0_95": extraction_rate is not None and extraction_rate >= EXTRACTION_MATCH_MIN,
        "no_pending_adjudication": not adjudication,
    }
    return {
        "method": "all_units_cohen_kappa_with_absent_class_dialogue_cluster_bootstrap",
        "matcher_calibrated": False,
        "aligned_unit_n": len(aligned),
        "unmatched_a_n": extraction["unmatched_a_n"],
        "unmatched_b_n": extraction["unmatched_b_n"],
        "unresolved_conflict_n": len(adjudication),
        "extraction_match_rate": extraction_match,
        "overall": {**overall, "ci_95_percentile_cluster_bootstrap": ci, "bootstrap_n": bootstrap_n, "bootstrap_skipped_n": boot_skipped},
        "matched_units_kappa_reference": matched_only,
        "per_type": per_type,
        "interval_agreement": ratio(interval_n, len(aligned)),
        "boundary_rules": boundary,
        "boundary_overall": boundary_all,
        "exit_checks": checks,
        "gold_freeze_allowed": all(checks.values()),
        "adjudication_queue": adjudication,
        "warning": (
            "Kappa includes unmatched units as an ABSENT class, so extraction "
            "disagreements can no longer be hidden by paraphrasing past the matcher. "
            "A separate extraction_match_rate >= 0.95 gate is enforced."
        ),
    }
