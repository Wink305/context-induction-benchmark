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


def _flatten_aligned(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], threshold: float = 0.32
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_b = {row["dialogue_id"]: row for row in rows_b}
    aligned = []
    adjudication = []
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
            reasons = []
            if a.get("type") != b.get("type"):
                reasons.append("type_mismatch")
            if a.get("valid_from") != b.get("valid_from") or a.get("valid_to") != b.get("valid_to"):
                reasons.append("interval_mismatch")
            if reasons:
                adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": reasons, "annotator_a": a, "annotator_b": b, "decision": None, "reviewer": None, "status": "pending"})
        for index in unmatched_a:
            adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": ["unmatched_a"], "annotator_a": annotations_a[index], "annotator_b": None, "decision": None, "reviewer": None, "status": "pending"})
        for index in unmatched_b:
            adjudication.append({"conflict_id": f"adj-{len(adjudication) + 1:04d}", "dialogue_id": dialogue_id, "reasons": ["unmatched_b"], "annotator_a": None, "annotator_b": annotations_b[index], "decision": None, "reviewer": None, "status": "pending"})
    return aligned, adjudication


def _interval_match(a: dict[str, Any], b: dict[str, Any], tolerance: int = 2) -> bool:
    if a.get("valid_to") is None or b.get("valid_to") is None:
        end_match = a.get("valid_to") == b.get("valid_to")
    else:
        end_match = abs(a["valid_to"] - b["valid_to"]) <= tolerance
    return abs(a.get("valid_from", -10**9) - b.get("valid_from", 10**9)) <= tolerance and end_match


def agreement_report(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, bootstrap_n: int = 1000, seed: int = 20261115
) -> dict[str, Any]:
    aligned, adjudication = _flatten_aligned(rows_a, rows_b)
    labels_a = [pair["a"]["type"] for pair in aligned]
    labels_b = [pair["b"]["type"] for pair in aligned]
    overall = cohen_kappa(labels_a, labels_b)

    by_dialogue: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(aligned):
        by_dialogue[pair["dialogue_id"]].append(index)
    dialogue_ids = sorted(by_dialogue)
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap_n):
        sampled_indices = []
        for _ in dialogue_ids:
            sampled = rng.choice(dialogue_ids)
            sampled_indices.extend(by_dialogue[sampled])
        value = cohen_kappa([labels_a[i] for i in sampled_indices], [labels_b[i] for i in sampled_indices])["kappa"]
        if value is not None:
            boot.append(value)
    boot.sort()
    ci = None
    if boot:
        ci = [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]

    per_type = {}
    for ctype in TYPE_TO_CODE:
        binary_a = [str(label == ctype) for label in labels_a]
        binary_b = [str(label == ctype) for label in labels_b]
        kappa = cohen_kappa(binary_a, binary_b)
        ac1 = gwet_ac1(binary_a, binary_b)
        n = len(binary_a)
        positive_n = sum(label == ctype for label in labels_a)
        rare = positive_n < 50
        threshold_metric = ac1["ac1"] if rare else kappa["kappa"]
        per_type[ctype] = {
            **kappa,
            "gwet_ac1": ac1["ac1"],
            "positive_n_annotator_a": positive_n,
            "prevalence_annotator_a": positive_n / n if n else None,
            "rare_class_fallback": rare,
            "pass_0_60": threshold_metric is not None and threshold_metric >= 0.60,
        }

    interval_n = sum(_interval_match(pair["a"], pair["b"]) for pair in aligned)
    boundary = {}
    for rule in (f"R{i}" for i in range(1, 9)):
        relevant = [pair for pair in aligned if rule in pair["a"].get("rule_tags", []) or rule in pair["b"].get("rule_tags", [])]
        agree = sum((rule in pair["a"].get("rule_tags", [])) == (rule in pair["b"].get("rule_tags", [])) for pair in relevant)
        boundary[rule] = ratio(agree, len(relevant))
    boundary_all = ratio(sum(item["numerator"] for item in boundary.values()), sum(item["denominator"] for item in boundary.values()))
    ci_lower = ci[0] if ci else None
    checks = {
        "overall_kappa_gte_0_70": overall["kappa"] is not None and overall["kappa"] >= 0.70,
        "overall_ci_lower_gte_0_60": ci_lower is not None and ci_lower >= 0.60,
        "per_type_pass": all(item["pass_0_60"] for item in per_type.values()),
        "interval_gte_0_80": interval_n / len(aligned) >= 0.80 if aligned else False,
        "boundary_gte_0_75": boundary_all["value"] is not None and boundary_all["value"] >= 0.75,
        "no_pending_adjudication": not adjudication,
    }
    return {
        "method": "matched_units_cohen_kappa_dialogue_cluster_bootstrap",
        "matcher_calibrated": False,
        "aligned_unit_n": len(aligned),
        "unresolved_conflict_n": len(adjudication),
        "overall": {**overall, "ci_95_percentile_cluster_bootstrap": ci, "bootstrap_n": bootstrap_n},
        "per_type": per_type,
        "interval_agreement": ratio(interval_n, len(aligned)),
        "boundary_rules": boundary,
        "boundary_overall": boundary_all,
        "exit_checks": checks,
        "gold_freeze_allowed": all(checks.values()),
        "adjudication_queue": adjudication,
        "warning": "Agreement is computed only on aligned units; unmatched extraction disagreements remain in the adjudication queue.",
    }
