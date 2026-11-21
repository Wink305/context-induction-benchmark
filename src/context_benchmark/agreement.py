"""P0 annotation agreement, cluster-bootstrap CI and adjudication queue."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .ontology import TYPE_TO_CODE
from .provenance import classify_annotator
from .scoring import greedy_match, ratio
from .validation import annotation_content_hash, row_units


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
        annotations_a = row_units(row_a)
        annotations_b = row_units(row_b)
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
# clear 0.60 and the point estimate to clear 0.70.
#
# The gate is Cohen's kappa and only Cohen's kappa — for every class,
# including the rare ones.  Gwet's AC1 is a different scale; it is reported
# next to kappa for information and never decides whether a run passes.
KAPPA_POINT_MIN = 0.70
KAPPA_CI_LOWER_MIN = 0.60
# Reference value for the informational AC1 only.  AC1 has no gate: it is
# reported, never enforced.
AC1_POINT_MIN = 0.75
EXTRACTION_MATCH_MIN = 0.95


def _binary_kappa(n11: int, n10: int, n01: int, n00: int) -> float | None:
    """Cohen's kappa for a two-category table.

    ``n11`` counts units both annotators put in the class, ``n10``/``n01``
    the two disagreement directions and ``n00`` the units neither put in it.
    Same arithmetic as :func:`cohen_kappa` on a binary label pair, written
    on counts so a bootstrap resample does not have to rebuild label lists.
    """
    n = n11 + n10 + n01 + n00
    if not n:
        return None
    observed = (n11 + n00) / n
    expected = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _percentile_ci(values: list[float]) -> list[float] | None:
    """95% percentile interval, or None when there are too few resamples."""
    if not values:
        return None
    ordered = sorted(values)
    return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]


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
    classes = list(TYPE_TO_CODE) + [ABSENT]
    rng = random.Random(seed)
    boot = []
    # Per-class bootstrap values.  The gate is per class, so every class gets
    # the same treatment — including the rare ones that used to fall back to
    # AC1 and escape the kappa threshold.
    type_boot: dict[str, list[float]] = {ctype: [] for ctype in classes}
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
        # One pass over the resampled units, then the two-category counts per
        # class.  Same numbers cohen_kappa would compute on a binary label.
        pair_counts: dict[tuple[Any, Any], int] = {}
        for index in sampled_indices:
            key = (labels_a[index], labels_b[index])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        for ctype in classes:
            n11 = pair_counts.get((ctype, ctype), 0)
            n10 = sum(v for (a, _b), v in pair_counts.items() if a == ctype) - n11
            n01 = sum(v for (_a, b), v in pair_counts.items() if b == ctype) - n11
            n00 = len(sampled_indices) - n11 - n10 - n01
            type_value = _binary_kappa(n11, n10, n01, n00)
            if type_value is not None:
                type_boot[ctype].append(type_value)
    boot.sort()
    # Require a usable number of non-degenerate resamples before quoting a CI.
    # A tiny pilot bootstrap (e.g. 20) still gets an interval, but we disclose
    # how many resamples were dropped so no one mistakes it for a stable estimate.
    min_boot = min(30, max(5, int(0.5 * bootstrap_n)))
    ci = _percentile_ci(boot) if len(boot) >= min_boot else None
    ci_lower = ci[0] if ci else None

    per_type = {}
    for ctype in classes:
        binary_a = [str(label == ctype) for label in labels_a]
        binary_b = [str(label == ctype) for label in labels_b]
        kappa = cohen_kappa(binary_a, binary_b)
        ac1 = gwet_ac1(binary_a, binary_b)
        n = len(binary_a)
        positive_n = sum(label == ctype for label in labels_a)
        positive_b = sum(label == ctype for label in labels_b)
        rare = max(positive_n, positive_b) < 50
        # If neither annotator ever used this class, kappa/AC1 are undefined by
        # construction (expected agreement == 1). That is trivial agreement, not
        # a failure — otherwise a clean run could never pass the per-type gate.
        both_absent = positive_n == 0 and positive_b == 0
        type_ci = _percentile_ci(type_boot[ctype]) if len(type_boot[ctype]) >= min_boot else None
        type_ci_lower = type_ci[0] if type_ci else None
        kappa_point = kappa["kappa"]
        failures: list[str] = []
        if not both_absent:
            # Kappa is the gate for every class.  A rare class is not allowed
            # to buy its way past the threshold with AC1.
            if kappa_point is None:
                failures.append("cohen_kappa undefined")
            elif kappa_point < KAPPA_POINT_MIN:
                failures.append(f"cohen_kappa {kappa_point:.3f} < {KAPPA_POINT_MIN}")
            # The CI lower bound is enforced whenever the bootstrap can produce
            # one.  A class with too little support to estimate an interval is
            # reported as such — that is a data limitation, not a disagreement,
            # so it never becomes a silent pass.
            if type_ci_lower is not None and type_ci_lower < KAPPA_CI_LOWER_MIN:
                failures.append(f"cohen_kappa CI lower {type_ci_lower:.3f} < {KAPPA_CI_LOWER_MIN}")
        per_type[ctype] = {
            **kappa,
            "gwet_ac1": ac1["ac1"],
            "gwet_ac1_report_only": True,
            "positive_n_annotator_a": positive_n,
            "positive_n_annotator_b": positive_b,
            "prevalence_annotator_a": positive_n / n if n else None,
            "rare_class": rare,
            "both_annotators_never_used_class": both_absent,
            "threshold_used": KAPPA_POINT_MIN,
            "threshold_metric": "cohen_kappa",
            "kappa_ci_95_percentile_cluster_bootstrap": type_ci,
            "kappa_ci_lower_min": KAPPA_CI_LOWER_MIN,
            "kappa_ci_status": "estimated" if type_ci else "not_estimable",
            "gate_failures": failures,
            "pass_gate": not failures,
        }

    interval_n = sum(_interval_match(pair["a"], pair["b"]) for pair in aligned)
    # Boundary agreement is computed on the annotators' own R1-R8 tags.
    # Those tags are a linguistic judgement and are therefore collected in
    # the sheet's seventh column, never inferred: on the shipped pilot,
    # 113 units share the shape (one_off + task scope) and only 35 of them
    # are R1, so any field-based derivation mislabels 35 % of the units.
    boundary = {}
    for rule in (f"R{i}" for i in range(1, 9)):
        relevant = [pair for pair in aligned if rule in pair["a"].get("rule_tags", []) or rule in pair["b"].get("rule_tags", [])]
        agree = sum((rule in pair["a"].get("rule_tags", [])) == (rule in pair["b"].get("rule_tags", [])) for pair in relevant)
        boundary[rule] = ratio(agree, len(relevant))
    boundary_denominator = sum(item["denominator"] for item in boundary.values())
    boundary_all = ratio(sum(item["numerator"] for item in boundary.values()), boundary_denominator)
    # A boundary denominator of zero means no unit carried a tag, so the
    # R1-R8 agreement was never exercised.  That is not a pass: it is
    # reported as not assessed, and the gate refuses to claim it.
    boundary_assessed = boundary_denominator > 0
    tagged_pair_n = sum(
        1 for pair in aligned if pair["a"].get("rule_tags") or pair["b"].get("rule_tags")
    )

    total_extracted = extraction["matched_n"] + extraction["unmatched_a_n"] + extraction["unmatched_b_n"]
    extraction_match = ratio(extraction["matched_n"], total_extracted)
    extraction_rate = extraction_match["value"]
    checks = {
        "overall_kappa_gte_0_70": overall["kappa"] is not None and overall["kappa"] >= KAPPA_POINT_MIN,
        "overall_ci_lower_gte_0_60": ci_lower is not None and ci_lower >= KAPPA_CI_LOWER_MIN,
        "per_type_pass": all(item["pass_gate"] for item in per_type.values()),
        "interval_gte_0_80": interval_n / len(aligned) >= 0.80 if aligned else False,
        # Not assessed means not passed.  An unexercised boundary rule is a
        # hole in the annotation, not a clean run.
        "boundary_gte_0_75": (
            (boundary_all["value"] is not None and boundary_all["value"] >= 0.75)
            if boundary_assessed
            else False
        ),
        "extraction_match_gte_0_95": extraction_rate is not None and extraction_rate >= EXTRACTION_MATCH_MIN,
        "no_pending_adjudication": not adjudication,
    }
    # Overall AC1: reported for information only.  It is a different scale
    # from kappa and it never gates a run.
    overall_ac1 = gwet_ac1(labels_a, labels_b)["ac1"]
    annotator_labels = [
        row.get("annotator")
        for row in list(rows_a) + list(rows_b)
        if isinstance(row, dict) and row.get("annotator") is not None
    ]
    annotator_labels = [str(label).strip() for label in annotator_labels if str(label).strip()]
    annotator_type = classify_annotator(annotator_labels)
    # A perfect kappa with no conflicts and no extraction disagreement can be
    # genuine agreement — or it can be one person labelling the same set twice
    # under two names.  Flag it, do not fail on it: a small clean corpus may
    # legitimately reach 1.0, and the freezer has a stronger content-identity
    # check for the copy case.  What matters here is that a downstream reader
    # cannot mistake a vacuous agreement measurement for a measured one.
    unmatched_n = extraction["unmatched_a_n"] + extraction["unmatched_b_n"]
    suspect_self_agreement = (
        overall["kappa"] == 1.0
        and not adjudication
        and not unmatched_n
        and bool(labels_a)
    )
    return {
        "method": "all_units_cohen_kappa_with_absent_class_dialogue_cluster_bootstrap",
        "matcher_calibrated": False,
        # Bind this report to the exact annotation sets it was computed from.
        # Without these hashes a hand-written report with fabricated gate
        # results is indistinguishable from a real one at the freeze gate.
        "annotations_a_sha256": annotation_content_hash(rows_a),
        "annotations_b_sha256": annotation_content_hash(rows_b),
        "suspect_self_agreement": suspect_self_agreement,
        "suspect_self_agreement_note": (
            "The two annotation sets produced perfect agreement with no conflicts and no "
            "extraction disagreements. This may be genuine, or one person labelling twice "
            "under two names. Reported as a warning only; the freeze gate compares the two "
            "sets by content."
            if suspect_self_agreement
            else "No vacuous-agreement pattern detected."
        ),
        "aligned_unit_n": len(aligned),
        "unmatched_a_n": extraction["unmatched_a_n"],
        "unmatched_b_n": extraction["unmatched_b_n"],
        "unresolved_conflict_n": len(adjudication),
        "extraction_match_rate": extraction_match,
        "unit_n": len(labels_a),
        "overall": {**overall, "ci_95_percentile_cluster_bootstrap": ci, "bootstrap_n": bootstrap_n, "bootstrap_skipped_n": boot_skipped},
        "gwet_ac1": {
            "overall": overall_ac1,
            "report_only": True,
            "reference_threshold": AC1_POINT_MIN,
            "at_or_above_reference": overall_ac1 is not None and overall_ac1 >= AC1_POINT_MIN,
            "note": "Gwet's AC1 is reported for information; the agreement gate is Cohen's kappa only.",
        },
        "annotator_type": annotator_type,
        "annotator_n": len(set(annotator_labels)) if annotator_labels else None,
        # The labels themselves, so the "two independent annotators" claim can
        # be audited after the fact.  The freezer can verify that two
        # *distinct labels* exist; it cannot verify that two distinct *people*
        # did the work — that is a process control, and it needs the labels to
        # be on the record.
        "annotator_labels": sorted(set(annotator_labels)),
        "adjudication_method": "third_reviewer_decision_per_conflict",
        "matched_units_kappa_reference": matched_only,
        "per_type": per_type,
        "interval_agreement": ratio(interval_n, len(aligned)),
        "boundary_rules": boundary,
        "boundary_overall": boundary_all,
        "boundary_assessed": boundary_assessed,
        "boundary_tagged_pair_n": tagged_pair_n,
        "boundary_note": (
            "R1-R8 agreement was measured on the annotators' own boundary tags."
            if boundary_assessed
            else "No annotated unit carried a boundary tag, so R1-R8 agreement was never "
            "exercised. Annotators must fill the 边界标签 column; this gate cannot be "
            "assumed to have passed."
        ),
        "exit_checks": checks,
        "gold_freeze_allowed": all(checks.values()),
        "adjudication_queue": adjudication,
        "warning": (
            "Kappa includes unmatched units as an ABSENT class, so extraction "
            "disagreements can no longer be hidden by paraphrasing past the matcher. "
            "A separate extraction_match_rate >= 0.95 gate is enforced. "
            "The gate is Cohen's kappa for every class, rare ones included; "
            "gwet_ac1 is reported for information and never gates."
        ),
    }
