"""Human adjudication gate and immutable gold freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .agreement import agreement_report
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .provenance import (
    collect_generator_findings,
    collect_provenance_findings,
    generator_truth_laundering,
    normalize_value,
)
from .validation import (
    annotation_content_hash,
    public_constraint,
    row_units,
    truth_rows_as_annotations,
    validate_output,
)

CORE_EXIT_CHECKS = (
    "overall_kappa_gte_0_70", "overall_ci_lower_gte_0_60", "per_type_pass",
    "interval_gte_0_80", "boundary_gte_0_75", "extraction_match_gte_0_95",
)

HUMAN_PROVENANCE = "human"
HUMAN_EVIDENCE_GRADE = "human_gold"


def _iter_dicts(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield every dictionary inside a nested payload."""
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_dicts(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_dicts(item)


def _assert_human_provenance(
    adjudicated_annotations: list[dict[str, Any]],
    agreement: dict[str, Any],
    resolutions: list[dict[str, Any]],
    candidate_dialogues: list[dict[str, Any]],
    annotations_b: list[dict[str, Any]] | None = None,
) -> None:
    """Refuse to freeze anything that is not provably human annotation.

    Five layers, all checked on the **raw** payload — before any field is
    normalised away — so a marker cannot be laundered by
    ``public_constraint``:

    1. generator origin: ``candidate_only`` on a unit, a generator ``status``
       on a dialogue, or a note that says the unit awaits a human decision;
    2. non-human origin: a ``provenance`` / ``annotator`` / ``reviewer``
       field or a ``note`` that names a non-human annotator (AI, model,
       generator, …);
    3. completion flags: ``human_annotation_complete`` that is not true, or
       an ``evidence_grade`` that is not human gold;
    4. an affirmative human claim: every annotation row must declare
       ``provenance: "human"``;
    5. two independent annotators — enforced by :func:`freeze_gold`, which
       has the agreement report in hand.

    Layers 1-3 are searches for bad news.  Layer 4 is the opposite: it
    requires the payload to *say* where it came from.  Without it, a payload
    with every marker stripped would pass the "no evidence of a non-human
    origin" test and then get stamped ``gold_provenance: "human"`` — a false
    claim made by the freezer itself.  Requiring the claim means the freezer
    checks a statement somebody actually wrote down, and the statement can be
    audited later.

    The order matters: the specific diagnoses (generator, AI) come first, so
    the message names the actual problem rather than the generic one.
    """
    marker_sources = (
        ("adjudicated_annotations", adjudicated_annotations),
        # The second annotator's set is just as capable of carrying a
        # generator/AI claim as the first one.  Scanning only the baseline
        # would let a clean A set certify a machine-produced B set.
        ("annotations_b", annotations_b or []),
        ("agreement_report", agreement),
        ("resolutions", resolutions),
        # The dialogue source is scanned for provenance too, not only for
        # completion flags: the generator marks its dialogue rows
        # ``status: "generated_unannotated_not_gold"``.
        ("candidate_dialogues", candidate_dialogues),
    )
    generator_findings: list[dict[str, Any]] = []
    for name, payload in marker_sources:
        generator_findings.extend(collect_generator_findings(payload, path=name))
    if generator_findings:
        detail = "; ".join(
            f"{item['path']}={item['value']!r} ({item['reason']})"
            for item in generator_findings[:6]
        )
        raise ValueError(
            "Refusing to freeze gold: the payload carries generator provenance: "
            + detail
            + ". Generator truth is rule-audited, not human-annotated; it cannot back a "
            "human gold claim. Point freeze-gold --candidate at the annotation worksheet."
        )
    findings = []
    for name, payload in marker_sources:
        findings.extend(collect_provenance_findings(payload, path=name))
    if findings:
        detail = "; ".join(
            f"{item['path']}={item['value']!r} ({item['reason']})" for item in findings[:6]
        )
        raise ValueError(
            "Refusing to freeze gold: the adjudication payload carries non-human provenance: "
            + detail
        )
    flag_sources = marker_sources
    for name, payload in flag_sources:
        for node in _iter_dicts(payload):
            if "human_annotation_complete" in node and node["human_annotation_complete"] is not True:
                raise ValueError(
                    "Refusing to freeze gold: "
                    f"{name} declares human_annotation_complete={node['human_annotation_complete']!r}; "
                    "human gold requires true."
                )
            grade = node.get("evidence_grade")
            if grade is not None and normalize_value(grade) != HUMAN_EVIDENCE_GRADE:
                raise ValueError(
                    f"Refusing to freeze gold: {name} declares evidence_grade={grade!r}; "
                    f"human gold requires {HUMAN_EVIDENCE_GRADE!r}."
                )
    # Last, because it is the generic diagnosis: if the payload had said
    # anything about its origin — AI, generator — the specific check above
    # would already have named the real problem.
    all_annotation_rows = list(adjudicated_annotations) + list(annotations_b or [])
    undeclared = [
        row.get("dialogue_id")
        for row in all_annotation_rows
        if isinstance(row, dict) and normalize_value(row.get("provenance")) != HUMAN_PROVENANCE
    ]
    if undeclared:
        raise ValueError(
            "Refusing to freeze gold: the annotation payload does not declare a human origin. "
            f"{len(undeclared)} of {len(all_annotation_rows)} rows lack "
            f"provenance={HUMAN_PROVENANCE!r} (first: {undeclared[0]!r}). The freezer will not "
            "stamp a human gold claim that nobody declared."
        )


def _apply_resolutions(
    base_annotations: list[dict[str, Any]],
    conflicts: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply reviewer decisions to a baseline annotation list.

    ``base_annotations`` is the annotator-A baseline (index ``a_index``). Each
    conflict carries the A/B indices from the matcher. Decisions:
      - ``a`` / ``b``       -> keep that annotator's unit at A's position
      - ``edited``          -> insert the supplied ``edited_annotation``
      - ``exclude``         -> drop the unit from the frozen gold
    Previously these decisions were validated then ignored, so the frozen gold
    was whatever the baseline happened to be.
    """
    # Work on a copy keyed by index so removals never shift later positions.
    slots = {index: public_constraint(item) for index, item in enumerate(base_annotations)}
    for conflict, resolution in conflicts:
        decision = resolution["decision"]
        a_index = conflict.get("a_index")
        a = conflict.get("annotator_a")
        b = conflict.get("annotator_b")
        cid = conflict["conflict_id"]
        if decision == "exclude":
            if a_index is not None:
                slots.pop(a_index, None)
            continue
        if decision == "edited":
            edited = resolution.get("edited_annotation")
            if not isinstance(edited, dict):
                raise ValueError(f"{cid}: decision=edited requires edited_annotation")
            if a_index is None:
                a_index = max(slots, default=-1) + 1
            slots[a_index] = public_constraint(edited)
            continue
        winner = a if decision == "a" else b
        if winner is None:
            raise ValueError(f"{cid}: decision={decision} but that annotation is absent")
        if a_index is None:
            a_index = max(slots, default=-1) + 1
        slots[a_index] = public_constraint(winner)
    return [slots[key] for key in sorted(slots)]


def _agreement_gate_facts(report: dict[str, Any]) -> dict[str, Any]:
    """The part of an agreement report that decides whether gold may freeze.

    Only gate-relevant fields take part.  Free-text warnings, notes and the
    informational AC1 are deliberately left out: they may be reworded
    without changing a single verdict.
    """
    queue = [
        (
            conflict.get("conflict_id"),
            tuple(conflict.get("reasons") or []),
            conflict.get("a_index"),
            conflict.get("b_index"),
        )
        for conflict in (report.get("adjudication_queue") or [])
        if isinstance(conflict, dict)
    ]
    overall = report.get("overall") or {}
    return {
        "unit_n": report.get("unit_n"),
        "overall_kappa": overall.get("kappa"),
        "overall_ci": overall.get("ci_95_percentile_cluster_bootstrap"),
        "exit_checks": report.get("exit_checks"),
        "annotator_n": report.get("annotator_n"),
        "annotator_type": report.get("annotator_type"),
        "annotator_labels": report.get("annotator_labels"),
        "boundary_assessed": report.get("boundary_assessed"),
        "adjudication_queue": queue,
    }


def freeze_gold(
    candidate_dialogues: list[dict[str, Any]],
    adjudicated_annotations: list[dict[str, Any]],
    agreement: dict[str, Any],
    resolutions: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    gold_provenance: str = HUMAN_PROVENANCE,
    annotations_b: list[dict[str, Any]] | None = None,
    distractor_source: str | Path | None = None,
    generator_truth_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Freeze gold after independent agreement and applied human resolutions.

    The reviewer decisions in ``resolutions`` are applied to the baseline
    annotations: ``a``/``b`` select an annotator's unit, ``edited`` inserts the
    supplied ``edited_annotation``, and ``exclude`` drops the unit. Without
    this step the gate was decorative — a resolution saying "a" could freeze
    annotations taken verbatim from "b".

    ``gold_provenance`` declares where the frozen labels come from.  Only
    ``"human"`` may be frozen, and the payload behind it must not carry
    evidence of an AI / generator origin (see :func:`_assert_human_provenance`).
    The declared provenance is recorded in the frozen manifest so a later
    claim can be checked against what was actually declared.

    The freezer does not trust a supplied agreement report on its word:

    * the second annotator's annotation set is required, because an
      agreement measurement computed from one set copied twice is vacuous;
    * the report must carry content hashes of both sets, and those hashes
      must match what the freezer sees;
    * the report is recomputed from the two sets and cross-checked on every
      gate-relevant field;
    * both the input annotations and the post-adjudication gold are checked
      against the shipped generator truth **by content**, so deleting the
      generator's marker fields cannot launder its labels into human gold.

    ``distractor_source`` supplies the negative-side items (``distractors``,
    ``non_units``).  They are rule-audited dataset artifacts, not human
    annotations, and the manifest says so explicitly.
    """
    declared = normalize_value(gold_provenance)
    if declared != HUMAN_PROVENANCE:
        raise ValueError(
            f"Refusing to freeze gold: gold_provenance={gold_provenance!r} is not "
            f"{HUMAN_PROVENANCE!r}. Only human-adjudicated annotations may be frozen "
            "as human gold."
        )
    # 1. A second annotation set is a prerequisite, not a convenience.
    if annotations_b is None:
        raise ValueError(
            "Refusing to freeze gold: the second annotator's annotation set is required; "
            "agreement cannot be verified without it."
        )
    rows_a = list(adjudicated_annotations)
    rows_b = list(annotations_b)
    # 2. Bind the report to the exact annotation sets it claims to describe.
    #    A fabricated report can copy the gate numbers, but it cannot invent
    #    the content hash of annotations the freezer has never seen.
    if not isinstance(agreement, dict):
        raise ValueError("Refusing to freeze gold: the agreement report must be a JSON object")
    expected_a = annotation_content_hash(rows_a)
    expected_b = annotation_content_hash(rows_b)
    if (
        agreement.get("annotations_a_sha256") != expected_a
        or agreement.get("annotations_b_sha256") != expected_b
    ):
        raise ValueError(
            "Refusing to freeze gold: the agreement report is not bound to the annotation "
            "sets it claims to describe (annotations_a_sha256 / annotations_b_sha256 do not "
            "match the sets supplied to the freezer). Regenerate the report with agreement."
        )
    # 3. Recompute the gate from the raw sets.  Reusing the report's own
    #    bootstrap_n matters: a different resample count changes the CI lower
    #    bound and would reject a perfectly valid report.
    bootstrap_n = (agreement.get("overall") or {}).get("bootstrap_n", 1000)
    if not isinstance(bootstrap_n, int) or isinstance(bootstrap_n, bool) or bootstrap_n < 1:
        raise ValueError(
            "Refusing to freeze gold: the agreement report does not declare a usable bootstrap_n"
        )
    recomputed = agreement_report(rows_a, rows_b, bootstrap_n=bootstrap_n)
    if _agreement_gate_facts(recomputed) != _agreement_gate_facts(agreement):
        raise ValueError(
            "Refusing to freeze gold: the supplied agreement report does not match a "
            "recomputation from the two annotation sets (kappa, CI, exit checks, annotators "
            "or the adjudication queue differ)."
        )
    # 4. Two annotation sets with identical content are not an agreement
    #    measurement — they are one set copied twice, whatever the names say.
    if expected_a == expected_b:
        raise ValueError(
            "Refusing to freeze gold: the two annotation sets are byte-identical in content. "
            "Agreement between copies is vacuous; two annotators must write their own labels."
        )
    # 5. Content-level origin check on the baseline, before any laundering
    #    through note/status editing can be considered human.
    laundering_a = generator_truth_laundering(rows_a, extra_paths=generator_truth_paths)
    if laundering_a["laundered"]:
        raise ValueError(
            "Refusing to freeze gold: the annotation content is the shipped generator truth "
            f"({laundering_a['whole_set_match_path'] or laundering_a['max_overlap_path']}). "
            "It is generator truth, not human annotation."
        )
    _assert_human_provenance(
        adjudicated_annotations, agreement, resolutions, candidate_dialogues, rows_b
    )
    # Agreement needs two *independent* annotators.  One person labelling the
    # same set twice produces a perfect kappa and would otherwise certify
    # its own consistency as if it were a double-blind measurement.
    annotator_type = agreement.get("annotator_type")
    annotator_n = agreement.get("annotator_n")
    if annotator_type != HUMAN_PROVENANCE:
        raise ValueError(
            "Refusing to freeze gold: the agreement report declares annotator_type="
            f"{annotator_type!r}; human gold requires {HUMAN_PROVENANCE!r}. "
            "An unknown origin is never promoted to human."
        )
    if not isinstance(annotator_n, int) or annotator_n < 2:
        raise ValueError(
            "Refusing to freeze gold: the agreement report covers "
            f"{annotator_n!r} distinct annotator(s); agreement between two "
            "independent annotators requires at least 2."
        )
    # The third reviewer must not be one of the two annotators.  Otherwise a
    # single person could label the set twice, invent a disagreement, and
    # then adjudicate their own conflict.
    annotator_labels = {
        str(label) for label in (agreement.get("annotator_labels") or []) if label
    }
    self_reviewed = sorted({
        str(row.get("reviewer"))
        for row in resolutions
        if row.get("reviewer") and str(row.get("reviewer")) in annotator_labels
    })
    if self_reviewed:
        raise ValueError(
            "Refusing to freeze gold: the third reviewer is also an annotator "
            f"({self_reviewed}); the adjudication must be done by someone who did "
            "not produce either annotation set."
        )
    failed = [name for name in CORE_EXIT_CHECKS if not agreement.get("exit_checks", {}).get(name)]
    if failed:
        if "boundary_gte_0_75" in failed and not agreement.get("boundary_assessed", False):
            raise ValueError(
                "Agreement exit checks not satisfied: ['boundary_gte_0_75']. "
                + str(agreement.get("boundary_note", ""))
            )
        raise ValueError(f"Agreement exit checks not satisfied: {failed}")
    pending_ids = {row["conflict_id"] for row in agreement.get("adjudication_queue", []) if row.get("status") == "pending"}
    resolution_by_id = {row.get("conflict_id"): row for row in resolutions}
    if pending_ids != set(resolution_by_id):
        raise ValueError(f"Resolution IDs do not match pending conflicts: pending={sorted(pending_ids)}, resolutions={sorted(resolution_by_id)}")
    conflict_by_id = {row["conflict_id"]: row for row in agreement.get("adjudication_queue", [])}
    for conflict_id, resolution in resolution_by_id.items():
        if not resolution.get("reviewer") or not resolution.get("rationale"):
            raise ValueError(f"{conflict_id}: reviewer and rationale are required")
        if resolution.get("decision") not in {"a", "b", "edited", "exclude"}:
            raise ValueError(f"{conflict_id}: invalid decision")
    # Baseline annotations are the pre-resolution state. Apply every decision
    # on a fresh copy so a failed freeze never mutates caller data. The baseline
    # is the annotator-A record; B's units enter only when a resolution selects
    # them. Without this, a resolution saying "b" could freeze A's annotation.
    from collections import defaultdict
    conflicts_by_dialogue: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for conflict_id, resolution in resolution_by_id.items():
        conflict = conflict_by_id[conflict_id]
        conflicts_by_dialogue[conflict["dialogue_id"]].append((conflict, resolution))
    resolved = []
    for row in adjudicated_annotations:
        base = [public_constraint(item) for item in row_units(row)]
        applied = _apply_resolutions(base, conflicts_by_dialogue.get(row["dialogue_id"], []))
        resolved.append({"dialogue_id": row["dialogue_id"], "annotations": applied})
    rows_by_id = {row["dialogue_id"]: row for row in resolved}
    if set(rows_by_id) != {row["dialogue_id"] for row in candidate_dialogues}:
        raise ValueError("Adjudicated annotations must cover exactly every dialogue")
    # The adjudications themselves must not turn a human set back into a copy
    # of generator truth: the post-resolution gold is what actually ships.
    resolved_rows = [
        {"dialogue_id": did, "annotations": row["annotations"]}
        for did, row in rows_by_id.items()
    ]
    laundering_gold = generator_truth_laundering(
        resolved_rows, extra_paths=generator_truth_paths
    )
    if laundering_gold["laundered"]:
        raise ValueError(
            "Refusing to freeze gold: the post-adjudication gold content is the shipped "
            f"generator truth ({laundering_gold['whole_set_match_path'] or laundering_gold['max_overlap_path']}). "
            "Adjudication cannot turn generator labels into human gold."
        )
    # Negative-side items (distractors / non-units) are rule-audited dataset
    # artifacts, not human annotations.  They must be carried explicitly:
    # without them FMR is undefined and the FAA denominator silently shrinks.
    distractor_path = Path(distractor_source) if distractor_source is not None else None
    distractors_by_id: dict[str, dict[str, Any]] = {}
    if distractor_path is not None:
        if not distractor_path.is_file():
            raise ValueError(
                f"Refusing to freeze gold: distractor_source {distractor_path} does not exist"
            )
        for row in read_jsonl(distractor_path):
            distractors_by_id[row["dialogue_id"]] = {
                "distractors": row.get("distractors") or [],
                "non_units": row.get("non_units") or [],
            }
        missing = [
            dialogue["dialogue_id"] for dialogue in candidate_dialogues
            if dialogue["dialogue_id"] not in distractors_by_id
        ]
        if missing:
            raise ValueError(
                f"Refusing to freeze gold: distractor_source does not cover {sorted(missing)}"
            )
    target = Path(output_dir)
    gold_path = target / "gold_v1.jsonl"
    manifest_path = target / "gold_manifest.json"
    if gold_path.exists() or manifest_path.exists():
        raise FileExistsError("Gold is immutable; create a new version directory instead of overwriting")
    gold_rows = []
    for dialogue in candidate_dialogues:
        annotations = rows_by_id[dialogue["dialogue_id"]].get("annotations", [])
        report = validate_output({"constraints": annotations}, dialogue["turns"])
        if not report.valid:
            raise ValueError(f"{dialogue['dialogue_id']}: adjudicated output invalid: {report.to_dict()}")
        gold_rows.append({
            "dialogue_id": dialogue["dialogue_id"],
            "status": "human_adjudicated_gold",
            "gold_provenance": declared,
            "gold_version": "v1.0",
            "turns": dialogue["turns"],
            # ``candidate_truth`` is the scorer-facing key: scoring reads
            # dialogue["candidate_truth"].  Without it the frozen gold could
            # not be scored by the benchmark's own scorer.  The key name is a
            # file format, not a provenance claim — provenance lives in the
            # fields above.
            "candidate_truth": annotations,
            "constraints": annotations,
            # Negative-side items travel with the gold so the scorer can
            # compute FMR and use the full forget-set denominator.
            "distractors": distractors_by_id.get(dialogue["dialogue_id"], {}).get("distractors", []),
            "non_units": distractors_by_id.get(dialogue["dialogue_id"], {}).get("non_units", []),
        })
    write_jsonl(gold_path, gold_rows)
    manifest = {
        "gold_version": "v1.0",
        "status": "human_adjudicated_gold_frozen",
        "gold_provenance": declared,
        "dialogue_n": len(gold_rows),
        "resolution_n": len(resolutions),
        "reviewers": sorted({row["reviewer"] for row in resolutions}),
        "annotator_n": agreement.get("annotator_n"),
        "annotator_type": agreement.get("annotator_type"),
        "annotator_labels": agreement.get("annotator_labels"),
        "boundary_assessed": agreement.get("boundary_assessed"),
        "gold_sha256": sha256_file(gold_path),
        "agreement_summary": {key: value for key, value in agreement.items() if key != "adjudication_queue"},
        "agreement_recomputed_by_freezer": True,
        "suspect_self_agreement": agreement.get("suspect_self_agreement"),
        "content_laundering_check": {
            "baseline_annotations": laundering_a,
            "frozen_gold": laundering_gold,
        },
        "distractor_set": "present" if distractor_path is not None else "absent",
        "distractor_source": str(distractor_path) if distractor_path is not None else None,
        "distractor_source_sha256": sha256_file(distractor_path) if distractor_path is not None else None,
        "distractor_n": sum(len(row["distractors"]) for row in gold_rows),
        "distractor_provenance": (
            "dataset_truth_not_human_annotated" if distractor_path is not None else "none_attached"
        ),
        "distractor_note": (
            "Negative-side items are carried from the rule-audited dataset; they are NOT human "
            "annotations. The human_gold claim covers the positive constraint labels only."
            if distractor_path is not None
            else "No negative-side items were attached; FMR is undefined for this gold set."
        ),
        "immutability": "Any content change requires a new gold version; never overwrite this file",
    }
    write_json(manifest_path, manifest)
    return manifest
