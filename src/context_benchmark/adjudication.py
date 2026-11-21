"""Human adjudication gate and immutable gold freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .io import sha256_file, write_json, write_jsonl
from .provenance import collect_generator_findings, collect_provenance_findings, normalize_value
from .validation import public_constraint, row_units, validate_output

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
    undeclared = [
        row.get("dialogue_id")
        for row in adjudicated_annotations
        if isinstance(row, dict) and normalize_value(row.get("provenance")) != HUMAN_PROVENANCE
    ]
    if undeclared:
        raise ValueError(
            "Refusing to freeze gold: the annotation payload does not declare a human origin. "
            f"{len(undeclared)} of {len(adjudicated_annotations)} rows lack "
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


def freeze_gold(
    candidate_dialogues: list[dict[str, Any]],
    adjudicated_annotations: list[dict[str, Any]],
    agreement: dict[str, Any],
    resolutions: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    gold_provenance: str = HUMAN_PROVENANCE,
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
    """
    declared = normalize_value(gold_provenance)
    if declared != HUMAN_PROVENANCE:
        raise ValueError(
            f"Refusing to freeze gold: gold_provenance={gold_provenance!r} is not "
            f"{HUMAN_PROVENANCE!r}. Only human-adjudicated annotations may be frozen "
            "as human gold."
        )
    _assert_human_provenance(adjudicated_annotations, agreement, resolutions, candidate_dialogues)
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
        "immutability": "Any content change requires a new gold version; never overwrite this file",
    }
    write_json(manifest_path, manifest)
    return manifest
