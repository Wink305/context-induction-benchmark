"""Human adjudication gate and immutable gold freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import sha256_file, write_json, write_jsonl
from .validation import public_constraint, validate_output

CORE_EXIT_CHECKS = (
    "overall_kappa_gte_0_70", "overall_ci_lower_gte_0_60", "per_type_pass",
    "interval_gte_0_80", "boundary_gte_0_75", "extraction_match_gte_0_95",
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
) -> dict[str, Any]:
    """Freeze gold after independent agreement and applied human resolutions.

    The reviewer decisions in ``resolutions`` are applied to the baseline
    annotations: ``a``/``b`` select an annotator's unit, ``edited`` inserts the
    supplied ``edited_annotation``, and ``exclude`` drops the unit. Without
    this step the gate was decorative — a resolution saying "a" could freeze
    annotations taken verbatim from "b".
    """
    failed = [name for name in CORE_EXIT_CHECKS if not agreement.get("exit_checks", {}).get(name)]
    if failed:
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
        base = [public_constraint(item) for item in row.get("annotations", [])]
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
            "gold_version": "v1.0",
            "turns": dialogue["turns"],
            "constraints": annotations,
        })
    write_jsonl(gold_path, gold_rows)
    manifest = {
        "gold_version": "v1.0",
        "status": "human_adjudicated_gold_frozen",
        "dialogue_n": len(gold_rows),
        "resolution_n": len(resolutions),
        "reviewers": sorted({row["reviewer"] for row in resolutions}),
        "gold_sha256": sha256_file(gold_path),
        "agreement_summary": {key: value for key, value in agreement.items() if key != "adjudication_queue"},
        "immutability": "Any content change requires a new gold version; never overwrite this file",
    }
    write_json(manifest_path, manifest)
    return manifest
