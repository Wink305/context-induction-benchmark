"""Human adjudication gate and immutable gold freezing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import sha256_file, write_json, write_jsonl
from .validation import validate_output

CORE_EXIT_CHECKS = (
    "overall_kappa_gte_0_70", "overall_ci_lower_gte_0_60", "per_type_pass",
    "interval_gte_0_80", "boundary_gte_0_75",
)


def freeze_gold(
    candidate_dialogues: list[dict[str, Any]],
    adjudicated_annotations: list[dict[str, Any]],
    agreement: dict[str, Any],
    resolutions: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze only after independent agreement and explicit human resolutions.

    This function never chooses a winning annotation. `adjudicated_annotations`
    must already contain the third reviewer's final decisions.
    """
    failed = [name for name in CORE_EXIT_CHECKS if not agreement.get("exit_checks", {}).get(name)]
    if failed:
        raise ValueError(f"Agreement exit checks not satisfied: {failed}")
    pending_ids = {row["conflict_id"] for row in agreement.get("adjudication_queue", []) if row.get("status") == "pending"}
    resolution_by_id = {row.get("conflict_id"): row for row in resolutions}
    if pending_ids != set(resolution_by_id):
        raise ValueError(f"Resolution IDs do not match pending conflicts: pending={sorted(pending_ids)}, resolutions={sorted(resolution_by_id)}")
    for conflict_id, resolution in resolution_by_id.items():
        if not resolution.get("reviewer") or not resolution.get("rationale"):
            raise ValueError(f"{conflict_id}: reviewer and rationale are required")
        if resolution.get("decision") not in {"a", "b", "edited", "exclude"}:
            raise ValueError(f"{conflict_id}: invalid decision")
    rows_by_id = {row["dialogue_id"]: row for row in adjudicated_annotations}
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
