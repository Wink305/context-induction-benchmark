"""Strict shape plus field-level semantic validation (no third-party dependency)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .ontology import CATEGORIES, SCOPES, TYPE_TO_CODE

REQUIRED_FIELDS = {
    "id", "type", "text", "evidence_turns", "valid_from", "valid_to",
    "scope", "superseded_by", "category",
}
BASE_FIELDS = REQUIRED_FIELDS


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str


@dataclass
class ValidationReport:
    valid: bool
    issues: list[Issue]
    constraint_n: int = 0

    @property
    def schema_error_n(self) -> int:
        return sum(issue.code.startswith("schema") or issue.code == "json_parse" for issue in self.issues)

    @property
    def semantic_error_n(self) -> int:
        return len(self.issues) - self.schema_error_n

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "constraint_n": self.constraint_n,
            "schema_error_n": self.schema_error_n,
            "semantic_error_n": self.semantic_error_n,
            "issues": [asdict(issue) for issue in self.issues],
        }


def parse_output(raw: str) -> tuple[Any | None, ValidationReport]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, ValidationReport(False, [Issue("json_parse", str(exc), "$")])
    return value, ValidationReport(True, [])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_output(value: Any, turns: list[dict[str, Any]] | None = None) -> ValidationReport:
    issues: list[Issue] = []
    if not isinstance(value, dict) or set(value) != {"constraints"} or not isinstance(value.get("constraints"), list):
        return ValidationReport(False, [Issue("schema_root", "root must contain only a constraints array", "$")])
    constraints = value["constraints"]
    turn_map = {turn["turn"]: turn for turn in (turns or [])}
    ids: set[str] = set()
    for index, item in enumerate(constraints):
        path = f"$.constraints[{index}]"
        if not isinstance(item, dict):
            issues.append(Issue("schema_item", "constraint must be an object", path))
            continue
        missing = REQUIRED_FIELDS - set(item)
        extra = set(item) - REQUIRED_FIELDS
        if missing or extra:
            issues.append(Issue("schema_fields", f"missing={sorted(missing)}, extra={sorted(extra)}", path))
            continue
        cid = item["id"]
        if not isinstance(cid, str) or not cid.startswith("c") or not cid[1:].isdigit():
            issues.append(Issue("schema_id", "id must match c<number>", f"{path}.id"))
        elif cid in ids:
            issues.append(Issue("semantic_duplicate_id", f"duplicate id {cid}", f"{path}.id"))
        ids.add(cid)
        if item["type"] not in TYPE_TO_CODE:
            issues.append(Issue("schema_type", "invalid type", f"{path}.type"))
        if item["category"] not in CATEGORIES:
            issues.append(Issue("schema_category", "invalid category", f"{path}.category"))
        if item["scope"] not in SCOPES:
            issues.append(Issue("schema_scope", "invalid scope", f"{path}.scope"))
        if not isinstance(item["text"], str) or not item["text"].strip():
            issues.append(Issue("schema_text", "text must be non-empty", f"{path}.text"))
        evidence = item["evidence_turns"]
        if not isinstance(evidence, list) or not evidence or any(not _is_int(turn) for turn in evidence):
            issues.append(Issue("schema_evidence", "evidence_turns must be non-empty integer array", f"{path}.evidence_turns"))
            evidence = []
        if len(set(evidence)) != len(evidence):
            issues.append(Issue("semantic_duplicate_evidence", "duplicate evidence turn", f"{path}.evidence_turns"))
        if item["type"] == "implicit" and len(set(evidence)) < 3:
            issues.append(Issue("semantic_implicit_evidence", "implicit constraint requires >=3 independent turns", f"{path}.evidence_turns"))
        valid_from = item["valid_from"]
        valid_to = item["valid_to"]
        if not _is_int(valid_from) or valid_from < 1:
            issues.append(Issue("schema_valid_from", "valid_from must be positive integer", f"{path}.valid_from"))
        if valid_to is not None and (not _is_int(valid_to) or valid_to < 1):
            issues.append(Issue("schema_valid_to", "valid_to must be positive integer or null", f"{path}.valid_to"))
        if _is_int(valid_from) and _is_int(valid_to) and valid_to < valid_from:
            issues.append(Issue("semantic_interval", "valid_to precedes valid_from", path))
        if item["type"] == "superseded" and valid_to is None:
            issues.append(Issue("semantic_superseded_open", "superseded constraint requires valid_to", f"{path}.valid_to"))
        if item["type"] != "superseded" and item["superseded_by"] is not None:
            issues.append(Issue("semantic_superseded_link", "only superseded items may have superseded_by", f"{path}.superseded_by"))
        if item["type"] == "one_off" and item["scope"] == "global":
            issues.append(Issue("semantic_one_off_global", "one_off cannot have global scope", f"{path}.scope"))
        if item["type"] == "standing" and item["scope"] == "task":
            issues.append(Issue("semantic_standing_task", "task-scoped rule should be one_off (R1)", f"{path}.scope"))
        if turn_map:
            for turn in evidence:
                if turn not in turn_map:
                    issues.append(Issue("semantic_evidence_missing", f"turn {turn} does not exist", f"{path}.evidence_turns"))
                elif turn_map[turn]["speaker"] != "user":
                    issues.append(Issue("semantic_evidence_not_user", f"turn {turn} is not user-side", f"{path}.evidence_turns"))
            for field, turn in (("valid_from", valid_from), ("valid_to", valid_to)):
                if turn is not None and _is_int(turn) and turn not in turn_map:
                    issues.append(Issue("semantic_turn_missing", f"turn {turn} does not exist", f"{path}.{field}"))
            if _is_int(valid_from) and valid_from not in evidence:
                issues.append(Issue("semantic_valid_from_evidence", "valid_from must occur in evidence_turns", f"{path}.valid_from"))
    by_id = {item.get("id"): item for item in constraints if isinstance(item, dict)}
    for index, item in enumerate(constraints):
        if not isinstance(item, dict):
            continue
        target_id = item.get("superseded_by")
        if target_id is not None:
            path = f"$.constraints[{index}].superseded_by"
            if target_id == item.get("id"):
                issues.append(Issue("semantic_self_supersession", "constraint cannot supersede itself", path))
            elif target_id not in by_id:
                issues.append(Issue("semantic_superseded_target", f"missing target {target_id}", path))
            else:
                target_from = by_id[target_id].get("valid_from")
                if _is_int(target_from) and _is_int(item.get("valid_to")) and target_from <= item["valid_to"]:
                    issues.append(Issue("semantic_supersession_order", "replacement must begin after old validity ends", path))
    return ValidationReport(not issues, issues, len(constraints))


def public_constraint(item: dict[str, Any]) -> dict[str, Any]:
    """Strip generator-only metadata before applying the model-output contract."""
    return {key: item[key] for key in BASE_FIELDS}


def validate_candidate_dialogue(dialogue: dict[str, Any]) -> ValidationReport:
    issues: list[Issue] = []
    turns = dialogue.get("turns", [])
    indices = [turn.get("turn") for turn in turns]
    if indices != list(range(1, len(turns) + 1)):
        issues.append(Issue("semantic_turn_sequence", "turns must be contiguous and 1-based", "$.turns"))
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "agent"
        if turn.get("speaker") != expected:
            issues.append(Issue("semantic_speaker_sequence", f"expected {expected}", f"$.turns[{index}].speaker"))
    output_report = validate_output(
        {"constraints": [public_constraint(item) for item in dialogue.get("candidate_truth", [])]}, turns
    )
    issues.extend(output_report.issues)
    for distractor in dialogue.get("distractors", []):
        for turn in distractor.get("evidence_turns", []):
            if turn not in {item["turn"] for item in turns}:
                issues.append(Issue("semantic_distractor_turn", f"missing turn {turn}", "$.distractors"))
    return ValidationReport(not issues, issues, output_report.constraint_n)
