"""Strict shape plus field-level semantic validation (no third-party dependency)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .io import canonical_json
from .ontology import CATEGORIES, SCOPES, TRAP_KINDS, TYPE_TO_CODE

REQUIRED_FIELDS = {
    "id", "type", "text", "evidence_turns", "valid_from", "valid_to",
    "scope", "superseded_by", "category",
}
BASE_FIELDS = REQUIRED_FIELDS

# The root object carries the constraint list and, optionally, the trap flags.
# ``trap_flags`` is optional on purpose: a prediction produced before this
# contract existed is still a valid prediction, and the trap axis reports
# "not evaluated" for it rather than scoring it zero.
ALLOWED_ROOT_KEYS = {"constraints", "trap_flags"}


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
    # Problems with an individual trap flag. They are deliberately kept out of
    # ``issues``: a malformed flag must not invalidate the constraints, because
    # one broken annotation in an extra field would otherwise zero the whole
    # dialogue on axes A and C. The trap axis reads these separately.
    trap_issues: list[Issue] = field(default_factory=list)

    @property
    def schema_error_n(self) -> int:
        return sum(issue.code.startswith("schema") or issue.code == "json_parse" for issue in self.issues)

    @property
    def semantic_error_n(self) -> int:
        return len(self.issues) - self.schema_error_n

    @property
    def trap_error_n(self) -> int:
        return len(self.trap_issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "constraint_n": self.constraint_n,
            "schema_error_n": self.schema_error_n,
            "semantic_error_n": self.semantic_error_n,
            "trap_error_n": self.trap_error_n,
            "issues": [asdict(issue) for issue in self.issues],
            "trap_issues": [asdict(issue) for issue in self.trap_issues],
        }


def parse_output(raw: str) -> tuple[Any | None, ValidationReport]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, ValidationReport(False, [Issue("json_parse", str(exc), "$")])
    return value, ValidationReport(True, [])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_trap_flags(
    flags: Any,
    turns: list[dict[str, Any]] | None = None,
    constraint_ids: set[str] | None = None,
) -> list[Issue]:
    """Validate the optional ``trap_flags`` array.

    A flag is a claim about the dialogue: "this kind of trap is here, at
    these turns, for this reason". Every part of the claim must be checkable
    by a program, because the trap axis is scored mechanically — no judge is
    involved.
    """
    if flags is None:
        return []
    if not isinstance(flags, list):
        return [Issue("schema_trap_flags", "trap_flags must be an array", "$.trap_flags")]
    issues: list[Issue] = []
    turn_map = {turn["turn"]: turn for turn in (turns or [])}
    for index, flag in enumerate(flags):
        path = f"$.trap_flags[{index}]"
        if not isinstance(flag, dict):
            issues.append(Issue("schema_trap_flag", "trap flag must be an object", path))
            continue
        kind = flag.get("kind")
        if kind not in TRAP_KINDS:
            issues.append(Issue(
                "schema_trap_kind",
                f"kind must be one of {', '.join(TRAP_KINDS)}",
                path,
            ))
        target = flag.get("target_turns")
        if not isinstance(target, list) or not target or any(not _is_int(turn) for turn in target):
            issues.append(Issue(
                "schema_trap_turns",
                "target_turns must be a non-empty integer array",
                path,
            ))
        elif turn_map:
            for turn in target:
                if turn not in turn_map:
                    issues.append(Issue(
                        "semantic_trap_turn_missing", f"turn {turn} does not exist", path
                    ))
                elif (
                    turn_map[turn]["speaker"] != "user"
                    and kind != "non_user_claim"
                ):
                    # A trap usually sits in a user turn, but the agent's own
                    # claims are the classic non_user_claim trap and they sit
                    # in an agent turn. Any other kind pointing at an agent
                    # turn is a flag the model cannot justify.
                    issues.append(Issue(
                        "semantic_trap_agent_turn_kind",
                        f"turn {turn} is not user-side; only a non_user_claim trap may point at one",
                        path,
                    ))
        reason = flag.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            issues.append(Issue(
                "schema_trap_reason", "reason must be a non-empty string", path
            ))
        ref = flag.get("constraint_ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref:
                issues.append(Issue(
                    "schema_trap_ref", "constraint_ref must be a string or null", path
                ))
            elif constraint_ids and ref not in constraint_ids:
                issues.append(Issue(
                    "semantic_trap_ref_missing",
                    f"constraint_ref {ref} is not one of this output's constraints",
                    path,
                ))
    return issues


def validate_output(value: Any, turns: list[dict[str, Any]] | None = None) -> ValidationReport:
    issues: list[Issue] = []
    if (
        not isinstance(value, dict)
        or not set(value).issubset(ALLOWED_ROOT_KEYS)
        or "constraints" not in value
        or not isinstance(value.get("constraints"), list)
    ):
        return ValidationReport(
            False,
            [Issue("schema_root", "root must contain a constraints array and may contain trap_flags", "$")],
        )
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
    trap_issues = validate_trap_flags(value.get("trap_flags"), turns, set(ids))
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
    return ValidationReport(not issues, issues, len(constraints), trap_issues=trap_issues)


def public_constraint(item: dict[str, Any]) -> dict[str, Any]:
    """Strip generator-only metadata before applying the model-output contract.

    Raises a diagnostic :class:`ValueError` when a required field is
    missing.  A bare ``KeyError`` used to escape from here, which told the
    caller neither which dialogue nor which unit was malformed.
    """
    missing = [key for key in BASE_FIELDS if key not in item]
    if missing:
        raise ValueError(
            f"constraint {item.get('id', '<no id>')!r} is missing required "
            f"field(s): {sorted(missing)}"
        )
    return {key: item[key] for key in BASE_FIELDS}


# The fields that decide what a unit *means*.  Deliberately excludes the
# bookkeeping and provenance metadata (``id``, ``note``, ``candidate_only``,
# ``status``, ``provenance``): a content fingerprint must survive an attempt
# to launder the origin of a label by editing those fields.
CONTENT_FIELDS = (
    "type",
    "scope",
    "category",
    "text",
    "evidence_turns",
    "valid_from",
    "valid_to",
    "superseded_by",
    "rule_tags",
)


def annotation_content_hash(rows: list[dict[str, Any]]) -> str:
    """A canonical fingerprint of the annotation **content** in these rows.

    Used for two things:

    * binding an agreement report to the annotations it was computed from,
      so a hand-written report with fabricated gate results cannot be used
      to freeze anything;
    * comparing a frozen gold against the shipped generator truth, so
      generator labels cannot be laundered into a ``human_gold`` claim by
      editing the marker fields.

    Only :data:`CONTENT_FIELDS` take part, so rewriting ``note`` or
    ``status`` does not change the fingerprint — but changing a label does.
    """
    payload = []
    for row in rows:
        units = row_units(row)
        payload.append(
            {
                "dialogue_id": row.get("dialogue_id"),
                "units": [
                    {field: unit.get(field) for field in CONTENT_FIELDS}
                    for unit in units
                ],
            }
        )
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def dialogue_content_hashes(rows: list[dict[str, Any]]) -> dict[str, str]:
    """One content fingerprint per dialogue.

    Used to tell *how much* of an annotation set is byte-identical to the
    shipped generator truth.  The whole-set fingerprint below cannot say
    that: a set that differs in one dialogue out of forty still looks
    completely different as a whole.
    """
    fingerprints: dict[str, str] = {}
    for row in rows:
        units = row_units(row)
        payload = [{field: unit.get(field) for field in CONTENT_FIELDS} for unit in units]
        fingerprints[row.get("dialogue_id")] = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
    return fingerprints


def truth_rows_as_annotations(truth_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape a truth file's ``candidate_truth`` into annotation rows.

    The generator's truth file and a parsed annotation file hold the same
    units under different keys.  This makes them comparable by
    :func:`annotation_content_hash` without pretending they are the same
    format.
    """
    return [
        {
            "dialogue_id": row.get("dialogue_id"),
            "annotations": row.get("candidate_truth") or [],
        }
        for row in truth_rows
    ]


def row_units(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The units carried by one dialogue's annotation row.

    ``annotations`` is the contract key — the annotation worksheet uses it,
    and so do the agreement gate and the adjudication gate.  ``constraints``
    is the legacy alias; both are accepted, so an older or hand-authored
    artefact still loads instead of silently agreeing on zero units.
    """
    units = row.get("annotations")
    if not isinstance(units, list):
        units = row.get("constraints")
    return units if isinstance(units, list) else []


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
