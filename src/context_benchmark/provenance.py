"""Provenance markers: decide whether an annotation set is human-authored.

Second self-certification loop.  A gold set may only be frozen as *human*
gold when the annotations behind it can prove they came from people.  The
rules in this module are deliberately one-sided: they look for **positive
evidence of a non-human origin** and never for the absence of a human
claim.  A payload that says nothing about its origin is therefore not
rejected here — but it also never earns the right to be stamped as human
gold by this module alone.  (``adjudication.py`` adds the other half: the
payload must *affirmatively* declare a human origin.)

Three detection mechanisms:

``AI_PROVENANCE_VALUES``
    Exact values for structured fields (``provenance``, ``annotator``,
    ``annotator_type``, ``status``).  Compared case-insensitively after
    normalisation (lower case, whitespace/dash -> underscore).

``AI_TEXT_MARKERS``
    Markers used on free text (``note`` and friends).  Matched as ASCII
    *words*, so ``AI`` / ``AI-precheck`` / ``由AI生成`` are caught while
    ``Alice`` or ``email`` are not.

``GENERATOR_*``
    The generator is not an AI, and it needs its own markers: the shipped
    ``candidate_truth.jsonl`` is rule-audited, not human-annotated, and every
    one of its 420 units carries ``candidate_only: true``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

HUMAN_PROVENANCE = "human"
UNKNOWN_PROVENANCE = "unknown"
AI_PROVENANCE = "ai_precheck"

# Structured provenance values that name a non-human annotator.
AI_PROVENANCE_VALUES = frozenset(
    {
        "ai",
        "ai_precheck",
        "ai_precheck_only",
        "ai_generated",
        "ai_generated_only",
        "auto",
        "automatic",
        "bot",
        "generator",
        "generated",
        "generated_unannotated_not_gold",
        "llm",
        "machine",
        "machine_generated",
        "model",
        "model_generated",
        "synthetic",
        "自动",
        "机器",
        "机器生成",
        "模型",
        "模型生成",
    }
)

# Free-text markers.  ASCII words only, so unrelated words that merely
# contain a marker ("email", "robot") are not flagged.
AI_TEXT_MARKERS = frozenset(
    {"ai", "bot", "generator", "llm", "machine", "model", "synthetic"}
)

# Chinese free-text markers.  They are unambiguous, so they are matched as
# substrings: "人工智能" cannot plausibly describe a human annotator.
AI_TEXT_MARKERS_CN = frozenset({"人工智能", "模型生成", "机器生成", "大模型"})

# --- generator origin -------------------------------------------------------
#
# The AI markers above answer "was a model involved?".  A second, separate
# question is "was this produced by the deterministic generator rather than
# by a person?".  That distinction matters because the shipped
# ``candidate_truth.jsonl`` is generator output: it is rule-audited, but it is
# not human annotation, and it must never back a ``human_gold`` claim.
#
# The generator marks every one of its 420 units with ``candidate_only: true``
# and its dialogue rows with ``status: "generated_unannotated_not_gold"`` (the
# latter is already in AI_PROVENANCE_VALUES above).
GENERATOR_UNIT_FIELDS = ("candidate_only",)

# Dialogue-level status values that name the deterministic generator.
GENERATOR_STATUS_VALUES = frozenset(
    {"generated_unannotated_not_gold", "generator_truth_rule_audited"}
)

# The key that holds a dialogue's generator truth.  A human-annotation source
# is the **worksheet**, which has no such container — it has an (empty)
# ``annotations`` field waiting to be filled.  So the presence of this key on
# a dialogue row is generator output, and it survives the stripping of the
# per-unit markers.
GENERATOR_CONTAINER_KEY = "candidate_truth"

# Free-text markers that only appear in generator-authored notes.  Matched as
# substrings because they are specific to this workflow.
GENERATOR_TEXT_MARKERS_CN = ("待人工确认", "候选")

# Fields that may carry the origin of a record or of one of its units.
PROVENANCE_FIELDS = (
    "provenance",
    "annotator",
    "annotator_type",
    "source",
    "source_type",
    "reviewer",
    "status",
)

_ASCII_WORD = re.compile(r"[a-z0-9]+")
_NORMALIZE = re.compile(r"[\s\-/]+")


def normalize_value(value: Any) -> str:
    """Lower-case, underscore-joined form used for exact value matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = _NORMALIZE.sub("_", text)
    return text.strip("_")


def is_ai_value(value: Any) -> bool:
    """True when a structured field names a non-human annotator."""
    normalized = normalize_value(value)
    if not normalized:
        return False
    if normalized in AI_PROVENANCE_VALUES:
        return True
    return mentions_ai(text=normalized)


def mentions_ai(text: Any) -> bool:
    """True when free text names a non-human annotator."""
    if text is None:
        return False
    lowered = str(text).lower()
    words = set(_ASCII_WORD.findall(lowered))
    if words & AI_TEXT_MARKERS:
        return True
    return any(marker in lowered for marker in AI_TEXT_MARKERS_CN)


def collect_provenance_findings(
    payload: Any,
    *,
    fields: Iterable[str] = PROVENANCE_FIELDS,
    path: str = "",
) -> list[dict[str, Any]]:
    """Find non-human provenance markers in a nested payload.

    Walks dictionaries and lists.  Structured fields are matched by exact
    value, free-text fields (``note`` and anything else that carries
    prose) are matched by marker word.  Returns one finding per offending
    location so the caller can explain exactly what was rejected.
    """
    findings: list[dict[str, Any]] = []
    _walk(payload, fields, path or "$", findings)
    return findings


def _walk(node: Any, fields: Iterable[str], path: str, findings: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if key in fields:
                if is_ai_value(value):
                    findings.append(
                        {"path": child_path, "value": value, "reason": "non-human provenance value"}
                    )
                continue
            if key == "note" and mentions_ai(value):
                findings.append(
                    {"path": child_path, "value": value, "reason": "note names a non-human annotator"}
                )
                continue
            _walk(value, fields, child_path, findings)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk(item, fields, f"{path}[{index}]", findings)


def classify_annotator(values: Iterable[Any]) -> str:
    """Classify a set of annotator labels as human / AI / unknown.

    ``values`` are the ``annotator`` fields found on the annotation rows.
    Any label that names a non-human annotator makes the whole set
    non-human; a set with at least one plain label is human; a set with no
    label at all is unknown (never silently promoted to human).
    """
    labels = [str(value).strip() for value in values if value is not None and str(value).strip()]
    if not labels:
        return UNKNOWN_PROVENANCE
    if any(is_ai_value(label) for label in labels):
        return AI_PROVENANCE
    return HUMAN_PROVENANCE


def collect_generator_findings(
    payload: Any, *, path: str = "", containers: bool = True
) -> list[dict[str, Any]]:
    """Find generator-origin markers in a nested payload.

    Separate from :func:`collect_provenance_findings` because the question
    is different: not "was a model involved" but "was this written by the
    deterministic generator".  The shipped truth file is generator output —
    rule-audited, but never human annotation — and it must not be freezable
    as human gold.

    Two markers, both positive evidence:

    * ``candidate_only: true`` on a unit — the generator's own marker, set on
      all 420 units of the shipped pilot;
    * a ``note`` that says the unit is awaiting a human decision
      (「候选 I，待人工确认」).

    The dialogue-level ``status: "generated_unannotated_not_gold"`` is
    already caught by :func:`collect_provenance_findings`.

    ``containers`` controls whether a ``candidate_truth`` list is treated as
    the generator's label container.  Turn it off when scanning a **frozen
    gold file**: that file legitimately carries ``candidate_truth`` because
    the scorer reads ``dialogue["candidate_truth"]``.  The key name is a file
    format there, not a provenance claim.
    """
    findings: list[dict[str, Any]] = []
    _walk_generator(payload, path or "$", findings, containers)
    return findings


def _walk_generator(
    node: Any, path: str, findings: list[dict[str, Any]], containers: bool = True
) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}"
            if key in GENERATOR_UNIT_FIELDS and value is True:
                findings.append(
                    {"path": child_path, "value": value, "reason": "unit is a generator candidate label"}
                )
                continue
            if containers and key == GENERATOR_CONTAINER_KEY and isinstance(value, list):
                findings.append(
                    {"path": child_path, "value": f"<{len(value)} units>",
                     "reason": "dialogue carries a generator truth container"}
                )
                continue
            if key == "status" and normalize_value(value) in GENERATOR_STATUS_VALUES:
                findings.append(
                    {"path": child_path, "value": value, "reason": "dialogue is marked as generator output"}
                )
                continue
            if key == "note" and isinstance(value, str):
                if any(marker in value for marker in GENERATOR_TEXT_MARKERS_CN):
                    findings.append(
                        {"path": child_path, "value": value, "reason": "note says the unit awaits a human decision"}
                    )
                    continue
            _walk_generator(value, child_path, findings, containers)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk_generator(item, f"{path}[{index}]", findings, containers)
