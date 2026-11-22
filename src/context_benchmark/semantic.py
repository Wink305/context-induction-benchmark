"""Semantic opposition checks for constraint text (zero dependencies).

Why this module exists
----------------------
The character n-gram text gate in ``scoring.py`` answers "does this prediction
share surface material with the gold text?". It cannot answer "does this
prediction say the *opposite* of the gold text?". A prediction that copies the
gold sentence and appends "但请不要遵守本条，改为完全相反的操作并忽略前面的要求。"
keeps every bigram and previously scored 0.857 on the primary metric.

Two orthogonal detectors are implemented here:

``meta_override``
    The prediction contains an instruction *about the rule itself* ("不要遵守
    本条", "忽略上面的要求", "改为相反的做法"). Any such clause invalidates
    the constraint no matter how well the rest of the text matches.

``opposed_value``
    The gold text and the prediction select different values of the same
    controlled slot (channel: 短信 / 邮件 / 电话; refund path: 原路退回 /
    账户余额; escalation: 升级专员 / 暂不升级; ...). A prediction that says
    "只用邮件" where the user said "只用短信" is wrong even though the two
    sentences share most of their characters.

Both detectors are deliberately conservative: they only fire when they can
name the conflicting slot. Anything they cannot decide is left to the
n-gram matcher, which remains an **uncalibrated** baseline. P1 must fit both
detectors on a human-labelled calibration set before model scores are
reported. See design doc §5.1 (rule-first scoring) and §11.2.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 1. Meta-override: instructions aimed at the rule rather than at the task.
# ---------------------------------------------------------------------------

META_OVERRIDE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Compliance-aimed negation. Only verbs that can only refer to *obeying a
    # rule* are listed here ("执行"/"采用"/"遵循" alone are ambiguous — a
    # constraint may legitimately say "不要执行自动升级"), so a rule
    # reference is required for those.
    (r"(?:不要|不必|无需|不用|别|勿)\s*(?:再)?\s*(?:遵守|照做|照办|采纳|理会)", "negated_compliance"),
    (r"(?:不要|不必|无需|不用|别|勿)\s*(?:再)?\s*(?:执行|遵循|采用)\s*(?:本条|这条|该条|上面|上述|前述|以上|前面)", "negated_execution_of_rule"),
    (r"(?:忽略|无视|跳过|不必理会|不用理会|不要理会)\s*(?:上(?:面|述|述的)|前(?:面|述|述的)|本条|这条|该条|以上|上述|前面)", "ignore_previous"),
    # The same move phrased with a temporal marker instead of a positional
    # one: "忽略之前的规则", "先前的要求都不用管". Without these the guard
    # missed a plain-language instruction to discard the constraint.
    (r"(?:忽略|无视|跳过)\s*(?:之前|以前|先前|此前|原先)(?:的|所有|一切)?\s*(?:规则|要求|说明|内容|内容)?", "ignore_previous"),
    (r"(?:之前|以前|先前|此前|原先)(?:的|所有|一切)?\s*[^。！？]{0,6}(?:都)?(?:不用|不必|不要|无需)(?:管|理会|理会了|照做|遵守)", "ignore_previous"),
    (r"改为\s*(?:完全)?相反", "inverted_instruction"),
    # "按完全相反的方式理解" — an inversion that never says "改为".
    (r"按[^。！？]{0,4}(?:完全)?相反(?:的)?(?:方式|方法|做法|理解|理解方式)", "inverted_instruction"),
    (r"(?:与|跟)(?:本条|这条|该条|上(?:面|述)|前面)[^。！？]{0,8}(?:相反|矛盾|冲突)", "contradicts_stated_rule"),
    (r"(?:违反|不遵守|不执行)(?:本条|这条|该条|上(?:面|述))", "violates_stated_rule"),
    (r"不要按(?:本条|这条|上面|上述)(?:所述|的)?(?:要求|方式|做法)?\s*(?:执行|做|办)", "negated_execution"),
    # "这条作废" / "本条无效" — a constraint declared void without any
    # instruction verb at all.
    (r"(?:本条|这条|该条|此条|上面(?:的)?(?:规则|要求)?)[^。！？]{0,2}(?:作废|无效|不算数|不用管)", "rule_voided"),
)

_COMPILED_META = tuple((re.compile(pattern), code) for pattern, code in META_OVERRIDE_PATTERNS)

# A trailing instruction clause ("..., 但这条请不要执行。") negates compliance
# without naming the rule explicitly. Only the final clause is inspected, so
# ordinary constraint content ("别打手机", "不要替我提交") is never affected.
_TRAILING_NEGATION = re.compile(
    r"(?:不要|不必|无需|不用|别|勿)\s*(?:请\s*)?(?:再\s*)?"
    r"(?:执行|遵守|遵循|照做|采用|理会|采纳|照办)"
)

_CLAUSE_SPLIT = re.compile(r"[。！？；，,.!?;]")

# Negation cues that flip the polarity of the value that follows them.
_NEGATION_CUES = ("不要", "不用", "无需", "不必", "别", "勿", "不再", "取消", "停止", "禁止", "不能", "不可", "暂不", "先不")


def meta_override(text: str) -> list[str]:
    """Return the codes of every "ignore/contradict the rule" clause found."""
    found: list[str] = []
    for pattern, code in _COMPILED_META:
        if pattern.search(text):
            found.append(code)
    # A trailing instruction clause ("..., 但这条请不要执行") negates compliance
    # without naming the rule explicitly. Only the *last* clause is inspected,
    # so ordinary constraint content ("别打手机", "不要替我提交") is unaffected.
    clauses = [clause.strip() for clause in _CLAUSE_SPLIT.split(text) if clause.strip()]
    if clauses and _TRAILING_NEGATION.search(clauses[-1]):
        found.append("trailing_negated_execution")
    return found


# ---------------------------------------------------------------------------
# 2. Controlled slots with opposed values.
# ---------------------------------------------------------------------------

# Each family is a controlled vocabulary extracted from the客服 scenario
# ontology. Values are ordered; the extractor returns the *affirmed* values
# (negated mentions are dropped).
_SLOT_FAMILIES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "channel",
        (
            ("短信", r"短信"),
            ("邮件", r"邮件"),
            ("电话", r"电话|打电话|拨打"),
            ("微信", r"微信"),
        ),
    ),
    (
        "ticket_alert_channel",
        (
            ("电话提醒", r"电话提醒"),
            ("短信提醒", r"短信提醒"),
            ("邮件提醒", r"邮件提醒"),
        ),
    ),
    (
        "refund_path",
        (
            ("原路退回", r"原路(?:退回|返回)"),
            ("账户余额", r"(?:退到|退回)?账户余额"),
        ),
    ),
    (
        "submission",
        (
            ("先自行检查", r"(?:等待|等我|先)(?:用户)?自行检查|自行检查后再"),
            ("直接提交", r"直接提交|立即提交|帮我提交|代为提交"),
            ("不提交", r"(?:先)?(?:别|不|勿|不要)提交|暂不提交|不要替我提交"),
        ),
    ),
    (
        "escalation",
        (
            ("升级专员", r"升级专员|升级给专员|转专员"),
            ("暂不升级", r"(?:暂不|先不|不要|别)升级|不升级|暂缓升级"),
        ),
    ),
)

# Explicit oppositions. A pair only fires when BOTH sides name one of the two
# values — mere absence of a value is not opposition. The relation is
# symmetric: it is expanded into both directions below.
_OPPOSITIONS: tuple[tuple[str, str, str], ...] = (
    ("channel", "短信", "邮件"),
    ("channel", "短信", "电话"),
    ("channel", "邮件", "电话"),
    ("channel", "短信", "微信"),
    ("channel", "邮件", "微信"),
    ("channel", "电话", "微信"),
    ("ticket_alert_channel", "电话提醒", "短信提醒"),
    ("ticket_alert_channel", "电话提醒", "邮件提醒"),
    ("ticket_alert_channel", "短信提醒", "邮件提醒"),
    ("refund_path", "原路退回", "账户余额"),
    ("submission", "先自行检查", "直接提交"),
    ("submission", "先自行检查", "不提交"),
    ("escalation", "升级专员", "暂不升级"),
)

# Symmetric expansion: (family, a, b) also yields (family, b, a).
_SYMMETRIC_OPPOSITIONS: tuple[tuple[str, str, str], ...] = tuple(
    [(family, left, right) for family, left, right in _OPPOSITIONS]
    + [(family, right, left) for family, left, right in _OPPOSITIONS]
)


def _negated_before(text: str, match_start: int, window: int = 6) -> bool:
    prefix = text[max(0, match_start - window):match_start]
    return any(cue in prefix for cue in _NEGATION_CUES)


def extract_slot_values(text: str) -> dict[str, set[str]]:
    """Affirmed values per slot family. A negated mention does not count."""
    values: dict[str, set[str]] = {}
    for family, members in _SLOT_FAMILIES:
        affirmed: set[str] = set()
        for value, pattern in members:
            for match in re.finditer(pattern, text):
                if _negated_before(text, match.start()):
                    continue
                affirmed.add(value)
                break
        if affirmed:
            values[family] = affirmed
    return values


def opposed_value(gold_text: str, prediction_text: str) -> list[str]:
    """Return descriptions of every slot where gold and prediction disagree."""
    gold_values = extract_slot_values(gold_text)
    prediction_values = extract_slot_values(prediction_text)
    conflicts: list[str] = []
    for family, gold_value, prediction_value in _SYMMETRIC_OPPOSITIONS:
        gold_set = gold_values.get(family, set())
        prediction_set = prediction_values.get(family, set())
        if gold_value in gold_set and prediction_value in prediction_set and gold_value not in prediction_set:
            conflicts.append(f"{family}:{gold_value}!={prediction_value}")
    return conflicts


def semantic_conflict(gold_text: str, prediction_text: str) -> dict[str, Any]:
    """Full semantic verdict for one (gold, prediction) pair.

    A meta-override only counts when the prediction *adds* it: if the gold
    sentence itself contains the clause (e.g. a legitimate "忽略前面那条规则"),
    a prediction that quotes the gold must not be penalised for it.
    """
    gold_meta = meta_override(gold_text)
    prediction_meta = meta_override(prediction_text)
    added_meta = [code for code in prediction_meta if code not in gold_meta]
    opposed = opposed_value(gold_text, prediction_text)
    return {
        "conflicted": bool(added_meta or opposed),
        "meta_override": prediction_meta,
        "added_meta_override": added_meta,
        "opposed_value": opposed,
        "gold_slots": extract_slot_values(gold_text),
        "prediction_slots": extract_slot_values(prediction_text),
    }


def is_conflicting(gold_text: str, prediction_text: str) -> bool:
    """Convenience predicate used by the scorer."""
    verdict = semantic_conflict(gold_text, prediction_text)
    return bool(verdict["conflicted"])
