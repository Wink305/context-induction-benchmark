"""Rule-based audit of the candidate truth set.

This module does not "annotate" anything.  It checks whether each candidate
unit is *derivable from the dialogue text* — i.e. whether the evidence turns
actually contain the wording the label claims, whether the label is consistent
with the surface cues the ontology promises (R1–R8), and whether bookkeeping
fields (valid_to, superseded_by) are coherent.

Purpose: a synthetic dataset's gold comes from construction, so the useful
question is not "can a second annotator reproduce it" but "is the construction
self-consistent and humanly recoverable".  Every finding is reported with the
offending unit, never silently repaired.
"""

from __future__ import annotations

import re
from typing import Any

STANDING_CUES = ("以后", "都", "每次", "一直", "任何", "始终", "所有", "别再", "不要", "不能", "不准")
LOCAL_CUES = ("这个项目", "这张工单", "这张", "这单", "这一单", "这次", "本单", "本月", "本次", "这趟", "下一趟", "这款")
ONE_OFF_CUES = ("这次", "仅", "先", "这张", "本单", "本月", "本次", "这趟", "下一趟", "等我")
WITHDRAW_CUES = ("取消", "不算", "换掉", "撤回", "作废", "不要了")
AGENT_SIDE_CUES = ("默认", "平台", "系统", "建议", "惯例")
REQUIREMENT_CUES = ("不要", "别", "必须", "不能", "不准", "以后", "每次", "任何", "都要", "只能用")


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"[\s，。、！？：；,.!?:;\"'（）()《》【】]", "", text)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} if len(cleaned) > 1 else {cleaned}


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _turns_by_number(dialogue: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {turn["turn"]: turn for turn in dialogue["turns"]}


def _check_unit(
    unit: dict[str, Any],
    turns: dict[int, dict[str, Any]],
    total_turns: int,
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(code: str, detail: str, severity: str = "major") -> None:
        issues.append({"code": code, "detail": detail, "severity": severity})

    evidence = unit.get("evidence_turns") or []
    if not evidence:
        add("E0", "evidence_turns 为空")
        return issues
    missing = [turn for turn in evidence if turn not in turns]
    if missing:
        add("E1", f"evidence 指向不存在的轮次 {missing}（共 {total_turns} 轮）")
        return issues
    if any(turn < 1 for turn in evidence):
        add("E2", f"evidence 含非法轮次 {evidence}")

    # A unit claiming evidence on an agent turn must be a superseded record
    # (withdrawal turns are user turns) — agent turns never carry user demand.
    user_evidence = [turn for turn in evidence if turns[turn]["speaker"] == "user"]
    if not user_evidence:
        add("E3", f"evidence 全部落在客服轮 {evidence}，用户约束不应只有客服侧依据")
    elif len(user_evidence) != len(evidence) and unit.get("type") != "superseded":
        agent_only = [turn for turn in evidence if turns[turn]["speaker"] != "user"]
        add("E4", f"非 T 类的 evidence 含客服轮 {agent_only}", "minor")

    kind = unit.get("type")

    # Content must be recoverable from the evidence turns.
    joined = " ".join(str(turns[turn]["text"]) for turn in evidence)
    score = _overlap(unit.get("text") or unit.get("content") or "", joined)
    if score < 0.08 and kind != "implicit":
        add(
            "C1",
            f"content 与 evidence 文本重叠仅 {score:.2f}，可能无法从对话读出：{unit.get('text') or unit.get('content')}",
            "major" if score == 0 else "minor",
        )

    surface = " ".join(str(turns[turn]["text"]) for turn in evidence)

    if kind == "standing":
        has_standing = any(cue in surface for cue in STANDING_CUES)
        has_one_off = any(cue in surface for cue in ONE_OFF_CUES)
        if not has_standing and has_one_off:
            add("T1", f"S 类但 evidence 只有一次性措辞，疑似应为 O：{surface[:40]}")
        elif not has_standing:
            add("T2", f"S 类但 evidence 无任何长期性措辞，依据偏弱：{surface[:40]}", "minor")
    elif kind == "one_off":
        if len(evidence) >= 3:
            add("T3", f"O 类但有 {len(evidence)} 处依据，达到 R2 的 I 类门槛，疑似应为 I")
        if not any(cue in surface for cue in ONE_OFF_CUES) and "R1" not in (unit.get("rule_tags") or []):
            add("T4", f"O 类但 evidence 无任何一次性措辞，依据偏弱：{surface[:40]}", "minor")
    elif kind == "implicit":
        if len(evidence) < 3:
            add("T5", f"I 类但依据只有 {len(evidence)} 处，违反 R2（≥3 次才算模式）")
    elif kind == "superseded":
        if unit.get("valid_to") is None:
            add("T6", "T 类缺少 valid_to")
        else:
            if unit.get("valid_from") is not None and unit["valid_to"] < unit["valid_from"]:
                add("T7", f"T 类 valid_to({unit['valid_to']}) < valid_from({unit['valid_from']})")
        if not any(cue in surface for cue in WITHDRAW_CUES):
            add("T8", "T 类但 evidence 无撤回/替换措辞")
        target = unit.get("superseded_by")
        if target is not None:
            if target not in by_id:
                add("T9", f"superseded_by 指向不存在的单元 {target}")
            else:
                other = by_id[target]
                if other.get("type") not in {"standing", "implicit"}:
                    add("T10", f"superseded_by 指向的类型是 {other.get('type')}，应为 S/I")
    else:
        add("Y1", f"未知类型 {kind!r}")

    # 作用域应由「规则生效那一轮」的措辞决定：T 类的第二条证据是撤回轮，
    # 撤回轮常带「这次」，不能拿来判定原规则的作用域。
    anchor = unit.get("valid_from")
    scope_text = str(turns[anchor]["text"]) if anchor in turns else surface

    scope = unit.get("scope")
    if scope not in {"global", "session", "task"}:
        add("S1", f"未知作用域 {scope!r}")
    elif scope == "global":
        # I 类的定义就是「表面像一次性、重复后才是稳定」，其证据必然带一次性措辞，
        # 因此对 implicit 豁免该检查。
        local_cues = [cue for cue in LOCAL_CUES if cue in scope_text] if kind != "implicit" else []
        if local_cues:
            add("S2", f"作用域标 global 但 evidence 有局部限定词 {local_cues}", "minor")
    elif scope in {"task", "session"}:
        if not any(cue in scope_text for cue in LOCAL_CUES):
            add("S3", f"作用域标 {scope} 但 evidence 无局部限定词", "minor")

    if unit.get("category") not in {"communication", "process", "domain"}:
        add("K1", f"未知类别 {unit.get('category')!r}")
    return issues


def audit_dataset(dialogues: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    unit_total = 0
    clean_units = 0
    for dialogue in dialogues:
        did = dialogue["dialogue_id"]
        turns = _turns_by_number(dialogue)
        truth = dialogue.get("candidate_truth", [])
        by_id = {unit["id"]: unit for unit in truth}
        unit_total += len(truth)
        for unit in truth:
            issues = _check_unit(unit, turns, len(turns), by_id)
            if not issues:
                clean_units += 1
                continue
            findings.append(
                {
                    "dialogue_id": did,
                    "unit_id": unit["id"],
                    "type": unit.get("type"),
                    "content": unit.get("text") or unit.get("content"),
                    "evidence_turns": unit.get("evidence_turns"),
                    "issues": issues,
                }
            )

        # distractors must actually occur in an agent turn
        full_text = " ".join(str(turn["text"]) for turn in dialogue["turns"])
        for distractor in dialogue.get("distractors", []):
            text = distractor["text"]
            if text not in full_text:
                findings.append(
                    {
                        "dialogue_id": did,
                        "unit_id": distractor["id"],
                        "type": "distractor",
                        "content": text,
                        "evidence_turns": distractor.get("evidence_turns"),
                        "issues": [
                            {
                                "code": "D1",
                                "detail": "伪约束文本未出现在对话中，FMR 题无法成立",
                                "severity": "critical",
                            }
                        ],
                    }
                )
            else:
                holder = [
                    turn for turn in dialogue["turns"] if text in str(turn["text"])
                ]
                if holder and all(turn["speaker"] != "agent" for turn in holder):
                    findings.append(
                        {
                            "dialogue_id": did,
                            "unit_id": distractor["id"],
                            "type": "distractor",
                            "content": text,
                            "evidence_turns": distractor.get("evidence_turns"),
                            "issues": [
                                {
                                    "code": "D2",
                                    "detail": "伪约束不是由客服说出，无法构成「客服单方声明」陷阱",
                                    "severity": "major",
                                }
                            ],
                        }
                    )

        # empty dialogues must not contain constraint-like user turns
        if not truth and not dialogue.get("distractors"):
            suspect = [
                turn["turn"]
                for turn in dialogue["turns"]
                if turn["speaker"] == "user"
                and any(cue in str(turn["text"]) for cue in REQUIREMENT_CUES)
            ]
            if suspect:
                findings.append(
                    {
                        "dialogue_id": did,
                        "unit_id": "empty-check",
                        "type": "empty",
                        "content": "本段设计为空约束段，但用户轮出现约束措辞",
                        "evidence_turns": suspect,
                        "issues": [
                            {
                                "code": "M1",
                                "detail": f"空段第 {suspect} 轮含约束措辞，与「无任何约束」设计冲突",
                                "severity": "critical",
                            }
                        ],
                    }
                )

    severity_counts: dict[str, int] = {}
    code_counts: dict[str, int] = {}
    for row in findings:
        for issue in row["issues"]:
            severity_counts[issue["severity"]] = severity_counts.get(issue["severity"], 0) + 1
            code_counts[issue["code"]] = code_counts.get(issue["code"], 0) + 1
    return {
        "dialogue_n": len(dialogues),
        "unit_n": unit_total,
        "clean_unit_n": clean_units,
        "flagged_unit_n": len(findings),
        "clean_rate": round(clean_units / unit_total, 4) if unit_total else None,
        "severity_counts": severity_counts,
        "code_counts": dict(sorted(code_counts.items())),
        "findings": findings,
    }
