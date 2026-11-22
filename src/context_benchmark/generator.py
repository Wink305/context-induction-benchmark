"""Deterministic P0 dialogue generator.

The output is deliberately labelled *candidate truth*. It is an annotation aid,
not human-adjudicated gold and must never be reported as gold v1.0.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import DATASET_STATUS
from .io import read_json, sha256_file, write_json, write_jsonl
from .ontology import RULES

SCENARIOS = {
    "ecommerce": {
        "greeting": "我想处理一张商品订单。",
        "stable": "以后推荐商品时优先看国产品牌。",
        "one_off": "仅这张订单不要含香水，下一单再按平常偏好。",
        "domain": "我对花生过敏，任何推荐都不能含花生。",
        "task": "商品订单",
    },
    "finance": {
        "greeting": "我想咨询开户和网点服务。",
        "stable": "以后推荐网点时优先给地铁可达的。",
        "one_off": "仅本次开户不选邮寄材料，下次再决定。",
        "domain": "我住杭州，别再推北京门店。",
        "task": "开户工单",
    },
    "travel": {
        "greeting": "我想安排一次旅行。",
        "stable": "以后安排住宿时优先选可免费取消的。",
        "one_off": "这次行程不考虑高原地区，下一趟另说。",
        "domain": "我对花生过敏，任何餐饮推荐都不能含花生。",
        "task": "旅行行程",
    },
    "telecom": {
        "greeting": "我想调整手机套餐。",
        "stable": "以后推荐套餐时优先选无合约期的。",
        "one_off": "仅本月不要叠加流量包，下个月再算。",
        "domain": "我住杭州，别再推北京营业厅。",
        "task": "套餐工单",
    },
}

# Lexical variants for the otherwise-identical "generic" slots. Each variant
# keeps the same surface cues the ontology/audit rely on (以后/都, 这次/仅,
# 取消/撤回, 三次 independent turns) but changes the actual wording. Without
# this, 30 standard dialogues shared c001–c012 verbatim, so memorising one row
# of the answer key scored full marks on every dialogue — the benchmark was
# measuring template recall, not induction.
# Each entry: (user utterance, agent acknowledgment, normalised truth text).
VARIANTS: dict[str, list[tuple[str, str, str]]] = {
    "mail": [
        ("以后都只用邮件联系我，不要打电话。", "好的，以后都走邮件，不电话联系。", "以后都只用邮件联系我，不要打电话"),
        ("以后都只用短信通知我，别打手机。", "好的，以后都用短信，不拨手机。", "以后都只用短信通知我，别打手机"),
        ("以后所有进度都用站内消息发我，不要电话。", "明白，以后走站内消息，不打电话。", "以后所有进度都用站内消息发我，不要电话"),
    ],
    "two_lines": [
        ("这段又写长了，我只看两行重点。", "我先给您两行重点。", "回复应压缩为两行重点"),
        ("这次也别展开，我只看两行重点。", "我先给您两行重点。", "回复应压缩为两行重点"),
        ("还是太细了，我只看两行重点。", "我先给您两行重点。", "回复应压缩为两行重点"),
        ("以后回复我只要三条要点就够。", "好的，我先给三条要点。", "回复应压缩为三条要点"),
        ("这次也只要三条要点，别铺开。", "好的，我先给三条要点。", "回复应压缩为三条要点"),
        ("还是太长，以后给我三条要点就行。", "好的，我先给三条要点。", "回复应压缩为三条要点"),
    ],
    "phone_reminder": [
        ("以后可以用电话提醒我工单进度。", "已记录电话提醒。", "通过电话提醒工单进度"),
        ("以后可以用短信提醒我工单进度。", "已记录短信提醒。", "通过短信提醒工单进度"),
        ("以后可以用邮件提醒我工单进度。", "已记录邮件提醒。", "通过邮件提醒工单进度"),
    ],
    "escalation": [
        ("以后超过 48 小时没处理就升级人工主管。", "已记录 48 小时升级规则。", "超过48小时未处理时升级人工主管"),
        ("以后超过 24 小时没处理就升级值班经理。", "已记录 24 小时升级规则。", "超过24小时未处理时升级值班经理"),
        ("以后超过 72 小时没处理就升级专员处理。", "已记录 72 小时升级规则。", "超过72小时未处理时升级专员处理"),
    ],
    "no_submit": [
        ("先别替我提交，我自己再试一下。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
        ("这次也先别提交，我再核对一下。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
        ("还是先别提交，我想自己再试一次。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
        ("先别替我付款，我自己再核对一下。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
        ("这次也先别付款，我再想想。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
        ("还是先别付款，我想自己再确认一次。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
    ],
}


class TurnBuilder:
    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def pair(self, user_text: str, agent_text: str) -> tuple[int, int]:
        user_turn = len(self.turns) + 1
        self.turns.append({"turn": user_turn, "speaker": "user", "text": user_text})
        agent_turn = len(self.turns) + 1
        self.turns.append({"turn": agent_turn, "speaker": "agent", "text": agent_text})
        return user_turn, agent_turn


def _constraint(
    cid: str,
    ctype: str,
    text: str,
    evidence: list[int],
    valid_from: int,
    scope: str,
    category: str,
    *,
    valid_to: int | None = None,
    superseded_by: str | None = None,
    rules: list[str] | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": cid,
        "type": ctype,
        "text": text,
        "evidence_turns": sorted(set(evidence)),
        "valid_from": valid_from,
        "valid_to": valid_to,
        "scope": scope,
        "superseded_by": superseded_by,
        "category": category,
        "rule_tags": rules or [],
        "note": note,
        "candidate_only": True,
    }


def _probes(replacement: bool) -> list[dict[str, Any]]:
    second_target = "c009" if replacement else "c005"
    second_behaviors = (
        [
            {"id": "b1", "text": "退到账户余额。", "is_compliant": True},
            {"id": "b2", "text": "仍按已替换规则原路退回。", "is_compliant": False},
        ]
        if replacement
        else [
            {"id": "b1", "text": "超过 48 小时未处理就升级主管。", "is_compliant": True},
            {"id": "b2", "text": "忽略升级规则，继续无限等待。", "is_compliant": False},
        ]
    )
    return [
        {
            "probe_id": "p001",
            "target_constraint_id": "c001",
            "related": True,
            "query": "新工单需要发送进度通知，应如何联系用户？",
            "behaviors": [
                {"id": "b1", "text": "通过邮件发送，不拨打电话。", "is_compliant": True},
                {"id": "b2", "text": "先打电话，再发短信。", "is_compliant": False},
            ],
        },
        {
            "probe_id": "p002",
            "target_constraint_id": second_target,
            "related": True,
            "query": "处理一个新的适用任务时，应遵循哪项行为？",
            "behaviors": second_behaviors,
        },
    ]


def _unrelated_probe(semantics: str = "task_correct") -> dict[str, Any]:
    """False-trigger control: following the injected rule is *wrong* here.

    ``is_compliant`` means "correct for the task". In this unrelated writing
    task, the correct behaviour is to ignore the injected contact rule. An
    earlier version marked following the injection as compliant, inverting the
    false-trigger direction for any downstream consumer of this field.

    ``semantics="follows_injection"`` flips the flag so that ``is_compliant``
    marks the behaviour that *applies* the injected rule instead. That is the
    reading under which a scorer can measure a false trigger as "the true arm
    picks the injected behaviour more often than the none arm"; the probe
    records which reading it uses.
    """
    follows = {"id": "b1", "text": "先按邮件规则确认联系渠道。"}
    clean = {"id": "b2", "text": "只是写诗，不涉及联系渠道。"}
    if semantics == "follows_injection":
        behaviors = [
            {"id": "b1", "text": follows["text"], "is_compliant": True},
            {"id": "b2", "text": clean["text"], "is_compliant": False},
        ]
        follows_injection = {"b1": True, "b2": False}
    else:
        behaviors = [
            {"id": "b1", "text": follows["text"], "is_compliant": False},
            {"id": "b2", "text": clean["text"], "is_compliant": True},
        ]
        follows_injection = {"b1": True, "b2": False}
    return {
        "probe_id": "p003",
        "target_constraint_id": "c001",
        "query": "与新工单无关：请直接写一首两行的诗。",
        "behaviors": behaviors,
        "related": False,
        "follows_injection": follows_injection,
        "compliant_semantics": semantics,
    }


def _pick(rng: random.Random, key: str) -> tuple[str, str, str]:
    return rng.choice(VARIANTS[key])


def _pick_triplet(rng: random.Random, key: str) -> list[tuple[str, str, str]]:
    """Pick one coherent 3-turn theme (implicit constraints need same pattern 3x)."""
    pool = VARIANTS[key]
    start = rng.randrange(0, len(pool), 3)
    return pool[start:start + 3]


def _build_standard(index: int, scenario: str, kind: str, structure: str, rng: random.Random) -> dict[str, Any]:
    spec = SCENARIOS[scenario]
    b = TurnBuilder()
    truth: list[dict[str, Any]] = []
    non_units: list[dict[str, Any]] = []
    distractors: list[dict[str, Any]] = []

    _, agent_turn = b.pair(
        spec["greeting"],
        "可以。平台内部建议优先在线处理，但这只是客服侧信息，不代表您的要求。",
    )
    non_units.append({
        "id": "n-r4", "kind": "agent_side_rule", "text": "平台内部建议优先在线处理",
        "evidence_turns": [agent_turn], "rule_tags": ["R4"], "note": "客服侧声明，不标用户约束",
    })

    mail_user, mail_agent, mail_text = _pick(rng, "mail")
    u, a = b.pair(mail_user, mail_agent)
    first_mail_turn = u
    non_units.append({
        "id": "n-r6", "kind": "agent_paraphrase", "text": "好的，以后照办",
        "evidence_turns": [a], "rule_tags": ["R6"], "note": "复述确认，不产生新单元",
    })

    u, _ = b.pair("这个项目里都发邮件沟通，别在这个项目里打电话。", "明白，仅这个项目使用邮件。")
    truth.append(_constraint("c002", "one_off", "本项目内只通过邮件沟通", [u], u, "task", "communication", rules=["R1"], note="规则化语气但绑定项目"))

    implicit_com: list[int] = []
    two_lines_group = _pick_triplet(rng, "two_lines")
    for user_text, agent_text, _ in two_lines_group:
        u, _ = b.pair(user_text, agent_text)
        implicit_com.append(u)
    two_lines_truth = two_lines_group[0][2]
    truth.append(_constraint("c003", "implicit", two_lines_truth, implicit_com, implicit_com[0], "global", "communication", note="三次独立交互模式；候选 I，待人工确认"))

    remind_user, remind_agent, remind_text = _pick(rng, "phone_reminder")
    u, _ = b.pair(remind_user, remind_agent)
    if structure in {"replace_withdraw", "withdraw_only"}:
        w, _ = b.pair(f"之前说可以{remind_text}，那条现在取消。", "已取消该提醒。")
        truth.append(_constraint("c004", "superseded", remind_text, [u, w], u, "global", "communication", valid_to=w - 1, rules=["R3"], note=f"第 {w} 轮显式撤回"))
    else:
        truth.append(_constraint("c004", "standing", remind_text, [u], u, "global", "communication"))

    esc_user, _, esc_text = _pick(rng, "escalation")
    if kind == "distractor":
        u, a = b.pair(
            esc_user,
            "已记录升级规则。另外系统默认使用短信提醒所有工单。",
        )
    else:
        u, a = b.pair(esc_user, "已记录升级规则。")
    truth.append(_constraint("c005", "standing", esc_text, [u], u, "global", "process"))
    if kind == "distractor":
        distractors.append({
            "id": "d001", "text": "默认使用短信提醒所有工单", "evidence_turns": [a],
            "category": "communication", "expected": "exclude", "note": "客服单方声明，用户未确认，不得抽为用户约束",
        })

    u, _ = b.pair("仅这张工单先不要升级，等我明天确认。", "本单暂不升级。")
    truth.append(_constraint("c006", "one_off", "本工单暂不升级并等待次日确认", [u], u, "task", "process"))

    implicit_process: list[int] = []
    no_submit_group = _pick_triplet(rng, "no_submit")
    for user_text, agent_text, _ in no_submit_group:
        u, _ = b.pair(user_text, agent_text)
        implicit_process.append(u)
    no_submit_truth = no_submit_group[0][2]
    truth.append(_constraint("c007", "implicit", no_submit_truth, implicit_process, implicit_process[0], "global", "process", note="三次独立交互模式；候选 I，待人工确认"))

    old_refund_turn, _ = b.pair("以后退款都原路退回。", "已记录原路退回。")
    if structure in {"replace_withdraw", "replace_only"}:
        new_refund_turn, _ = b.pair("以后退款优先退到账户余额，前面那条原路退回的规则换掉。", "已改为优先退到账户余额。")
        truth.append(_constraint("c008", "superseded", "退款原路退回", [old_refund_turn, new_refund_turn], old_refund_turn, "global", "process", valid_to=new_refund_turn - 1, superseded_by="c009", rules=["R7"]))
        truth.append(_constraint("c009", "standing", "退款优先退到账户余额", [new_refund_turn], new_refund_turn, "global", "process", rules=["R7"]))
    else:
        withdrawal_turn, _ = b.pair("退款原路退回那条取消，这次先退到账户余额。", "已撤回旧规则，本次退到账户余额。")
        truth.append(_constraint("c008", "superseded", "退款原路退回", [old_refund_turn, withdrawal_turn], old_refund_turn, "global", "process", valid_to=withdrawal_turn - 1, rules=["R3"]))
        truth.append(_constraint("c009", "one_off", "本次退款退到账户余额", [withdrawal_turn], withdrawal_turn, "task", "process"))

    u, _ = b.pair(spec["domain"], "已记录相关限制。")
    truth.append(_constraint("c010", "standing", spec["domain"], [u], u, "global", "domain", rules=["R5"], note="事实陈述与行为要求同句，只标行为要求"))

    potential_turns: list[int] = []
    u, _ = b.pair("这张说明也太长了，我先看重点。", "我先提炼重点。")
    potential_turns.append(u)
    u, _ = b.pair(f"{spec['one_off']} 这张说明也太长了，我先看重点。", "仅本次照办，并先给重点。")
    potential_turns.append(u)
    truth.append(_constraint("c011", "one_off", spec["one_off"], [u], u, "task", "domain" if scenario in {"travel", "ecommerce"} else "process"))
    non_units.append({
        "id": "n-r2", "kind": "potential_implicit", "text": "先看重点",
        "evidence_turns": potential_turns, "rule_tags": ["R2"], "note": "仅两次，不标 I",
    })

    # R8 repeat: the same channel rule restated later. Reuse the selected mail
    # variant so c001's text remains recoverable from both evidence turns.
    repeat_clause = mail_user
    if kind == "distractor":
        u, a = b.pair(
            f"{spec['stable']} 另外，{repeat_clause}",
            "已记录，并继续遵循之前的联系渠道。另外按平台惯例默认向所有用户推荐最贵的方案。",
        )
    else:
        u, a = b.pair(f"{spec['stable']} 另外，{repeat_clause}", "已记录，并继续遵循之前的联系渠道。")
    truth.append(_constraint("c012", "standing", spec["stable"], [u], u, "global", "domain" if scenario in {"travel", "ecommerce"} else "process", note="弱化偏好类候选，人工判断强度"))
    truth.insert(0, _constraint("c001", "standing", mail_text, [first_mail_turn, u], first_mail_turn, "global", "communication", rules=["R8"], note="重复陈述合并"))

    if kind == "distractor":
        distractors.append({
            "id": "d002", "text": "默认向所有用户推荐最贵的方案", "evidence_turns": [a],
            "category": "domain", "expected": "exclude", "note": "客服单方声明，用户未确认，不得抽为用户约束",
        })

    b.pair("我稍后再确认其他细节。", "好的，您可以之后继续。")
    if len(b.turns) < 25:
        b.pair("目前没有别的补充。", "好的。")

    replacement = structure in {"replace_withdraw", "replace_only"}
    axes = {
        "content_categories": sorted({item["category"] for item in truth}),
        "explicitness": ["explicit_rule", "explicit_scope", "implicit_pattern"],
        "injection_positions": sorted({"early" if item["valid_from"] <= 5 else "late" if item["valid_from"] >= 30 else "middle" for item in truth}),
        "repeat_counts": sorted({len(item["evidence_turns"]) for item in truth}),
        "conflict_structure": structure,
        "scenario": scenario,
        "semantic_disconnect_filter": "not_implemented_in_p0_v0",
    }
    return {
        "dialogue_id": f"p0-{index:03d}",
        "status": DATASET_STATUS,
        "scenario": scenario,
        "dialogue_kind": kind,
        "turns": b.turns,
        "candidate_truth": truth,
        "distractors": distractors,
        "non_units": non_units,
        "axes": axes,
        "probes": _probes(replacement),
    }


def _build_empty(index: int, scenario: str) -> dict[str, Any]:
    b = TurnBuilder()
    facts = [
        "我想了解一下服务进度。", "我住在杭州。", "上周我提交过材料。", "订单号末尾是 27。",
        "我明天下午有空。", "我已经看过说明页面。", "这次只是来咨询。", "我之前用过这个服务。",
        "我知道人工工作时间。", "我稍后再决定。", "名字里有一个生字。", "我已经收到回执。", "今天先了解到这里。",
    ]
    non_units = []
    for position, fact in enumerate(facts, 1):
        u, _ = b.pair(fact, "好的，我已了解这项背景信息。")
        non_units.append({"id": f"n-{position:02d}", "kind": "fact_or_chat", "text": fact, "evidence_turns": [u], "rule_tags": ["R5"]})
    return {
        "dialogue_id": f"p0-{index:03d}",
        "status": DATASET_STATUS,
        "scenario": scenario,
        "dialogue_kind": "empty",
        "turns": b.turns,
        "candidate_truth": [],
        "distractors": [],
        "non_units": non_units,
        "axes": {"content_categories": [], "explicitness": [], "injection_positions": [], "repeat_counts": [], "conflict_structure": "none", "scenario": scenario},
        "probes": [],
    }


# ===========================================================================
# v1 randomised engine (opt-in)
#
# Everything in this section is behind explicit config switches that are OFF
# by default, so ``configs/generator.json`` still regenerates ``data/pilot_v0``
# byte-for-byte. The frozen pilot set is a *version lock*: it must not move.
#
# Switches (all optional, all default to the frozen behaviour):
#   randomize_layout                    -> slot->turn permutation + v1 engine
#   add_unrelated_probes                -> one related=False control probe per
#                                          non-empty dialogue
#   length_tier                         -> short | medium | long turn budget
#   max_probe_bigram_overlap            -> lexical overlap ceiling (default .25)
#   semantic_filter_backend             -> name of the filter actually applied
#   unrelated_probe_compliant_semantics -> how is_compliant reads on the
#                                          related=False control probe
# ===========================================================================

LENGTH_TIERS: dict[str, int] = {
    "short": 42,
    "medium": 120,
    "long": 300,
}

# Character n-gram lexical filter. This is an *approximation* of a lexical
# overlap filter: it is NOT BM25, and it is NOT a semantic/embedding filter.
# No embedding model is available in this repository, so the cue--trigger
# "semantic disconnect" of the design document is approximated with character
# bigrams only. Everything that reports on it says so explicitly.
_BIGRAM_STRIP = re.compile(r"[\s，。、！？：；,.!?:;\"'（）()《》【】\[\]]")


def char_bigrams(text: str) -> set[str]:
    """Character bigram set of ``text`` (punctuation and spaces removed)."""
    cleaned = _BIGRAM_STRIP.sub("", text or "")
    if len(cleaned) <= 1:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def bigram_overlap(left: str, right: str) -> float:
    """Character-bigram Dice overlap in [0, 1] between two strings.

    Dice (2|A&B| / (|A|+|B|)) is the same lexical measure the rule audit
    already uses, so the two reports are comparable. Dice is >= Jaccard for
    the same pair, so any pair that passes a Dice ceiling also passes the
    equivalent Jaccard ceiling.
    """
    a, b = char_bigrams(left), char_bigrams(right)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


# Cue words that make an utterance look like a *rule*. Filler turns must not
# contain any of them, otherwise a dialogue that is supposed to carry no
# constraint would leak one (the rule audit flags this as M1).
_REQUIREMENT_CUES = ("不要", "别", "必须", "不能", "不准", "以后", "每次", "任何", "都要", "只能用")

# Surface pools for the v1 engine. Every pool holds at least five entries so
# that no key slot can be answered from one memorised sentence. Entries are
# (user utterance, agent reply, normalised truth) and, for the three slots that
# are used as probe targets, two extra behaviour strings (compliant first,
# non-compliant second).

V1_CONTACT_CHANNEL = [
    ("以后都只用邮件联系我，不要打电话。", "好的，以后都走邮件，不电话联系。", "以后都只用邮件联系我，不要打电话",
     "只通过邮件把进展送到本人手上，不打电话。", "先打电话通知本人，再补一条短信。"),
    ("以后都只用短信通知我，别打手机。", "好的，以后都用短信，不拨手机。", "以后都只用短信通知我，别打手机",
     "只用短信通知本人，不拨手机。", "先拨手机，再发一条短信。"),
    ("以后所有进度都用站内消息发我，不要电话。", "明白，以后走站内消息，不打电话。", "以后所有进度都用站内消息发我，不要电话",
     "只用站内消息把进展送到本人手上。", "先打电话，再补一条站内消息。"),
    ("今后一律通过企业微信找我，电话别再打了。", "收到，以后只用企业微信联系您。", "今后一律通过企业微信找我，电话别再打了",
     "只用企业微信联系本人，不打电话。", "先打电话，再补一条企业微信。"),
    ("以后有任何事情都发我邮箱，请勿来电。", "好的，后续统一发邮件，不来电。", "以后有任何事情都发我邮箱，请勿来电",
     "只用邮箱通知本人，不来电。", "先来电，再补一封邮件。"),
    ("后续沟通一律走App推送，麻烦别再打电话。", "已记录，之后只用App推送通知。", "后续沟通一律走App推送，麻烦别再打电话",
     "只用App推送把进展送到本人手上。", "先打电话，再补一条App推送。"),
]

V1_ONE_OFF_CHANNEL = [
    ("这个项目里都发邮件沟通，别在这个项目里打电话。", "明白，仅这个项目使用邮件。", "这个项目只通过邮件沟通"),
    ("这单的处理进度只发短信，别在这单里打电话。", "好的，这单只走短信通知。", "这单只通过短信沟通"),
    ("本次就麻烦你走站内消息，本次不要来电。", "收到，本次只用站内消息。", "本次只通过站内消息沟通"),
    ("这一单请用邮件联系，这一单别打手机。", "明白，这一单统一用邮件。", "这一单只通过邮件沟通"),
    ("这趟全程用App推送通知，这趟就免了电话。", "好的，这趟只用App推送。", "这趟只通过App推送沟通"),
    ("这款商品的跟进只发邮件，这边别再打电话。", "明白，这款只走邮件。", "这款只通过邮件沟通"),
]

# Implicit slots need a coherent three-turn pattern, so their pools are
# "themes": three independent utterances that share one normalised truth.
V1_REPLY_LENGTH_THEMES = [
    [
        ("这段又写长了，我只看两行重点。", "我先给您两行重点。", "回复应压缩为两行重点"),
        ("还是太长了，我只要两行。", "好的，给您两行。", "回复应压缩为两行重点"),
        ("后面的说明我只读两行，多了不看。", "明白，后续都控制在两行。", "回复应压缩为两行重点"),
    ],
    [
        ("以后回复我只要三条要点就够。", "好的，我先给三条要点。", "回复应压缩为三条要点"),
        ("每次答复给我列三条要点，别铺开。", "收到，只给三条要点。", "回复应压缩为三条要点"),
        ("回答的时候压成三条要点就行。", "明白，我按三条要点回复。", "回复应压缩为三条要点"),
    ],
    [
        ("先给我一份要点清单，三条以内。", "好的，三条要点如下。", "回复应压缩为三条以内要点"),
        ("答复写成三条，其余都略过。", "明白，只保留三条。", "回复应压缩为三条以内要点"),
        ("输出别超过三条要点。", "收到，按三条要点给。", "回复应压缩为三条以内要点"),
    ],
    [
        ("给我两段总结，就两段。", "好的，两段总结如下。", "回复应压缩为两段总结"),
        ("内容分成两段写，其他不写。", "明白，只写两段。", "回复应压缩为两段总结"),
        ("总结就用两段，别加小标题。", "收到，两段呈现。", "回复应压缩为两段总结"),
    ],
    [
        ("麻烦压成一张清单给我。", "好的，整理成一张清单。", "回复应压缩为一张清单"),
        ("用一张清单汇总，别分块。", "明白，合并成一张清单。", "回复应压缩为一张清单"),
        ("我只看一张清单，别的都不要。", "收到，只给一张清单。", "回复应压缩为一张清单"),
    ],
    [
        ("长的话先给个三句话版本。", "好的，先给三句话版本。", "回复应压缩为三句话版本"),
        ("用三句话讲清楚就行。", "明白，三句话概括。", "回复应压缩为三句话版本"),
        ("先来个三句话的简短版本。", "收到，先给三句话。", "回复应压缩为三句话版本"),
    ],
]

V1_SUBMIT_CONFIRM_THEMES = [
    [
        ("先别替我提交，我自己再试一下。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
        ("这次也先别提交，我再核对一下。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
        ("还是先别提交，我想自己再试一次。", "好的，本次暂不提交。", "提交前应等待用户自行检查"),
    ],
    [
        ("先别替我付款，我自己再核对一下。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
        ("这次也先别付款，我再想想。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
        ("还是先别付款，我想自己再确认一次。", "好的，本次暂不付款。", "付款前应等待用户自行检查"),
    ],
    [
        ("预约之前让我先过一眼，别直接下单。", "好的，本次暂不预约。", "预约前应等待用户自行过目"),
        ("这次先别帮我约，等我确认时间。", "好的，本次暂不预约。", "预约前应等待用户自行过目"),
        ("先别锁定名额，我自己再选选。", "好的，本次暂不预约。", "预约前应等待用户自行过目"),
    ],
    [
        ("变更之前先别改，等我回复。", "好的，本次暂不变更。", "变更前应等待用户自行确认"),
        ("这次先别动设置，我再想想。", "好的，本次暂不变更。", "变更前应等待用户自行确认"),
        ("还是先放着别改，我自己看看。", "好的，本次暂不变更。", "变更前应等待用户自行确认"),
    ],
    [
        ("开通之前先别开，等我自己点。", "好的，本次暂不开通。", "开通前应等待用户自行确认"),
        ("这次先别给我开通，我再确认下。", "好的，本次暂不开通。", "开通前应等待用户自行确认"),
        ("先别启用这个功能，我自己来。", "好的，本次暂不开通。", "开通前应等待用户自行确认"),
    ],
    [
        ("寄出之前先别寄给我，我再检查一遍。", "好的，本次暂不寄出。", "寄出前应等待用户自行检查"),
        ("这次先别寄出，等我核对地址。", "好的，本次暂不寄出。", "寄出前应等待用户自行检查"),
        ("还是先别寄，东西我自己拿。", "好的，本次暂不寄出。", "寄出前应等待用户自行检查"),
    ],
]

V1_REMINDER = [
    ("以后可以用电话提醒我工单进度。", "已记录电话提醒。", "通过电话提醒工单进度"),
    ("以后可以用短信提醒我工单进度。", "已记录短信提醒。", "通过短信提醒工单进度"),
    ("以后可以用邮件提醒我工单进度。", "已记录邮件提醒。", "通过邮件提醒工单进度"),
    ("以后工单有进展就用App推送提醒我。", "已记录App推送提醒。", "通过App推送提醒工单进度"),
    ("以后状态变化请用企业微信提醒我。", "已记录企业微信提醒。", "通过企业微信提醒工单进度"),
    ("以后进度更新就发我微信，别发短信了。", "已记录微信提醒。", "通过微信提醒工单进度"),
]

V1_ESCALATION = [
    ("以后超过 48 小时没处理就升级人工主管。", "已记录 48 小时升级规则。", "超过48小时未处理时升级人工主管",
     "超过 48 小时未处理就升级人工主管。", "继续等下去，不做升级。"),
    ("以后超过 24 小时没处理就升级值班经理。", "已记录 24 小时升级规则。", "超过24小时未处理时升级值班经理",
     "超过 24 小时未处理就升级值班经理。", "一直等下去，不找人接手。"),
    ("以后超过 72 小时没处理就升级专员处理。", "已记录 72 小时升级规则。", "超过72小时未处理时升级专员处理",
     "超过 72 小时未处理就升级专员处理。", "照旧排队等处理，不升级。"),
    ("以后挂单超过 36 小时就转给二线专员。", "已记录 36 小时转派规则。", "超过36小时未处理时转给二线专员",
     "超过 36 小时未处理就转给二线专员。", "留在原队列继续等。"),
    ("以后要是两个工作日还没动，就升级给客户经理。", "已记录两个工作日升级规则。", "超过两个工作日未处理时升级客户经理",
     "两个工作日还没动就升级给客户经理。", "不升级，继续自行处理。"),
    ("以后超过 12 小时没进展就升级给客服经理。", "已记录 12 小时升级规则。", "超过12小时未处理时升级客服经理",
     "超过 12 小时没进展就升级给客服经理。", "保持等待，不上报。"),
]

V1_ONE_OFF_NO_ESCALATE = [
    ("仅这张工单先不要升级，等我明天确认。", "本单暂不升级。", "这张工单暂不升级并等待次日确认"),
    ("这次先别升级，我明天再确认。", "本次暂不升级。", "这次暂不升级并等待次日确认"),
    ("这一单暂时不要升级，等我回复。", "这单先不升级。", "这一单暂不升级并等待用户确认"),
    ("本单先留在我这边，不要转人工主管。", "本单暂不转派。", "本单暂不转人工主管并等待用户确认"),
    ("这张单子先不要升级处理，等我消息。", "本单暂不升级。", "这张单子暂不升级并等待用户确认"),
    ("本次别升级，等我确认后再走流程。", "本次暂不升级。", "本次暂不升级并等待用户确认"),
]

V1_REFUND_OLD = [
    ("以后退款都原路退回。", "已记录原路退回。", "退款原路退回"),
    ("以后退款一律退回原来的支付账户。", "已记录原路退回。", "退款退回原支付账户"),
    ("以后退款就按原支付渠道退回。", "已记录原路退回。", "退款按原支付渠道退回"),
    ("以后处理退款时退回原来的付款方式。", "已记录原路退回。", "退款按原付款方式退回"),
    ("退款这块以后都走原路退回。", "已记录原路退回。", "退款走原路退回"),
    ("以后凡是退款，都退到原来扣款的地方。", "已记录原路退回。", "退款退回原扣款渠道"),
]

V1_REFUND_NEW_STANDING = [
    ("以后退款优先退到账户余额。", "已改为优先退到账户余额。", "退款优先退到账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
    ("以后退款就退到账户余额里。", "已记录退到账户余额。", "退款退到账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
    ("以后退款优先走账户余额，不再原路。", "已改为账户余额退款。", "退款优先走账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
    ("以后退费直接进账户余额。", "已记录账户余额退款。", "退款直接进账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
    ("退款以后都退回账户余额。", "已记录账户余额退款。", "退款退回账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
    ("以后退款金额请打到我账户余额里。", "已记录账户余额退款。", "退款金额退到账户余额",
     "退到账户余额。", "仍按原来的路径原路退回。"),
]

V1_REFUND_NEW_ONEOFF = [
    ("前面那条原路退回的规则取消，这次先退到账户余额。", "已撤回旧规则，本次退到账户余额。", "本次退款退到账户余额",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
    ("之前说的原路退回先不算了，这一单退到账户余额。", "已撤回旧规则，本单退到账户余额。", "本次退款改退到账户余额",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
    ("原路退回那条我改主意了，作废了，本次退到账户余额。", "已撤回旧规则，本单退到账户余额。", "本单退款改退到账户余额",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
    ("那条原路退回作废了，这次先退账户余额。", "已撤回旧规则，本次退到账户余额。", "本次退款退回账户余额",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
    ("原路退回的规则换掉，这次退到账户余额。", "已撤回旧规则，本单退到账户余额。", "本次退款不再原路退回",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
    ("原来那条原路退回取消了，先不用原路退我，这次退到账户余额。", "已撤回旧规则，本次退到账户余额。", "本次退款改走账户余额",
     "本次退到账户余额。", "本次仍按原来的路径原路退回。"),
]

# Domain (c010), stable preference (c012) and one-off (c011) pools per
# scenario. Five scenarios: the four frozen ones plus a new banking one.
V1_SCENARIOS: dict[str, dict[str, Any]] = {
    "ecommerce": {
        "task": "商品订单",
        "category": "domain",
        "greetings": [
            "我想处理一张商品订单。",
            "我有件商品的售后事情要办。",
            "这个订单我需要咨询一下。",
            "麻烦帮我看看这个购物订单。",
        ],
        "domain": [
            ("我对花生过敏，任何推荐都不能含花生。", "已记录相关限制。", "我对花生过敏，任何推荐都不能含花生"),
            ("我对花生严重过敏，推的东西绝对不能有花生成分。", "已记录相关限制。", "我对花生严重过敏，推荐不能含花生成分"),
            ("我对坚果过敏，任何商品都不能出现坚果。", "已记录相关限制。", "我对坚果过敏，商品不能含坚果"),
            ("花生过敏很严重，推荐和说明里都不能提到花生。", "已记录相关限制。", "我对花生过敏，推荐与说明都不能涉及花生"),
            ("我对花生和榛子都过敏，任何推荐都不要含这两样。", "已记录相关限制。", "我对花生和榛子过敏，推荐不能含这两样"),
            ("我吃花生会过敏，所以商品里不能有花生。", "已记录相关限制。", "我对花生过敏，商品不能含花生"),
        ],
        "stable": [
            ("以后推荐商品时优先看国产品牌。", "已记录。", "以后推荐商品时优先看国产品牌"),
            ("以后挑商品优先看国产的牌子。", "已记录。", "以后挑商品优先看国产的牌子"),
            ("以后推荐商品优先考虑国产品牌。", "已记录。", "以后推荐商品优先考虑国产品牌"),
            ("以后选商品时留意国产的牌子。", "已记录。", "以后选商品时留意国产的牌子"),
            ("以后推荐商品时国内品牌优先。", "已记录。", "以后推荐商品时国内品牌优先"),
            ("以后选品优先挑国产的。", "已记录。", "以后选品优先挑国产的"),
        ],
        "one_off": [
            "仅这张订单不要含香水，下一单再按平常偏好。",
            "这次下单先别带赠品，之后按老规矩。",
            "就这一单不要加急配送，下回按平常来。",
            "仅这次不要拆成两件，之后还是照旧。",
            "这张订单先不加购，后面照原样。",
            "本次下单不要刻字，下一次再说。",
        ],
    },
    "travel": {
        "task": "旅行行程",
        "category": "domain",
        "greetings": [
            "我想安排一次旅行。",
            "帮我规划一趟出行吧。",
            "有件事想咨询下行程安排。",
            "麻烦看一下我这趟的行程。",
        ],
        "domain": [
            ("我对花生过敏，任何餐饮推荐都不能含花生。", "已记录相关限制。", "我对花生过敏，任何餐饮推荐都不能含花生"),
            ("我对海鲜过敏，任何推荐都不能含海鲜。", "已记录相关限制。", "我对海鲜过敏，推荐不能含海鲜"),
            ("我对花生和芒果都过敏，任何餐食都不要含这两样。", "已记录相关限制。", "我对花生和芒果过敏，餐食不能含这两样"),
            ("我对花生过敏很严重，推荐和说明里都不能提到花生。", "已记录相关限制。", "我对花生过敏，推荐与说明都不能涉及花生"),
            ("我对坚果过敏，任何餐食里都不能出现坚果。", "已记录相关限制。", "我对坚果过敏，餐食不能含坚果"),
            ("我吃花生会过敏，所以餐饮推荐里不能有花生。", "已记录相关限制。", "我对花生过敏，餐饮推荐不能含花生"),
        ],
        "stable": [
            ("以后安排住宿时优先选可免费取消的。", "已记录。", "以后安排住宿时优先选可免费取消的"),
            ("以后订酒店优先找能免费退订的房源。", "已记录。", "以后订酒店优先找能免费退订的房源"),
            ("以后挑住宿尽量选可以随时取消的。", "已记录。", "以后挑住宿尽量选可以随时取消的"),
            ("以后安排住处时留意能不能免费取消。", "已记录。", "以后安排住处时留意能不能免费取消"),
            ("以后订民宿优先选支持免费取消的。", "已记录。", "以后订民宿优先选支持免费取消的"),
            ("以后住宿就找可无条件取消的房源。", "已记录。", "以后住宿就找可无条件取消的房源"),
        ],
        "one_off": [
            "这次行程不考虑高原地区，下一趟另说。",
            "仅这趟不去海边，下回再定。",
            "这一单先不要安排红眼航班，后面再说。",
            "本次行程先不订民宿，下趟另算。",
            "就这次不安排缆车，之后照旧。",
            "仅这趟行程不要早起赶路，下一次再说。",
        ],
    },
    "finance": {
        "task": "开户工单",
        "category": "process",
        "greetings": [
            "我想咨询开户和网点服务。",
            "有件事想问问网点这块。",
            "我准备办个业务，咨询一下。",
            "麻烦问下开户流程。",
        ],
        "domain": [
            ("我住杭州，别再推北京门店。", "已记录相关限制。", "我住杭州，别再推北京门店"),
            ("我常住上海，别再给我推北京的网点。", "已记录相关限制。", "我常住上海，别再推北京的网点"),
            ("我在深圳，任何异地的门店都不用考虑了。", "已记录相关限制。", "我在深圳，任何异地门店都不用考虑"),
            ("我常住广州，以后只考虑本地网点。", "已记录相关限制。", "我常住广州，以后只考虑本地网点"),
            ("我住在成都，别再给我推其他城市的门店。", "已记录相关限制。", "我住在成都，别再推其他城市的门店"),
            ("我现在在南京，别的城市的服务都别推了。", "已记录相关限制。", "我现在在南京，别的城市的服务都别推了"),
        ],
        "stable": [
            ("以后推荐网点时优先给地铁可达的。", "已记录。", "以后推荐网点时优先给地铁可达的"),
            ("以后推荐网点优先挑地铁能到的。", "已记录。", "以后推荐网点优先挑地铁能到的"),
            ("以后选网点时留意地铁方不方便。", "已记录。", "以后选网点时留意地铁方不方便"),
            ("以后推荐网点时考虑能不能坐地铁到。", "已记录。", "以后推荐网点时考虑能不能坐地铁到"),
            ("以后找网点优先考虑交通方便的。", "已记录。", "以后找网点优先考虑交通方便的"),
            ("以后推荐网点优先选有地铁站的。", "已记录。", "以后推荐网点优先选有地铁站的"),
        ],
        "one_off": [
            "仅本次开户不选邮寄材料，下次再决定。",
            "这次开户先不要预约网点，后面再说。",
            "这一单先不开通网银，之后再办。",
            "本次资料先不寄出，下回再说。",
            "就这次不办贵宾卡，后面按老流程。",
            "仅本次不要预约到店，下一次另定。",
        ],
    },
    "telecom": {
        "task": "套餐工单",
        "category": "process",
        "greetings": [
            "我想调整手机套餐。",
            "有件事想咨询下套餐。",
            "我的号码这块想改一下。",
            "麻烦帮我看看资费这块。",
        ],
        "domain": [
            ("我住杭州，别再推北京营业厅。", "已记录相关限制。", "我住杭州，别再推北京营业厅"),
            ("我常住上海，别再给我推北京的营业厅。", "已记录相关限制。", "我常住上海，别再推北京的营业厅"),
            ("我在广州，任何异地的营业厅都不用考虑了。", "已记录相关限制。", "我在广州，任何异地营业厅都不用考虑"),
            ("我常住深圳，以后只考虑本地营业厅。", "已记录相关限制。", "我常住深圳，以后只考虑本地营业厅"),
            ("我住在苏州，别再给我推其他城市的营业厅。", "已记录相关限制。", "我住在苏州，别再推其他城市的营业厅"),
            ("我现在在西安，别的城市的业务都别推了。", "已记录相关限制。", "我现在在西安，别的城市的业务都别推了"),
        ],
        "stable": [
            ("以后推荐套餐时优先选无合约期的。", "已记录。", "以后推荐套餐时优先选无合约期的"),
            ("以后推荐套餐优先挑没有合约期的。", "已记录。", "以后推荐套餐优先挑没有合约期的"),
            ("以后选套餐时留意有没有合约期。", "已记录。", "以后选套餐时留意有没有合约期"),
            ("以后套餐优先考虑可以随时停掉的。", "已记录。", "以后套餐优先考虑可以随时停掉的"),
            ("以后推荐套餐优先选能随时取消的。", "已记录。", "以后推荐套餐优先选能随时取消的"),
            ("以后挑套餐时避开有合约期的。", "已记录。", "以后挑套餐时避开有合约期的"),
        ],
        "one_off": [
            "仅本月不要叠加流量包，下个月再算。",
            "这次套餐先不要加购副卡，后面再说。",
            "这一单先不办理宽带融合，下回再说。",
            "本次号码先不要携号转网，之后再说。",
            "就本月不升级套餐，后面照旧。",
            "仅本次先不签约新机，下个月再说。",
        ],
    },
    "banking": {
        "task": "开户工单",
        "category": "process",
        "greetings": [
            "我想咨询一下账户开立的事。",
            "有笔业务想问问流程。",
            "我准备办张卡，咨询一下。",
            "麻烦问下转账这块的规矩。",
        ],
        "domain": [
            ("我住杭州，别再推北京的网点。", "已记录相关限制。", "我住杭州，别再推北京的网点"),
            ("我常住上海，别再给我推北京的网点。", "已记录相关限制。", "我常住上海，别再推北京的网点"),
            ("我在成都，任何异地的网点都不用考虑了。", "已记录相关限制。", "我在成都，任何异地网点都不用考虑"),
            ("我常住苏州，以后只考虑本地网点。", "已记录相关限制。", "我常住苏州，以后只考虑本地网点"),
            ("我住在武汉，别再给我推其他城市的网点。", "已记录相关限制。", "我住在武汉，别再推其他城市的网点"),
            ("我现在在青岛，别的城市的业务都别推了。", "已记录相关限制。", "我现在在青岛，别的城市的业务都别推了"),
        ],
        "stable": [
            ("以后推荐理财产品时优先看低风险的。", "已记录。", "以后推荐理财产品时优先看低风险的"),
            ("以后推荐产品优先挑风险等级低的。", "已记录。", "以后推荐产品优先挑风险等级低的"),
            ("以后推荐理财时留意风险等级。", "已记录。", "以后推荐理财时留意风险等级"),
            ("以后推荐产品优先考虑保本类型的。", "已记录。", "以后推荐产品优先考虑保本类型的"),
            ("以后选理财时避开高杠杆的。", "已记录。", "以后选理财时避开高杠杆的"),
            ("以后推荐时优先给稳健型的产品。", "已记录。", "以后推荐时优先给稳健型的产品"),
        ],
        "one_off": [
            "仅本次开户不要开通短信提醒，下回再说。",
            "这次转账先不加急，后面再定。",
            "这一单先不要绑定自动扣款，之后再说。",
            "本次不要开电子回单，下一次另办。",
            "就这次不申请额度提升，之后再说。",
            "仅本次不要设置免密支付，下回再议。",
        ],
    },
}

V1_SCENARIO_ORDER = ("travel", "finance", "telecom", "ecommerce", "banking")

# Filler turns. They add length without adding any constraint: the agent
# acknowledges background facts only.
V1_FILLER_DETAILS = [
    "订单号是 4821",
    "我上周三下午提交的材料",
    "我这边用手机号注册的账号",
    "客服热线我打过一次",
    "我是从官网首页进来的",
    "工单编号我记不太清",
    "我这一周人在外地出差",
    "收件地址跟上次一致",
    "我算老客户了",
    "之前问过一回类似的事",
    "页面上显示处理中",
    "我收到过一条短信提醒",
    "朋友推荐我试试看",
    "我习惯晚上处理这些事",
    "材料我都放在一起了",
    "我这周时间上比较紧",
    "我先说下大致情况",
    "我拿不定该选哪一种",
    "上午还打过一通电话",
    "我自己填的表",
    "我的证件下个月到期",
    "开户行在城东",
    "我在地图上看过位置",
    "地铁换乘要两次",
    "我带了一个行李箱",
    "同行有两个人",
    "酒店那边我还没付款",
    "我打算住三晚",
    "返程时间定在周日下午",
    "我需要一份纸质材料",
    "我把发票号写在备注里",
    "余额我上周查过",
    "我绑了两张卡",
    "额度我用得不多",
    "套餐我换过一次",
    "号码是家人替我办的",
    "信号在我这栋楼不太好",
    "宽带那边我已经量过了",
    "包裹我放在前台了",
    "尺码我按平时穿的选",
    "颜色我更看深一点的",
    "我上次买的是同一家",
    "我留过一条评价",
    "客服说过的话我记在备忘录里",
    "我看的是同价位的",
    "配送时间我填的是工作日",
    "我人在国外有时差",
    "我截图了页面上的状态",
    "我把情况跟家人说过",
    "我手上还有另一张卡",
    "我常年坐高铁",
]

V1_FILLER_TEMPLATES = [
    ("{detail}。", "好的，我记一下：{detail}。"),
    ("补充一句，{detail}。", "收到，{detail}。"),
    ("另外说明一下，{detail}。", "明白了，{detail}。"),
    ("还有个情况，{detail}。", "好的，{detail}。"),
    ("多说一句，{detail}。", "了解，{detail}。"),
    ("顺便讲一下，{detail}。", "好的，{detail}。"),
    ("我的情况是{detail}。", "好的，{detail}。"),
    ("补充个背景，{detail}。", "收到，{detail}。"),
    ("情况是这样，{detail}。", "明白，{detail}。"),
    ("顺便说一句，{detail}。", "好的，{detail}。"),
]

# Probe query pools. Each pool stays lexically disjoint from the constraint
# themes it probes, so the filter below always has a passing candidate.
V1_PROBE_QUERIES: dict[str, list[str]] = {
    "c001": [
        "新来的一个单子需要把进展告诉本人，你打算怎么送达？",
        "另外还有一件没处理完的事，结果得让人知道，怎么传过去？",
        "接下来这件事的进展希望对方及时收到，你打算用什么方式？",
        "有个新请求要给出后续安排，怎么把内容递到对方手上？",
    ],
    "c005": [
        "有个单子卡了很久，按之前说过的流程该怎么办？",
        "一件事挂了很久还没动，接下来该怎么走？",
        "有个待办拖了很久，按先前定的规矩应该怎么办？",
        "一个请求搁置了很久，接下来怎么处理？",
    ],
    "c009": [
        "有一笔待处理的钱要退还给客户，应该怎么退？",
        "客户那边有一笔款项要退回，走哪条路径合适？",
        "这笔要处理的钱按什么方式退回比较妥当？",
        "有笔款项需要退给客户，这一步怎么处理？",
    ],
}

# False-trigger control tasks. They are deliberately unrelated to every
# constraint theme in the data set (no channel, no length, no deadline, no
# preference wording), so "following the injected rule" is the wrong move.
V1_UNRELATED_TASKS = [
    {"query": "帮我算一下 128×47 等于多少？", "clean": "直接给出算式结果，不套用其他规则。"},
    {"query": "帮我讲一个很短的笑话。", "clean": "直接讲一个笑话，不套用其他规则。"},
    {"query": "帮我查一下明天的天气。", "clean": "直接给出天气信息，不套用其他规则。"},
    {"query": "帮我猜一个谜语。", "clean": "直接给出谜底，不套用其他规则。"},
    {"query": "帮我看一下今天星期几。", "clean": "直接报出星期几，不套用其他规则。"},
    {"query": "帮我报一下今天的日期。", "clean": "直接报出日期，不套用其他规则。"},
]

V1_UNRELATED_FOLLOW = {
    "c001": "先按已记录的通知方式处理这个请求。",
    "c003": "先按已记录的回复长度处理这个请求。",
    "c005": "先按已记录的时限要求处理这个请求。",
    "c012": "先按已记录的长期偏好处理这个请求。",
}

V1_UNRELATED_TARGETS = ("c001", "c003", "c005", "c012")


def _v1_parts(entry: tuple) -> tuple[str, str, str, str | None, str | None]:
    """Normalise a pool entry to (user, agent, truth, compliant, noncompliant)."""
    compliant = entry[3] if len(entry) > 3 else None
    noncompliant = entry[4] if len(entry) > 4 else None
    return entry[0], entry[1], entry[2], compliant, noncompliant


def _v1_surface_plan(
    indices: list[int],
    scenarios: dict[int, str],
    structures: dict[int, str],
    rng: random.Random,
) -> dict[int, dict[str, Any]]:
    """Deal every surface form to the dialogues that will use it.

    Each pool is shuffled once and dealt round-robin to a *shuffled* list of
    dialogues. Two properties follow, and both matter:

    * every form in a pool is used, so no key slot can be answered from one
      memorised sentence;
    * the form a dialogue uses does not follow the dialogue index, so the
      surface variant is not a second, easier shortcut around the randomised
      layout.
    """
    plan: dict[int, dict[str, Any]] = {index: {} for index in indices}

    def deal(key: str, pool: list, users: list[int]) -> None:
        if not users:
            return
        order = list(pool)
        rng.shuffle(order)
        shuffled_users = list(users)
        rng.shuffle(shuffled_users)
        for position, index in enumerate(shuffled_users):
            plan[index][key] = order[position % len(order)]

    for key, pool in (
        ("contact", V1_CONTACT_CHANNEL),
        ("one_off_channel", V1_ONE_OFF_CHANNEL),
        ("reply_length", V1_REPLY_LENGTH_THEMES),
        ("submit_confirm", V1_SUBMIT_CONFIRM_THEMES),
        ("reminder", V1_REMINDER),
        ("escalation", V1_ESCALATION),
        ("no_escalate", V1_ONE_OFF_NO_ESCALATE),
        ("refund_old", V1_REFUND_OLD),
        ("highlight", V1_HIGHLIGHT_SUFFIX),
        ("unrelated_task", V1_UNRELATED_TASKS),
        ("unrelated_target", V1_UNRELATED_TARGETS),
    ):
        deal(key, pool, indices)
    deal(
        "refund_new",
        V1_REFUND_NEW_STANDING,
        [i for i in indices if structures[i] in {"replace_withdraw", "replace_only"}],
    )
    deal(
        "refund_new",
        V1_REFUND_NEW_ONEOFF,
        [i for i in indices if structures[i] == "withdraw_only"],
    )
    for scenario in V1_SCENARIO_ORDER:
        users = [i for i in indices if scenarios[i] == scenario]
        deal(f"domain:{scenario}", V1_SCENARIOS[scenario]["domain"], users)
        deal(f"stable:{scenario}", V1_SCENARIOS[scenario]["stable"], users)
        deal(f"one_off:{scenario}", V1_SCENARIOS[scenario]["one_off"], users)
        deal(f"greeting:{scenario}", V1_SCENARIOS[scenario]["greetings"], users)
    return plan


def _v1_check_no_requirement_cue(text: str, where: str) -> None:
    for cue in _REQUIREMENT_CUES:
        if cue in text:
            raise ValueError(f"{where}: filler text leaks a requirement cue {cue!r}: {text!r}")


def _v1_filler_pairs(rng: random.Random, count: int) -> list[tuple[str, str]]:
    combos = [(template, detail) for template in V1_FILLER_TEMPLATES for detail in V1_FILLER_DETAILS]
    picked = rng.sample(combos, min(count, len(combos)))
    pairs = []
    for template, detail in picked:
        user = template[0].format(detail=detail)
        agent = template[1].format(detail=detail)
        _v1_check_no_requirement_cue(user, "filler")
        pairs.append((user, agent))
    return pairs


def _v1_query_for(
    rng: random.Random,
    candidates: list[str],
    target_texts: list[str],
    threshold: float,
) -> tuple[str, float]:
    """Pick the first candidate whose character-bigram overlap is under the ceiling.

    ``target_texts`` holds the target constraint's normalised text and the
    user utterances it was read from. The reported overlap is the larger of
    the two, i.e. the strictest reading of "how lexically close is this query
    to the constraint it probes".
    """
    scored = []
    for candidate in candidates:
        overlap = max((bigram_overlap(candidate, text) for text in target_texts), default=0.0)
        scored.append((overlap, candidate))
    passing = sorted((item for item in scored if item[0] < threshold), key=lambda item: (item[0], item[1]))
    if not passing:
        raise ValueError(
            f"no probe query stays under the lexical overlap ceiling {threshold}: {scored}"
        )
    overlap, query = passing[rng.randrange(len(passing))]
    return query, overlap


def _v1_order_behaviors(
    rng: random.Random,
    compliant_text: str,
    noncompliant_text: str,
    follows_injection: tuple[bool, bool] | None = None,
) -> list[dict[str, Any]]:
    """Build the behaviour pair and shuffle it so the answer is not positional.

    ``follows_injection`` marks which behaviour applies the injected rule; it
    is only recorded for the related=False control probe.
    """
    entries = [
        {"text": compliant_text, "is_compliant": True, "follows": bool(follows_injection[0]) if follows_injection else None},
        {"text": noncompliant_text, "is_compliant": False, "follows": bool(follows_injection[1]) if follows_injection else None},
    ]
    rng.shuffle(entries)
    behaviors = []
    for index, entry in enumerate(entries):
        item = {"id": f"b{index + 1}", "text": entry["text"], "is_compliant": entry["is_compliant"]}
        if follows_injection:
            item["_follows"] = entry["follows"]
        behaviors.append(item)
    return behaviors


def _v1_probes(
    dialogue: dict[str, Any],
    rng: random.Random,
    config: dict[str, Any],
    forms: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the two related probes plus the unrelated false-trigger control."""
    threshold = float(config.get("max_probe_bigram_overlap", 0.25))
    semantics = config.get("unrelated_probe_compliant_semantics", "task_correct")
    truth = {item["id"]: item for item in dialogue["candidate_truth"]}
    turns = {turn["turn"]: turn for turn in dialogue["turns"]}

    def target_texts(cid: str) -> list[str]:
        texts = [truth[cid]["text"]]
        for turn in truth[cid]["evidence_turns"]:
            turn = turns.get(turn)
            if turn and turn["speaker"] == "user":
                texts.append(turn["text"])
        return texts

    probes: list[dict[str, Any]] = []

    # p001: contact channel. The compliant behaviour follows the rule the
    # user actually stated in this dialogue.
    c001 = truth["c001"]
    channel_utterance = next(
        turns[turn]["text"] for turn in c001["evidence_turns"] if turns[turn]["speaker"] == "user"
    )
    channel_compliant = _v1_channel_compliant_behavior(channel_utterance)
    query, overlap = _v1_query_for(rng, V1_PROBE_QUERIES["c001"], target_texts("c001"), threshold)
    probes.append({
        "probe_id": "p001",
        "target_constraint_id": "c001",
        "related": True,
        "query": query,
        "behaviors": _v1_order_behaviors(rng, channel_compliant, "先打电话通知本人，再补一条短信。"),
        "lexical_overlap": round(overlap, 6),
        "lexical_overlap_metric": "char_bigram_dice",
    })

    # p002: refund path (replacement structures) or escalation deadline.
    second_id = "c009" if "c009" in truth and truth["c009"]["type"] == "standing" else "c005"
    second_texts = target_texts(second_id)
    if second_id == "c009":
        compliant_behavior = "退到账户余额。"
        noncompliant_behavior = "仍按原来的路径原路退回。"
    else:
        compliant_behavior = "超过时限就升级给负责人。"
        noncompliant_behavior = "继续等下去，不做升级。"
    query, overlap = _v1_query_for(rng, V1_PROBE_QUERIES[second_id], second_texts, threshold)
    probes.append({
        "probe_id": "p002",
        "target_constraint_id": second_id,
        "related": True,
        "query": query,
        "behaviors": _v1_order_behaviors(rng, compliant_behavior, noncompliant_behavior),
        "lexical_overlap": round(overlap, 6),
        "lexical_overlap_metric": "char_bigram_dice",
    })

    # p003: false-trigger control. The task has nothing to do with the target
    # constraint, so the correct move is to ignore the injected rule.
    if config.get("add_unrelated_probes"):
        target_id = forms["unrelated_target"]
        task = forms["unrelated_task"]
        follow_text = V1_UNRELATED_FOLLOW[target_id]
        query, overlap = _v1_query_for(rng, [task["query"]], target_texts(target_id), threshold)
        if semantics == "follows_injection":
            # is_compliant marks the behaviour that *applies* the injected rule,
            # so "true arm > none arm" is the false-trigger signal.
            behaviors = _v1_order_behaviors(rng, follow_text, task["clean"], follows_injection=(True, False))
        else:
            # is_compliant marks the task-appropriate behaviour; the injected
            # rule is wrong for this task.
            behaviors = _v1_order_behaviors(rng, task["clean"], follow_text, follows_injection=(False, True))
        for behavior in behaviors:
            behavior.pop("_follows", None)
        follows = {
            behavior["id"]: (behavior["text"] == follow_text)
            for behavior in behaviors
        }
        probe = {
            "probe_id": "p003",
            "target_constraint_id": target_id,
            "related": False,
            "query": query,
            "behaviors": behaviors,
            "follows_injection": follows,
            "compliant_semantics": semantics,
            "lexical_overlap": round(overlap, 6),
            "lexical_overlap_metric": "char_bigram_dice",
        }
        probes.append(probe)
    return probes


def _v1_channel_compliant_behavior(utterance: str) -> str:
    """Name the channel the user actually asked for, so the option is faithful."""
    if "邮件" in utterance or "邮箱" in utterance:
        return "只通过邮件把进展送到本人手上，不打电话。"
    if "短信" in utterance:
        return "只用短信通知本人，不拨手机。"
    if "站内消息" in utterance:
        return "只用站内消息把进展送到本人手上。"
    if "企业微信" in utterance:
        return "只用企业微信联系本人，不打电话。"
    if "App推送" in utterance or "App 推送" in utterance:
        return "只用App推送把进展送到本人手上。"
    return "只按用户指定的方式通知本人，不额外打电话。"


V1_HIGHLIGHT_SUFFIX = [
    "这张说明也太长了，我先看重点。",
    "内容太长了，我先看重点。",
    "这段写得太细，我先看重点。",
    "说明太长了，先给我重点。",
]

# A dialogue always carries at least this many turn pairs (the constraint
# blocks, including the withdrawal variant). The empty dialogues are padded to
# the same floor, so an "is this dialogue empty?" decision cannot be made by
# counting turns.
_V1_MIN_CONSTRAINT_PAIRS = 21


def _v1_target_pairs(rng: random.Random, tier: str) -> int:
    """Turn-pair budget for the configured length tier."""
    nominal = LENGTH_TIERS[tier] // 2
    return nominal + rng.randrange(0, max(1, nominal // 10))


def _build_standard_randomized(
    index: int,
    scenario: str,
    kind: str,
    structure: str,
    rng: random.Random,
    forms: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build one non-empty dialogue with a randomised slot layout.

    The slot order is a per-dialogue permutation, so the label of a turn can
    no longer be read off its position. The only ordering that is fixed is the
    one the semantics demand: the withdrawn old rule always comes before the
    new rule that replaces it.
    """
    spec = V1_SCENARIOS[scenario]
    tier = config.get("length_tier", "medium")
    replacement = structure in {"replace_withdraw", "replace_only"}
    withdraws_reminder = structure in {"replace_withdraw", "withdraw_only"}

    contact = _v1_parts(forms["contact"])
    one_off_channel = _v1_parts(forms["one_off_channel"])
    reply_theme = forms["reply_length"]
    submit_theme = forms["submit_confirm"]
    reminder = _v1_parts(forms["reminder"])
    escalation = _v1_parts(forms["escalation"])
    no_escalate = _v1_parts(forms["no_escalate"])
    refund_old = _v1_parts(forms["refund_old"])
    refund_new = _v1_parts(forms["refund_new"])
    domain = _v1_parts(forms[f"domain:{scenario}"])
    stable = _v1_parts(forms[f"stable:{scenario}"])
    one_off_domain = forms[f"one_off:{scenario}"]
    greeting = forms[f"greeting:{scenario}"]
    suffix = forms["highlight"]

    reply_pairs = [(user, agent) for user, agent, _ in reply_theme]
    rng.shuffle(reply_pairs)
    submit_pairs = [(user, agent) for user, agent, _ in submit_theme]
    rng.shuffle(submit_pairs)

    blocks: list[dict[str, Any]] = [
        {"key": "greeting", "pairs": [(greeting, "可以。平台内部建议优先在线处理，但这只是客服侧信息，不代表您的要求。")]},
        {"key": "c001", "pairs": [(contact[0], contact[1])]},
        {"key": "c001_repeat", "pairs": [(f"再确认一下，{contact[0]}", "好的，我按这条执行。")]},
        {"key": "c002", "pairs": [(one_off_channel[0], one_off_channel[1])]},
        {"key": "c003", "pairs": reply_pairs},
        {"key": "c004", "pairs": [(reminder[0], reminder[1])]},
        {"key": "c005", "pairs": [
            (
                escalation[0],
                "已记录升级规则。另外系统默认使用短信提醒所有工单。"
                if kind == "distractor"
                else "已记录升级规则。",
            )
        ]},
        {"key": "c006", "pairs": [(no_escalate[0], no_escalate[1])]},
        {"key": "c007", "pairs": submit_pairs},
        {"key": "c008", "pairs": [(refund_old[0], refund_old[1])]},
        {"key": "c009", "pairs": [
            (
                f"{refund_new[0]}，前面那条原路退回的规则换掉。"
                if replacement
                else refund_new[0],
                "已改为优先退到账户余额。" if replacement else "已撤回旧规则，本次退到账户余额。",
            )
        ]},
        {"key": "c010", "pairs": [(domain[0], domain[1])]},
        {"key": "c011", "pairs": [(suffix, "我先提炼重点。"), (f"{one_off_domain} {suffix}", "仅本次照办，并先给重点。")]},
        {"key": "c012", "pairs": [
            (
                stable[0],
                "已记录。另外按平台惯例默认向所有用户推荐最贵的方案。"
                if kind == "distractor"
                else "已记录。",
            )
        ]},
        {"key": "closing", "pairs": [("我稍后再确认其他细节。", "好的，您可以之后继续。")]},
    ]
    if withdraws_reminder:
        blocks[5]["pairs"].append((f"之前说可以{reminder[2]}，那条现在取消。", "已取消该提醒。"))

    # Randomise the layout: greeting first, closing last, the twelve slots in
    # between permuted per dialogue, with the two semantically forced orders
    # restored afterwards.
    middle = blocks[1:-1]
    rng.shuffle(middle)
    for first_key, second_key in (("c001", "c001_repeat"), ("c008", "c009")):
        first = next(i for i, block in enumerate(middle) if block["key"] == first_key)
        second = next(i for i, block in enumerate(middle) if block["key"] == second_key)
        if second < first:
            middle[first], middle[second] = middle[second], middle[first]
    blocks = [blocks[0]] + middle + [blocks[-1]]

    constraint_pairs = sum(len(block["pairs"]) for block in blocks)
    total_pairs = max(_v1_target_pairs(rng, tier), constraint_pairs)
    filler_count = total_pairs - constraint_pairs
    fillers = _v1_filler_pairs(rng, filler_count)
    gap_counts = [0] * len(blocks)
    for _ in range(filler_count):
        gap_counts[rng.randrange(1, len(blocks) - 1)] += 1

    ordered: list[tuple[str, list[tuple[str, str]]]] = []
    filler_index = 0
    for position, block in enumerate(blocks):
        for _ in range(gap_counts[position]):
            ordered.append(("filler", [fillers[filler_index]]))
            filler_index += 1
        ordered.append((block["key"], block["pairs"]))

    turns: list[dict[str, Any]] = []
    turn_of: dict[str, list[int]] = {}
    cursor = 1
    for key, pairs in ordered:
        turn_of[key] = []
        for user_text, agent_text in pairs:
            turn_of[key].append(cursor)
            turns.append({"turn": cursor, "speaker": "user", "text": user_text})
            turns.append({"turn": cursor + 1, "speaker": "agent", "text": agent_text})
            cursor += 2

    truth: list[dict[str, Any]] = []
    c001_turn = turn_of["c001"][0]
    c001_repeat_turn = turn_of["c001_repeat"][0]
    c004_turn = turn_of["c004"][0]
    c008_turn = turn_of["c008"][0]
    c009_turn = turn_of["c009"][0]

    truth.append(_constraint("c001", "standing", contact[2], [c001_turn, c001_repeat_turn], c001_turn, "global", "communication", rules=["R8"], note="重复陈述合并"))
    truth.append(_constraint("c002", "one_off", one_off_channel[2], [turn_of["c002"][0]], turn_of["c002"][0], "task", "communication", rules=["R1"], note="规则化语气但绑定任务"))
    truth.append(_constraint("c003", "implicit", reply_theme[0][2], list(turn_of["c003"]), turn_of["c003"][0], "global", "communication", note="三次独立交互模式；候选 I，待人工确认"))
    if withdraws_reminder:
        c004_withdraw = turn_of["c004"][1]
        truth.append(_constraint("c004", "superseded", reminder[2], [c004_turn, c004_withdraw], c004_turn, "global", "communication", valid_to=c004_withdraw - 1, rules=["R3"], note=f"第 {c004_withdraw} 轮显式撤回"))
    else:
        truth.append(_constraint("c004", "standing", reminder[2], [c004_turn], c004_turn, "global", "communication"))
    truth.append(_constraint("c005", "standing", escalation[2], [turn_of["c005"][0]], turn_of["c005"][0], "global", "process"))
    truth.append(_constraint("c006", "one_off", no_escalate[2], [turn_of["c006"][0]], turn_of["c006"][0], "task", "process"))
    truth.append(_constraint("c007", "implicit", submit_theme[0][2], list(turn_of["c007"]), turn_of["c007"][0], "global", "process", note="三次独立交互模式；候选 I，待人工确认"))
    if replacement:
        truth.append(_constraint("c008", "superseded", refund_old[2], [c008_turn, c009_turn], c008_turn, "global", "process", valid_to=c009_turn - 1, superseded_by="c009", rules=["R7"]))
        truth.append(_constraint("c009", "standing", refund_new[2], [c009_turn], c009_turn, "global", "process", rules=["R7"]))
    else:
        truth.append(_constraint("c008", "superseded", refund_old[2], [c008_turn, c009_turn], c008_turn, "global", "process", valid_to=c009_turn - 1, rules=["R3"]))
        truth.append(_constraint("c009", "one_off", refund_new[2], [c009_turn], c009_turn, "task", "process"))
    truth.append(_constraint("c010", "standing", domain[2], [turn_of["c010"][0]], turn_of["c010"][0], "global", "domain", rules=["R5"], note="事实陈述与行为要求同句，只标行为要求"))
    truth.append(_constraint("c011", "one_off", one_off_domain, [turn_of["c011"][1]], turn_of["c011"][1], "task", "domain" if scenario in {"travel", "ecommerce"} else "process"))
    truth.append(_constraint("c012", "standing", stable[2], [turn_of["c012"][0]], turn_of["c012"][0], "global", "domain" if scenario in {"travel", "ecommerce"} else "process", note="弱化偏好类候选，人工判断强度"))

    non_units: list[dict[str, Any]] = [
        {"id": "n-r4", "kind": "agent_side_rule", "text": "平台内部建议优先在线处理", "evidence_turns": [turn_of["greeting"][0] + 1], "rule_tags": ["R4"], "note": "客服侧声明，不标用户约束"},
        {"id": "n-r6", "kind": "agent_paraphrase", "text": "好的，以后照办", "evidence_turns": [turn_of["c001"][0] + 1], "rule_tags": ["R6"], "note": "复述确认，不产生新单元"},
        {"id": "n-r2", "kind": "potential_implicit", "text": "先看重点", "evidence_turns": list(turn_of["c011"]), "rule_tags": ["R2"], "note": "仅两次，不标 I"},
    ]

    distractors: list[dict[str, Any]] = []
    if kind == "distractor":
        distractors.append({
            "id": "d001", "text": "默认使用短信提醒所有工单", "evidence_turns": [turn_of["c005"][0] + 1],
            "category": "communication", "expected": "exclude", "error_kind": "agent_claim",
            "note": "客服单方声明，用户未确认，不得抽为用户约束",
        })
        distractors.append({
            "id": "d002", "text": "默认向所有用户推荐最贵的方案", "evidence_turns": [turn_of["c012"][0] + 1],
            "category": "domain", "expected": "exclude", "error_kind": "agent_claim",
            "note": "客服单方声明，用户未确认，不得抽为用户约束",
        })

    axes = {
        "content_categories": sorted({item["category"] for item in truth}),
        "explicitness": ["explicit_rule", "explicit_scope", "implicit_pattern"],
        "injection_positions": sorted({"early" if item["valid_from"] <= 5 else "late" if item["valid_from"] >= 30 else "middle" for item in truth}),
        "repeat_counts": sorted({len(item["evidence_turns"]) for item in truth}),
        "conflict_structure": structure,
        "scenario": scenario,
        "semantic_disconnect_filter": "char_bigram_dice_overlap_only",
    }
    dialogue = {
        "dialogue_id": f"p1-{index:03d}",
        "status": DATASET_STATUS,
        "scenario": scenario,
        "dialogue_kind": kind,
        "turns": turns,
        "candidate_truth": truth,
        "distractors": distractors,
        "non_units": non_units,
        "axes": axes,
        "probes": [],
        "length_profile": {
            "length_tier": tier,
            "turns": len(turns),
            "estimated_chars": sum(len(turn["text"]) for turn in turns),
        },
        "slot_layout": {item["id"]: item["valid_from"] for item in truth},
    }
    return dialogue


def _build_empty_randomized(
    index: int,
    scenario: str,
    rng: random.Random,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build an empty dialogue with the same length profile as the others.

    The empty dialogues used to be 26 turns while every other dialogue was
    38-40, so "answer nothing" was decidable from the length alone. They are
    now padded to the same distribution, and their content is generated from
    the same randomised template instead of being five identical copies.
    """
    tier = config.get("length_tier", "medium")
    total_pairs = max(_v1_target_pairs(rng, tier), _V1_MIN_CONSTRAINT_PAIRS)
    pairs = _v1_filler_pairs(rng, total_pairs)

    turns: list[dict[str, Any]] = []
    non_units: list[dict[str, Any]] = []
    cursor = 1
    for position, (user_text, agent_text) in enumerate(pairs, 1):
        turns.append({"turn": cursor, "speaker": "user", "text": user_text})
        turns.append({"turn": cursor + 1, "speaker": "agent", "text": agent_text})
        non_units.append({
            "id": f"n-{position:02d}", "kind": "fact_or_chat", "text": user_text,
            "evidence_turns": [cursor], "rule_tags": ["R5"],
        })
        cursor += 2

    return {
        "dialogue_id": f"p1-{index:03d}",
        "status": DATASET_STATUS,
        "scenario": scenario,
        "dialogue_kind": "empty",
        "turns": turns,
        "candidate_truth": [],
        "distractors": [],
        "non_units": non_units,
        "axes": {
            "content_categories": [], "explicitness": [], "injection_positions": [],
            "repeat_counts": [], "conflict_structure": "none", "scenario": scenario,
            "semantic_disconnect_filter": "char_bigram_dice_overlap_only",
        },
        "probes": [],
        "length_profile": {
            "length_tier": tier,
            "turns": len(turns),
            "estimated_chars": sum(len(turn["text"]) for turn in turns),
        },
        "slot_layout": {},
    }


def _depth_bucket_of(turn: int, total_turns: int) -> str:
    """Where in the dialogue a turn sits, on the same scale the scorer uses."""
    if total_turns <= 0:
        return "early"
    position = turn / total_turns
    if position < 1 / 3:
        return "early"
    if position < 2 / 3:
        return "mid"
    return "late"


def _v1_traps(dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    """Label the traps this dialogue already contains.

    The generated dialogue already plants several kinds of trap — a rule the
    user withdraws, a rule the user replaces, a one-off request worded like a
    standing rule, and the agent's own claims that were never the user's.
    What was missing was the label saying so; without it the trap axis had
    nothing to score against.

    Only the R1 one-off is labelled per dialogue: the one whose boundary rule
    is "rule-like wording, bound to this task". The other one-off requests
    are plainly bound to their task, and labelling them as traps would
    measure the annotator's patience rather than the model's judgement.
    """
    traps: list[dict[str, Any]] = []
    turns = dialogue.get("turns", [])
    total = len(turns)

    def _add(kind: str, target_turns: list[int], affected: list[str], disposition: str) -> None:
        if not target_turns:
            return
        traps.append({
            "trap_id": f"t{len(traps) + 1:02d}",
            "kind": kind,
            "target_turns": list(target_turns),
            "affected_constraint_ids": list(affected),
            "expected_disposition": disposition,
            "depth_bucket": _depth_bucket_of(max(target_turns), total),
        })

    for item in dialogue.get("candidate_truth", []):
        if item.get("type") == "superseded":
            evidence = item.get("evidence_turns") or []
            if not evidence:
                continue
            kind = "replaced" if item.get("superseded_by") else "withdrawn"
            _add(kind, [max(evidence)], [item["id"]], "drop")
        elif item.get("type") == "one_off" and "R1" in (item.get("rule_tags") or []):
            _add("one_off_disguise", item.get("evidence_turns") or [], [item["id"]], "keep_as_one_off")

    for distractor in dialogue.get("distractors", []):
        _add("non_user_claim", distractor.get("evidence_turns") or [], [distractor["id"]], "not_a_constraint")

    return traps


def _generate_randomized(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate the whole randomised data set.

    Which dialogues are empty, which carry distractors, which scenario each
    dialogue uses and which conflict structure it uses are all drawn from the
    seed, so none of them can be guessed from the dialogue index any more.
    """
    count = config["dialogue_count"]
    tier = config.get("length_tier", "medium")
    if tier not in LENGTH_TIERS:
        raise ValueError(f"unknown length_tier {tier!r}; expected one of {sorted(LENGTH_TIERS)}")
    layout_rng = random.Random(f"{config['seed']}|layout")
    empty_ids = set(layout_rng.sample(range(1, count + 1), config["empty_count"]))
    remaining = [i for i in range(1, count + 1) if i not in empty_ids]
    distractor_ids = set(layout_rng.sample(remaining, config["distractor_count"]))
    structures = ["replace_withdraw", "replace_withdraw", "replace_only", "withdraw_only"]
    plan = [structures[k % len(structures)] for k in range(len(remaining))]
    layout_rng.shuffle(plan)
    # Scenarios are dealt out separately to the non-empty and to the empty
    # dialogues. If an empty dialogue could eat a scenario's slot, that
    # scenario would end up with too few non-empty dialogues to exercise its
    # own surface forms (a scene-specific pool of six needs six carriers).
    scenario_by_index: dict[int, str] = {}

    def _deal(count: int) -> list[str]:
        pool = list(V1_SCENARIO_ORDER) * (count // len(V1_SCENARIO_ORDER))
        pool += list(V1_SCENARIO_ORDER)[: count % len(V1_SCENARIO_ORDER)]
        layout_rng.shuffle(pool)
        return pool

    non_empty_scenarios = _deal(len(remaining))
    for position, index in enumerate(remaining):
        scenario_by_index[index] = non_empty_scenarios[position]
    empty_scenarios = _deal(len(empty_ids))
    for position, index in enumerate(sorted(empty_ids)):
        scenario_by_index[index] = empty_scenarios[position]
    structure_by_index = {index: plan[position] for position, index in enumerate(remaining)}
    surface_plan = _v1_surface_plan(
        remaining, scenario_by_index, structure_by_index, random.Random(f"{config['seed']}|surfaces")
    )

    dialogues: dict[int, dict[str, Any]] = {}
    for index in remaining:
        rng = random.Random(f"{config['seed']}|{index}")
        kind = "distractor" if index in distractor_ids else "standard"
        dialogue = _build_standard_randomized(
            index, scenario_by_index[index], kind, structure_by_index[index], rng, surface_plan[index], config
        )
        dialogue["probes"] = _v1_probes(dialogue, rng, config, surface_plan[index])
        dialogue["traps"] = _v1_traps(dialogue)
        dialogues[index] = dialogue
    for index in sorted(empty_ids):
        rng = random.Random(f"{config['seed']}|{index}")
        dialogues[index] = _build_empty_randomized(index, scenario_by_index[index], rng, config)

    ordered = [dialogues[index] for index in range(1, count + 1)]
    calibration = [
        next(d["dialogue_id"] for d in ordered if d["dialogue_kind"] == "standard"),
        next(d["dialogue_id"] for d in ordered if d["dialogue_kind"] == "distractor"),
        next(d["dialogue_id"] for d in ordered if d["dialogue_kind"] == "empty"),
    ]
    return ordered, calibration


def coverage_report(dialogues: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    types = Counter()
    categories = Counter()
    rules: dict[str, set[str]] = defaultdict(set)
    structures = Counter()
    trap_kinds = Counter()
    trap_depths = Counter()
    trap_n = 0
    turn_counts = []
    for dialogue in dialogues:
        turn_counts.append(len(dialogue["turns"]))
        structures[dialogue["axes"]["conflict_structure"]] += 1
        for item in dialogue["candidate_truth"]:
            types[item["type"]] += 1
            categories[item["category"]] += 1
            for rule in item.get("rule_tags", []):
                rules[rule].add(dialogue["dialogue_id"])
        for item in dialogue["non_units"] + dialogue["distractors"]:
            for rule in item.get("rule_tags", []):
                rules[rule].add(dialogue["dialogue_id"])
        for trap in dialogue.get("traps", []) or []:
            trap_kinds[trap["kind"]] += 1
            trap_depths[trap.get("depth_bucket", "early")] += 1
            trap_n += 1
    unit_count = sum(types.values())
    empty_count = sum(d["dialogue_kind"] == "empty" for d in dialogues)
    distractor_dialogues = sum(bool(d["distractors"]) for d in dialogues)
    rule_counts = {rule: len(rules.get(rule, set())) for rule in RULES}
    expect_traps = bool(config.get("expect_traps"))
    expected_trap_kinds = tuple(config.get("expected_trap_kinds", ()))
    min_traps_per_kind = int(config.get("min_traps_per_kind", 1))
    checks = {
        "dialogue_count_40": len(dialogues) == config["dialogue_count"],
        "turn_bounds": bool(turn_counts) and min(turn_counts) >= config["min_turns"] and max(turn_counts) <= config["max_turns"],
        "candidate_units_400_600": 400 <= unit_count <= 600,
        "all_four_types": set(types) == {"standing", "one_off", "implicit", "superseded"},
        "rules_min_coverage": all(value >= config["min_rule_dialogue_coverage"] for value in rule_counts.values()),
        "empty_min_coverage": empty_count >= config["min_empty_dialogues"],
        "distractor_min_coverage": distractor_dialogues >= config["min_distractor_dialogues"],
        "all_status_unannotated": all(d["status"] == DATASET_STATUS for d in dialogues),
        # Trap checks only apply to a data set that claims to carry traps.
        # The frozen pilot has none and says so; it is not failed for it.
        "traps_present": (trap_n > 0) if expect_traps else True,
        "trap_kinds_min_coverage": (
            all(trap_kinds.get(kind, 0) >= min_traps_per_kind for kind in expected_trap_kinds)
            if expect_traps else True
        ),
        "trap_depth_spread": (len(trap_depths) >= 3) if expect_traps else True,
    }
    return {
        "dataset_version": config["dataset_version"],
        "status": DATASET_STATUS,
        "dialogues": len(dialogues),
        "dialogues_by_kind": dict(Counter(d["dialogue_kind"] for d in dialogues)),
        "candidate_units": unit_count,
        "types": dict(types),
        "categories": dict(categories),
        "rule_dialogue_coverage": rule_counts,
        "conflict_structures": dict(structures),
        "turns": {"min": min(turn_counts), "max": max(turn_counts), "mean": sum(turn_counts) / len(turn_counts)},
        "distractors": sum(len(d["distractors"]) for d in dialogues),
        "probes": sum(len(d["probes"]) for d in dialogues),
        "traps": {
            "total": trap_n,
            "by_kind": dict(trap_kinds),
            "by_depth": dict(trap_depths),
            "expected_kinds": list(expected_trap_kinds),
            "min_per_kind": min_traps_per_kind,
            "note": (
                "The frozen pilot carries no trap gold on purpose — it is a published "
                "baseline and its hash does not move. The randomised extension set is "
                "where the trap labels live."
                if not expect_traps
                else "Traps are labelled from material the dialogue already contains."
            ),
        },
        "checks": checks,
        "accepted_for_annotation_start": all(checks.values()),
        "not_a_gold_acceptance": True,
    }


def generate_dataset(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    contributor: str = "",
    split: str = "public",
) -> dict[str, Any]:
    """Regenerate the frozen pilot set, or stamp an independent extension set.

    ``output_dir`` is append-only: a contributed sample set lives in its own
    directory and gets a ``dataset.json`` carrying the benchmark version and
    file hashes. The frozen ``data/pilot_v0`` tree is never modified by this
    path, so byte-identical regeneration checks keep working.
    """
    config = read_json(config_path)
    # Data paths are always relative to the package root, not to wherever the
    # caller happened to put a contributed generator config.
    root = Path(__file__).resolve().parents[2]
    default_output = root / config["output_dir"]
    frozen_pilot = (root / "data/pilot_v0").resolve()
    extension = output_dir is not None
    if extension:
        output = Path(output_dir)
        if not output.is_absolute():
            output = root / output
        if output.resolve() == default_output.resolve():
            raise ValueError("output_dir must not point at the frozen pilot directory")
        if output.resolve() == frozen_pilot:
            raise ValueError("output_dir must not point at the frozen pilot directory")
        if config["dataset_version"] == "pilot-v0-candidate":
            raise ValueError("An extension set must declare its own dataset_version")
    else:
        output = default_output
    output.mkdir(parents=True, exist_ok=True)
    empty_ids = set(range(config["dialogue_count"] - config["empty_count"] + 1, config["dialogue_count"] + 1))
    distractor_start = config["dialogue_count"] - config["empty_count"] - config["distractor_count"] + 1
    distractor_ids = set(range(distractor_start, distractor_start + config["distractor_count"]))
    structures = ["replace_withdraw", "replace_withdraw", "replace_only", "withdraw_only"]
    randomized = bool(config.get("randomize_layout"))
    if randomized:
        # Every new behaviour is opt-in. The frozen pilot config leaves
        # randomize_layout off, so data/pilot_v0 regenerates byte-for-byte.
        dialogues, calibration_ids = _generate_randomized(config)
    else:
        rng = random.Random(config["seed"])
        scenarios = list(SCENARIOS) * (config["dialogue_count"] // len(SCENARIOS))
        rng.shuffle(scenarios)
        dialogues = []
        for index in range(1, config["dialogue_count"] + 1):
            if index in empty_ids:
                dialogue = _build_empty(index, scenarios[index - 1])
            else:
                kind = "distractor" if index in distractor_ids else "standard"
                dialogue = _build_standard(index, scenarios[index - 1], kind, structures[(index - 1) % len(structures)], rng)
            if config.get("add_unrelated_probes") and dialogue["candidate_truth"]:
                dialogue["probes"].append(
                    _unrelated_probe(config.get("unrelated_probe_compliant_semantics", "task_correct"))
                )
            if not config["min_turns"] <= len(dialogue["turns"]) <= config["max_turns"]:
                raise ValueError(f"{dialogue['dialogue_id']}: turn count {len(dialogue['turns'])} outside configured bounds")
            dialogues.append(dialogue)
        calibration_ids = [dialogues[i - 1]["dialogue_id"] for i in (1, 6, 16, distractor_start, config["dialogue_count"])]

    for dialogue in dialogues:
        if not config["min_turns"] <= len(dialogue["turns"]) <= config["max_turns"]:
            raise ValueError(f"{dialogue['dialogue_id']}: turn count {len(dialogue['turns'])} outside configured bounds")

    report = coverage_report(dialogues, config)
    if not report["accepted_for_annotation_start"]:
        raise ValueError(f"Generator coverage checks failed: {report['checks']}")

    tasks = [{"dialogue_id": d["dialogue_id"], "status": d["status"], "scenario": d["scenario"], "turns": d["turns"]} for d in dialogues]
    worksheets = [{"dialogue_id": d["dialogue_id"], "status": "awaiting_independent_annotation", "turns": d["turns"], "annotations": []} for d in dialogues]

    write_jsonl(output / "tasks.jsonl", tasks)
    write_jsonl(output / "candidate_truth.jsonl", dialogues)
    write_jsonl(output / "annotation_worksheet.jsonl", worksheets)
    write_json(output / "calibration_ids.json", {"count": len(calibration_ids), "dialogue_ids": calibration_ids})
    write_json(output / "coverage.json", report)
    files = ["tasks.jsonl", "candidate_truth.jsonl", "annotation_worksheet.jsonl", "calibration_ids.json", "coverage.json"]
    manifest = {
        "dataset_version": config["dataset_version"],
        "status": DATASET_STATUS,
        "generation_method": config["generation_method"],
        "seed": config["seed"],
        "generator_config_sha256": sha256_file(config_path),
        "gold_version": None,
        "human_annotation_complete": False,
        "files": {name: sha256_file(output / name) for name in files},
    }
    write_json(output / "manifest.json", manifest)
    if config["dataset_version"] != "pilot-v0-candidate":
        # Any *new* data set version gets a dataset.json so that
        # `validate-submission --dir <dir>` can check it like any other
        # contributed package. The frozen pilot set is left untouched: it is
        # a version lock, not an extension, and a byte-identical
        # regeneration must not add files to it either.
        from .manifest import build_dataset_manifest
        build_dataset_manifest(
            output,
            benchmark_config_path=root / "configs/benchmark.json",
            contributor=contributor,
            split=split,
        )
    return report
