"""Deterministic P0 dialogue generator.

The output is deliberately labelled *candidate truth*. It is an annotation aid,
not human-adjudicated gold and must never be reported as gold v1.0.
"""

from __future__ import annotations

import random
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
            "query": "新工单需要发送进度通知，应如何联系用户？",
            "behaviors": [
                {"id": "b1", "text": "通过邮件发送，不拨打电话。", "is_compliant": True},
                {"id": "b2", "text": "先打电话，再发短信。", "is_compliant": False},
            ],
        },
        {
            "probe_id": "p002",
            "target_constraint_id": second_target,
            "query": "处理一个新的适用任务时，应遵循哪项行为？",
            "behaviors": second_behaviors,
        },
    ]


def _build_standard(index: int, scenario: str, kind: str, structure: str) -> dict[str, Any]:
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

    u, a = b.pair("以后都只用邮件联系我，不要打电话。", "好的，以后都走邮件，不电话联系。")
    first_mail_turn = u
    non_units.append({
        "id": "n-r6", "kind": "agent_paraphrase", "text": "以后都走邮件",
        "evidence_turns": [a], "rule_tags": ["R6"], "note": "复述确认，不产生新单元",
    })

    u, _ = b.pair("这个项目里都发邮件沟通，别在这个项目里打电话。", "明白，仅这个项目使用邮件。")
    truth.append(_constraint("c002", "one_off", "本项目内只通过邮件沟通", [u], u, "task", "communication", rules=["R1"], note="规则化语气但绑定项目"))

    implicit_com: list[int] = []
    for text in ("这段又写长了，我只看两行重点。", "这次也别展开，我只看两行重点。", "还是太细了，我只看两行重点。"):
        u, _ = b.pair(text, "我先给您两行重点。")
        implicit_com.append(u)
    truth.append(_constraint("c003", "implicit", "回复应压缩为两行重点", implicit_com, implicit_com[0], "global", "communication", note="三次独立交互模式；候选 I，待人工确认"))

    u, _ = b.pair("以后可以用电话提醒我工单进度。", "已记录电话提醒。")
    if structure in {"replace_withdraw", "withdraw_only"}:
        w, _ = b.pair("之前说可以电话提醒，那条现在取消。", "已取消电话提醒。")
        truth.append(_constraint("c004", "superseded", "通过电话提醒工单进度", [u, w], u, "global", "communication", valid_to=w - 1, rules=["R3"], note=f"第 {w} 轮显式撤回"))
    else:
        truth.append(_constraint("c004", "standing", "通过电话提醒工单进度", [u], u, "global", "communication"))

    if kind == "distractor":
        u, a = b.pair(
            "以后超过 48 小时没处理就升级人工主管。",
            "已记录 48 小时升级规则。另外系统默认使用短信提醒所有工单。",
        )
    else:
        u, a = b.pair("以后超过 48 小时没处理就升级人工主管。", "已记录 48 小时升级规则。")
    truth.append(_constraint("c005", "standing", "超过48小时未处理时升级人工主管", [u], u, "global", "process"))
    if kind == "distractor":
        distractors.append({
            "id": "d001", "text": "默认使用短信提醒所有工单", "evidence_turns": [a],
            "category": "communication", "expected": "exclude", "note": "客服单方声明，用户未确认，不得抽为用户约束",
        })

    u, _ = b.pair("仅这张工单先不要升级，等我明天确认。", "本单暂不升级。")
    truth.append(_constraint("c006", "one_off", "本工单暂不升级并等待次日确认", [u], u, "task", "process"))

    implicit_process: list[int] = []
    for text in ("先别替我提交，我自己再试一下。", "这次也先别提交，我再核对一下。", "还是先别提交，我想自己再试一次。"):
        u, _ = b.pair(text, "好的，本次暂不提交。")
        implicit_process.append(u)
    truth.append(_constraint("c007", "implicit", "提交前应等待用户自行检查", implicit_process, implicit_process[0], "global", "process", note="三次独立交互模式；候选 I，待人工确认"))

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

    if kind == "distractor":
        u, a = b.pair(
            f"{spec['stable']} 另外，以后仍只用邮件联系我，不要打电话。",
            "已记录，并继续遵循邮件联系。另外按平台惯例默认向所有用户推荐最贵的方案。",
        )
    else:
        u, a = b.pair(f"{spec['stable']} 另外，以后仍只用邮件联系我，不要打电话。", "已记录，并继续遵循邮件联系。")
    truth.append(_constraint("c012", "standing", spec["stable"], [u], u, "global", "domain" if scenario in {"travel", "ecommerce"} else "process", note="弱化偏好类候选，人工判断强度"))
    truth.insert(0, _constraint("c001", "standing", "只通过邮件联系且不打电话", [first_mail_turn, u], first_mail_turn, "global", "communication", rules=["R8"], note="重复陈述合并"))

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


def coverage_report(dialogues: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    types = Counter()
    categories = Counter()
    rules: dict[str, set[str]] = defaultdict(set)
    structures = Counter()
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
    unit_count = sum(types.values())
    empty_count = sum(d["dialogue_kind"] == "empty" for d in dialogues)
    distractor_dialogues = sum(bool(d["distractors"]) for d in dialogues)
    rule_counts = {rule: len(rules.get(rule, set())) for rule in RULES}
    checks = {
        "dialogue_count_40": len(dialogues) == config["dialogue_count"],
        "turn_bounds": bool(turn_counts) and min(turn_counts) >= config["min_turns"] and max(turn_counts) <= config["max_turns"],
        "candidate_units_400_600": 400 <= unit_count <= 600,
        "all_four_types": set(types) == {"standing", "one_off", "implicit", "superseded"},
        "rules_min_coverage": all(value >= config["min_rule_dialogue_coverage"] for value in rule_counts.values()),
        "empty_min_coverage": empty_count >= config["min_empty_dialogues"],
        "distractor_min_coverage": distractor_dialogues >= config["min_distractor_dialogues"],
        "all_status_unannotated": all(d["status"] == DATASET_STATUS for d in dialogues),
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
        "checks": checks,
        "accepted_for_annotation_start": all(checks.values()),
        "not_a_gold_acceptance": True,
    }


def generate_dataset(config_path: str | Path) -> dict[str, Any]:
    config = read_json(config_path)
    root = Path(config_path).resolve().parents[1]
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config["seed"])
    scenarios = list(SCENARIOS) * (config["dialogue_count"] // len(SCENARIOS))
    rng.shuffle(scenarios)

    empty_ids = set(range(config["dialogue_count"] - config["empty_count"] + 1, config["dialogue_count"] + 1))
    distractor_start = config["dialogue_count"] - config["empty_count"] - config["distractor_count"] + 1
    distractor_ids = set(range(distractor_start, distractor_start + config["distractor_count"]))
    structures = ["replace_withdraw", "replace_withdraw", "replace_only", "withdraw_only"]
    dialogues = []
    for index in range(1, config["dialogue_count"] + 1):
        if index in empty_ids:
            dialogue = _build_empty(index, scenarios[index - 1])
        else:
            kind = "distractor" if index in distractor_ids else "standard"
            dialogue = _build_standard(index, scenarios[index - 1], kind, structures[(index - 1) % len(structures)])
        if not config["min_turns"] <= len(dialogue["turns"]) <= config["max_turns"]:
            raise ValueError(f"{dialogue['dialogue_id']}: turn count {len(dialogue['turns'])} outside configured bounds")
        dialogues.append(dialogue)

    report = coverage_report(dialogues, config)
    if not report["accepted_for_annotation_start"]:
        raise ValueError(f"Generator coverage checks failed: {report['checks']}")

    tasks = [{"dialogue_id": d["dialogue_id"], "status": d["status"], "scenario": d["scenario"], "turns": d["turns"]} for d in dialogues]
    worksheets = [{"dialogue_id": d["dialogue_id"], "status": "awaiting_independent_annotation", "turns": d["turns"], "annotations": []} for d in dialogues]
    calibration_ids = [dialogues[i - 1]["dialogue_id"] for i in (1, 6, 16, distractor_start, config["dialogue_count"])]

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
    return report
