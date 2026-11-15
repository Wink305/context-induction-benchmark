"""Render human-fillable annotation sheets and parse them back into gold records.

The sheet is a Markdown file per dialogue.  The annotator writes one line per
constraint unit between two HTML comment markers.  This module only converts
between that plain-text form and the JSON records used by the scorer; it never
guesses labels on the annotator's behalf.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_json, write_jsonl

ANSWER_START = "<!-- ANSWER_START -->"
ANSWER_END = "<!-- ANSWER_END -->"

TYPE_ALIASES = {
    "s": "standing",
    "standing": "standing",
    "稳定": "standing",
    "稳定约束": "standing",
    "o": "one_off",
    "one_off": "one_off",
    "one-off": "one_off",
    "一次性": "one_off",
    "一次性请求": "one_off",
    "i": "implicit",
    "implicit": "implicit",
    "隐含": "implicit",
    "隐含约束": "implicit",
    "t": "superseded",
    "superseded": "superseded",
    "失效": "superseded",
    "已失效": "superseded",
    "已失效约束": "superseded",
}

SCOPE_ALIASES = {
    "global": "global",
    "全局": "global",
    "session": "session",
    "会话": "session",
    "本次": "session",
    "task": "task",
    "任务": "task",
    "项目": "task",
}

CATEGORY_ALIASES = {
    "c": "communication",
    "communication": "communication",
    "沟通": "communication",
    "联系方式": "communication",
    "p": "process",
    "process": "process",
    "流程": "process",
    "d": "domain",
    "domain": "domain",
    "领域": "domain",
    "业务": "domain",
}


def _norm_token(text: str) -> str:
    return re.sub(r"[\s`'\"（）()\[\]]", "", text).strip().lower()


def _split_row(line: str) -> list[str]:
    normalized = line.replace("｜", "|").replace("，", ",")
    return [part.strip() for part in normalized.split("|")]


def _parse_turns(text: str) -> list[int]:
    numbers: list[int] = []
    for chunk in re.split(r"[,，、/\s]+", text.strip()):
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*[-–~]\s*(\d+))?", chunk)
        if not match:
            raise ValueError(f"无法识别的轮次：{chunk!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            raise ValueError(f"轮次区间颠倒：{chunk!r}")
        numbers.extend(range(start, end + 1))
    return sorted(set(numbers))


def _parse_end(text: str) -> tuple[int | None, str | None]:
    """Return (valid_to, superseded_by_raw) from a free-form invalidation note."""
    if not text:
        return None, None
    raw = text.strip()
    replaced = re.search(r"(?:被|由)\s*(\d+)\s*轮(?:替代|取代|覆盖)", raw)
    if replaced:
        return int(replaced.group(1)) - 1, int(replaced.group(1))
    withdrawn = re.search(r"(\d+)", raw)
    if withdrawn:
        turn = int(withdrawn.group(1))
        return turn - 1, None
    raise ValueError(f"失效说明里找不到轮次：{raw!r}")


def parse_answer_block(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse the answer block. Returns (records, problems)."""
    if ANSWER_START not in text or ANSWER_END not in text:
        return [], ["缺少 ANSWER_START / ANSWER_END 标记，文件可能被改动过"]
    body = text.split(ANSWER_START, 1)[1].split(ANSWER_END, 1)[0]
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for lineno, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        question = None
        if "#疑问" in stripped:
            stripped, _, question_text = stripped.partition("#疑问")
            question = question_text.lstrip("：: ").strip() or "(未写明)"
            stripped = stripped.rstrip(" |")
        try:
            parts = _split_row(stripped)
            if len(parts) < 5:
                raise ValueError(f"列数不足（需要至少 5 列，实际 {len(parts)} 列）")
            turns = _parse_turns(parts[0])
            if not turns:
                raise ValueError("依据轮次为空")
            type_key = _norm_token(parts[1])
            if type_key not in TYPE_ALIASES:
                raise ValueError(f"类型无法识别：{parts[1]!r}（应为 S/O/I/T）")
            kind = TYPE_ALIASES[type_key]
            scope_key = _norm_token(parts[2])
            if scope_key not in SCOPE_ALIASES:
                raise ValueError(f"作用域无法识别：{parts[2]!r}（应为 global/session/task）")
            scope = SCOPE_ALIASES[scope_key]
            category_key = _norm_token(parts[3])
            if category_key not in CATEGORY_ALIASES:
                raise ValueError(f"类别无法识别：{parts[3]!r}（应为 C/P/D）")
            category = CATEGORY_ALIASES[category_key]
            content = parts[4].strip()
            if not content:
                raise ValueError("内容摘要为空")
            note = parts[5].strip() if len(parts) > 5 else ""
            valid_to, replaced_turn = _parse_end(note)
            if kind == "superseded" and valid_to is None:
                raise ValueError("T 类必须填写失效说明（撤回或替换的轮次）")
            if kind != "superseded" and note:
                raise ValueError(f"只有 T 类才填失效说明，当前类型是 {kind}")
            records.append(
                {
                    "type": kind,
                    "scope": scope,
                    "category": category,
                    "content": content,
                    "evidence_turns": turns,
                    "valid_from": min(turns),
                    "valid_to": valid_to,
                    "superseded_by": None,
                    "_replaced_turn": replaced_turn,
                    "question": question,
                }
            )
        except ValueError as exc:
            problems.append(f"第 {lineno} 行：{exc} ｜ 原文：{stripped}")
    _link_superseded(records, problems)
    return records, problems


def _link_superseded(records: list[dict[str, Any]], problems: list[str]) -> None:
    """Point each superseded record at the standing record born at the replacing turn."""
    for record in records:
        if record["type"] != "superseded":
            continue
        target = record.pop("_replaced_turn", None)
        if target is None:
            continue
        candidates = [
            position
            for position, other in enumerate(records)
            if other is not record
            and other["type"] in {"standing", "implicit"}
            and min(other["evidence_turns"]) == target
        ]
        if candidates:
            record["_superseded_index"] = candidates[0]
        else:
            problems.append(
                f"T 类（轮 {record['evidence_turns']}）声明被轮 {target} 替代，但该轮没有对应的 S/I 单元"
            )
    for record in records:
        record.pop("_replaced_turn", None)


def finalize_records(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    for index, record in enumerate(records, start=1):
        record["id"] = f"{prefix}-a{index:03d}"
        record.setdefault("note", "human annotation")
        record.setdefault("rule_tags", [])
    for record in records:
        position = record.pop("_superseded_index", None)
        if position is not None:
            record["superseded_by"] = records[position]["id"]
    return records


def _focus_hints(dialogue: dict[str, Any]) -> list[str]:
    truth = dialogue.get("candidate_truth", [])
    hints: list[str] = []
    if not truth:
        hints.append(
            "**本段大概率一条约束都没有。正确答案是留空**——不要为了填而填。"
        )
    if dialogue.get("distractors"):
        hints.append(
            "本段含**客服侧信息或与行为无关的内容**，它们不是用户约束，注意别标进来。"
        )
    tags = {tag for unit in truth for tag in unit.get("rule_tags", [])}
    if "R3" in tags:
        hints.append("本段有**显式撤回**（用户说「那条不算」一类），被撤回的条目要标 T。")
    if "R7" in tags:
        hints.append("本段有**规则替换**（新规则覆盖旧规则），旧条目标 T 并注明被哪一轮替代。")
    if "R1" in tags:
        hints.append("本段有**语气像规则但只管某个项目/对象**的句子，按 O 处理。")
    if "R5" in tags:
        hints.append("本段有**事实陈述**（如过敏、住址），只有当它蕴含行为要求时才标。")
    if "R8" in tags:
        hints.append("本段同一要求可能**出现两次**（一次规则化、一次单独提），合并成一条而不是两条。")
    return hints


def render_sheet(
    dialogue: dict[str, Any],
    index: int,
    total: int,
    calibration: bool,
    annotator: str,
) -> str:
    rows = []
    for turn in dialogue["turns"]:
        speaker = "用户" if turn["speaker"] == "user" else "客服"
        text = str(turn["text"]).replace("|", "｜")
        rows.append(f"| {turn['turn']} | {speaker} | {text} |")
    hints = _focus_hints(dialogue)
    hint_block = "\n".join(f"- {item}" for item in hints) if hints else "- 常规段落，按手册逐条判断即可。"
    tag = "校准集" if calibration else "正式标注"
    return f"""# 标注工作表 · {dialogue['dialogue_id']}

> {tag}（第 {index}/{total} 段）｜共 {len(dialogue['turns'])} 轮｜标注者：{annotator}
> 先读《P0_标注手册_v1.md》，再填本节末尾的答案区。

## 一、对话全文

| 轮 | 说话人 | 内容 |
|---|---|---|
{chr(10).join(rows)}

## 二、本段提示

{hint_block}

## 三、答案区

格式：`依据轮次 | 类型 | 作用域 | 类别 | 内容摘要 | 失效说明`

- 依据轮次：`3` 或 `7,9,11`（隐含约束需要多处依据）
- 类型：`S` 稳定 / `O` 一次性 / `I` 隐含 / `T` 已失效
- 作用域：`global` / `session` / `task`
- 类别：`C` 沟通 / `P` 流程 / `D` 业务
- 内容摘要：一句话，不超过 20 字
- 失效说明：**只有 T 类填**，写 `15`（15 轮撤回）或 `被 29 轮替代`；其他留空

示例：
```
3 | S | global | C | 仅用邮件联系 | 
13 | T | global | C | 电话提醒工单 | 15
7,9,11 | I | global | C | 回复只要两行重点 |
```

在下面两行注释之间逐行填写，不要删除注释标记：

{ANSWER_START}

{ANSWER_END}

## 四、交卷前自查

- [ ] 每一条都写了依据轮次，且轮次真的在对话里
- [ ] T 类都写了失效说明，非 T 类没写
- [ ] 没有把客服的话、客服的承诺当成用户约束
- [ ] 只出现两次的重复说法，没有标成 I
- [ ] 空段就留空，没有硬凑
"""


def make_sheets(
    worksheet_path: str | Path,
    out_dir: str | Path,
    ids: Iterable[str] | None = None,
    annotator: str = "A",
    calibration_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    dialogues = read_jsonl(worksheet_path)
    wanted = set(ids) if ids else None
    cal = set(calibration_ids or [])
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    cal_total = len(cal)
    cal_index = 0
    for dialogue in dialogues:
        if wanted is not None and dialogue["dialogue_id"] not in wanted:
            continue
        is_cal = dialogue["dialogue_id"] in cal
        if is_cal:
            cal_index += 1
        path = target_dir / f"{dialogue['dialogue_id']}.md"
        path.write_text(
            render_sheet(
                dialogue,
                index=cal_index if is_cal else 0,
                total=cal_total if is_cal else 0,
                calibration=is_cal,
                annotator=annotator,
            ),
            encoding="utf-8",
        )
        created.append(str(path))
    manifest = {
        "annotator": annotator,
        "sheets": len(created),
        "calibration": sorted(cal & {d["dialogue_id"] for d in dialogues}),
        "out_dir": str(target_dir),
    }
    write_json(target_dir / "_manifest.json", manifest)
    return manifest


def parse_sheets(sheet_dir: str | Path, annotator: str = "A") -> dict[str, Any]:
    directory = Path(sheet_dir)
    paths = sorted(p for p in directory.glob("*.md"))
    annotations: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    empty: list[str] = []
    questions: list[dict[str, Any]] = []
    for path in paths:
        dialogue_id = path.stem
        text = path.read_text(encoding="utf-8")
        records, issues = parse_answer_block(text)
        if issues:
            problems.append({"dialogue_id": dialogue_id, "issues": issues})
        if not records:
            empty.append(dialogue_id)
        for record in records:
            question = record.pop("question", None)
            if question:
                questions.append(
                    {
                        "dialogue_id": dialogue_id,
                        "evidence_turns": record["evidence_turns"],
                        "content": record["content"],
                        "question": question,
                    }
                )
        records = finalize_records(records, dialogue_id)
        annotations.append(
            {
                "dialogue_id": dialogue_id,
                "annotator": annotator,
                "status": "annotated",
                "constraints": records,
            }
        )
    return {
        "annotator": annotator,
        "sheet_n": len(paths),
        "annotated_n": len([row for row in annotations if row["constraints"]]),
        "empty_sheets": empty,
        "problems": problems,
        "questions": questions,
        "annotations": annotations,
    }
