from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from context_benchmark.sheets import (
    ANSWER_END,
    ANSWER_START,
    finalize_records,
    make_sheets,
    parse_answer_block,
    parse_sheets,
    render_sheet,
)
from context_benchmark.io import read_json, read_jsonl, write_jsonl


def wrap(body: str) -> str:
    return f"# sheet\n\n{ANSWER_START}\n{body}\n{ANSWER_END}\n"


def hint_block(text: str) -> str:
    """The per-segment hint section of a rendered sheet."""
    return text.split("## 二、本段提示", 1)[1].split("## 三、答案区", 1)[0]


class SheetRenderTest(unittest.TestCase):
    def test_render_contains_markers_and_turns(self):
        dialogue = {
            "dialogue_id": "p0-999",
            "turns": [
                {"turn": 1, "speaker": "user", "text": "以后都只用邮件"},
                {"turn": 2, "speaker": "agent", "text": "好的"},
            ],
            "candidate_truth": [],
            "distractors": [],
        }
        text = render_sheet(dialogue, index=1, total=5, calibration=True, annotator="A")
        self.assertIn(ANSWER_START, text)
        self.assertIn(ANSWER_END, text)
        self.assertIn("以后都只用邮件", text)
        self.assertIn("客服", text)
        self.assertIn("校准集", text)
        # neutral guidance must remain, but never reveal the answer
        self.assertIn("留空", text)
        self.assertNotIn("正确答案是留空", text)

    def test_example_does_not_leak_pilot_gold(self):
        """The format example must be fictional, never a real gold unit."""
        text = render_sheet(
            read_jsonl("data/pilot_v0/candidate_truth.jsonl")[0],
            index=1, total=5, calibration=True, annotator="A",
        )
        # The fictional framing appears before the example block; gold values
        # must not appear either in the framing or in the example rows.
        leaked_region = text.split("示例：", 1)[0].split("## 三、答案区", 1)[-1] + text.split("示例：", 1)[-1].split("在下面两行注释", 1)[0]
        # A leak is a *gold unit* reproduced in the example.  Its text is what
        # identifies it: a bare turn number carries no information, because
        # the example is a fictional dialogue and says nothing about which
        # segment of the real task holds a constraint.  Checking every gold
        # text is strictly stronger than the two hard-coded strings this
        # test used to look for.
        gold_texts = {
            unit["text"]
            for row in read_jsonl("data/pilot_v0/candidate_truth.jsonl")
            for unit in row.get("candidate_truth", [])
            if unit.get("text")
        }
        self.assertGreaterEqual(len(gold_texts), 20, "the pilot gold corpus is unexpectedly small")
        for value in gold_texts:
            self.assertNotIn(value, leaked_region, f"gold text {value!r} leaked into the sheet example")
        self.assertIn("虚构对话", leaked_region)

    def test_worksheet_without_truth_never_produces_structural_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worksheet = root / "worksheet.jsonl"
            record = {
                "dialogue_id": "p0-777",
                "status": "awaiting_independent_annotation",
                "turns": [{"turn": 1, "speaker": "user", "text": "以后都只用邮件"}],
                "annotations": [],
            }
            write_jsonl(worksheet, [record])
            out = root / "sheets"
            manifest = make_sheets(worksheet, out, calibration_ids=["p0-777"])
            text = (out / "p0-777.md").read_text(encoding="utf-8")
            self.assertFalse(manifest["hints_from_truth"])
            # Every structural hint the truth path can add starts with
            # 「本段有」/「本段含」.  Testing that prefix catches all eight of
            # them at once — not just the one the old test looked for — and it
            # does not trip on the field documentation, which is identical
            # for every segment and therefore leaks nothing.
            for banned in ("本段有", "本段含", "正确答案是留空", "大概率一条约束都没有"):
                self.assertNotIn(banned, text)
            self.assertNotIn("显式撤回", hint_block(text))

    def test_truth_opt_in_adds_hints_without_revealing_empty_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worksheet = root / "worksheet.jsonl"
            truth = root / "truth.jsonl"
            record = {
                "dialogue_id": "p0-778",
                "status": "awaiting_independent_annotation",
                "turns": [{"turn": 1, "speaker": "user", "text": "那条不算"}],
                "annotations": [],
            }
            write_jsonl(worksheet, [record])
            write_jsonl(truth, [{
                **record,
                "candidate_truth": [{
                    "id": "c001", "type": "superseded", "text": "旧规则", "evidence_turns": [1],
                    "valid_from": 1, "valid_to": 1, "scope": "global", "superseded_by": None,
                    "category": "process", "rule_tags": ["R3"],
                }],
                "distractors": [],
            }])
            out = root / "sheets"
            manifest = make_sheets(worksheet, out, calibration_ids=["p0-778"], truth_path=truth)
            text = (out / "p0-778.md").read_text(encoding="utf-8")
            self.assertTrue(manifest["hints_from_truth"])
            self.assertIn("显式撤回", text)
            self.assertNotIn("正确答案是留空", text)

    def test_pipe_in_text_does_not_break_table(self):
        dialogue = {
            "dialogue_id": "p0-998",
            "turns": [{"turn": 1, "speaker": "user", "text": "A | B"}],
            "candidate_truth": [],
            "distractors": [],
        }
        text = render_sheet(dialogue, index=1, total=1, calibration=False, annotator="A")
        self.assertIn("A ｜ B", text)


class AnswerFreeRenderTest(unittest.TestCase):
    """The default render must not leak which segments carry constraints."""

    EMPTY_DIALOGUE = {
        "dialogue_id": "p0-empty",
        "turns": [{"turn": 1, "speaker": "user", "text": "我住在杭州。"}],
    }
    FILLED_DIALOGUE = {
        "dialogue_id": "p0-filled",
        "turns": [{"turn": 1, "speaker": "user", "text": "以后都只用邮件联系我，不要打电话。"}],
    }

    def test_default_render_is_uniform_across_segments(self):
        empty = render_sheet(self.EMPTY_DIALOGUE, index=1, total=2, calibration=True, annotator="A")
        filled = render_sheet(self.FILLED_DIALOGUE, index=2, total=2, calibration=True, annotator="A")
        self.assertEqual(hint_block(empty), hint_block(filled))

    def test_default_render_never_mentions_emptiness(self):
        for dialogue in (self.EMPTY_DIALOGUE, self.FILLED_DIALOGUE):
            text = render_sheet(dialogue, index=1, total=2, calibration=True, annotator="A")
            for banned in ("没有用户约束", "正确答案是留空", "大概率一条约束都没有", "答案区留空"):
                self.assertNotIn(banned, text)

    def test_worksheet_rows_carrying_a_truth_do_not_leak_it(self):
        """A worksheet row that still holds labels must not become a hint."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worksheet = root / "worksheet.jsonl"
            record = dict(self.FILLED_DIALOGUE)
            record["status"] = "awaiting_independent_annotation"
            record["annotations"] = []
            record["candidate_truth"] = [{"id": "c001", "type": "standing", "text": "旧规则", "rule_tags": ["R3"]}]
            write_jsonl(worksheet, [record])
            out = root / "sheets"
            make_sheets(worksheet, out, calibration_ids=["p0-filled"])
            text = (out / "p0-filled.md").read_text(encoding="utf-8")
            for banned in ("本段有", "本段含", "正确答案是留空"):
                self.assertNotIn(banned, text)
            self.assertNotIn("显式撤回", hint_block(text))
            self.assertEqual(hint_block(text), hint_block(
                render_sheet(self.FILLED_DIALOGUE, index=1, total=1, calibration=True, annotator="A")
            ))

    @unittest.skipUnless(
        Path("data/annotation/calibration_A").is_dir(),
        "calibration sheets are local working artifacts and are not published",
    )
    def test_committed_calibration_sheets_stay_answer_free(self):
        truth = {row["dialogue_id"]: row for row in read_jsonl("data/pilot_v0/candidate_truth.jsonl")}
        blocks = set()
        paths = sorted(Path("data/annotation/calibration_A").glob("*.md"))
        self.assertTrue(paths, "calibration sheets are missing")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            answer = text.split(ANSWER_START, 1)[1].split(ANSWER_END, 1)[0]
            self.assertEqual(answer.strip(), "", f"{path.name} already contains answers")
            tail = text.split("## 二、本段提示", 1)[1]
            for unit in truth.get(path.stem, {}).get("candidate_truth", []):
                value = unit.get("text")
                if value:
                    self.assertNotIn(value, tail, f"{path.name} leaks gold text {value!r}")
            for banned in ("没有用户约束", "正确答案是留空", "大概率一条约束都没有"):
                self.assertNotIn(banned, text, f"{path.name} hints at the answer")
            blocks.add(hint_block(text))
        self.assertEqual(len(blocks), 1, "every calibration sheet must carry the same neutral hint")

    @unittest.skipUnless(
        Path("data/annotation/precheck_ai_annotations.jsonl").is_file()
        and Path("data/annotation/precheck_ai_report.json").is_file(),
        "the AI pre-check artefacts are local working artifacts and are not published",
    )
    def test_precheck_ai_sheets_are_not_human_annotation_material(self):
        """The AI pre-check sheets must never be shipped as annotator sheets."""
        report = read_json("data/annotation/precheck_ai_report.json")
        self.assertEqual(report["annotator_type"], "ai_precheck")
        self.assertEqual(report["evidence_grade"], "ai_precheck_only")
        self.assertFalse(report["human_annotation_complete"])
        rows = read_jsonl("data/annotation/precheck_ai_annotations.jsonl")
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["provenance"], "ai_precheck")
            self.assertFalse(row["human_annotation_complete"])
            for unit in row.get("constraints", []):
                self.assertNotEqual(unit.get("note"), "human annotation")
                self.assertIn("AI", unit.get("note", ""))


class ParseAnswerTest(unittest.TestCase):
    def test_parses_plain_rows(self):
        body = "\n".join(
            [
                "3 | S | global | C | 仅用邮件联系 |",
                "19 | O | session | P | 仅本单不升级 |",
            ]
        )
        records, problems = parse_answer_block(wrap(body))
        self.assertEqual(problems, [])
        self.assertEqual([r["type"] for r in records], ["standing", "one_off"])
        self.assertEqual(records[0]["evidence_turns"], [3])
        self.assertEqual(records[1]["scope"], "session")
        self.assertEqual(records[0]["valid_from"], 3)

    def test_accepts_chinese_aliases_and_fullwidth_pipe(self):
        body = "7，9，11 ｜ 隐含 ｜ 全局 ｜ 沟通 ｜ 回复只要两行重点 ｜"
        records, problems = parse_answer_block(wrap(body))
        self.assertEqual(problems, [])
        self.assertEqual(records[0]["type"], "implicit")
        self.assertEqual(records[0]["scope"], "global")
        self.assertEqual(records[0]["category"], "communication")
        self.assertEqual(records[0]["evidence_turns"], [7, 9, 11])

    def test_turn_range_is_expanded(self):
        records, _ = parse_answer_block(wrap("7-9 | I | global | C | 只要重点 |"))
        self.assertEqual(records[0]["evidence_turns"], [7, 8, 9])

    def test_superseded_requires_invalidation(self):
        records, problems = parse_answer_block(wrap("13 | T | global | C | 电话提醒 |"))
        self.assertEqual(records, [])
        self.assertTrue(any("T 类必须填写失效说明" in p for p in problems))

    def test_non_superseded_rejects_invalidation(self):
        records, problems = parse_answer_block(wrap("3 | S | global | C | 仅用邮件 | 15"))
        self.assertEqual(records, [])
        self.assertTrue(any("只有 T 类才填失效说明" in p for p in problems))

    def test_superseded_links_to_replacing_standing(self):
        body = "\n".join(
            [
                "27 | T | global | P | 退款原路退回 | 被 29 轮替代",
                "29 | S | global | P | 退款优先退账户余额 |",
            ]
        )
        records, problems = parse_answer_block(wrap(body))
        self.assertEqual(problems, [])
        records = finalize_records(records)
        self.assertEqual(records[0]["valid_to"], 28)
        self.assertEqual(records[0]["superseded_by"], records[1]["id"])

    def test_withdrawal_note_sets_valid_to(self):
        records, problems = parse_answer_block(wrap("13 | T | global | C | 电话提醒 | 15"))
        self.assertEqual(problems, [])
        self.assertEqual(records[0]["valid_to"], 14)

    def test_question_is_extracted_not_treated_as_note(self):
        body = "19 | O | session | P | 仅本单不升级 | #疑问：这算不算 R1"
        records, problems = parse_answer_block(wrap(body))
        self.assertEqual(problems, [])
        self.assertEqual(records[0]["question"], "这算不算 R1")
        self.assertIsNone(records[0]["valid_to"])

    def test_empty_block_is_valid_but_recorded_empty(self):
        records, problems = parse_answer_block(wrap("   \n\n"))
        self.assertEqual(records, [])
        self.assertEqual(problems, [])

    def test_missing_markers_reported(self):
        records, problems = parse_answer_block("no markers here")
        self.assertEqual(records, [])
        self.assertTrue(problems)

    def test_bad_type_reports_line_number(self):
        _, problems = parse_answer_block(wrap("3 | X | global | C | 内容 |"))
        self.assertTrue(any("类型无法识别" in p for p in problems))

    def test_comment_lines_skipped(self):
        body = "# 这是注释\n3 | S | global | C | 仅用邮件 |"
        records, problems = parse_answer_block(wrap(body))
        self.assertEqual(problems, [])
        self.assertEqual(len(records), 1)


class RoundTripTest(unittest.TestCase):
    def test_make_then_parse_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worksheet = root / "worksheet.jsonl"
            write_jsonl(
                worksheet,
                [
                    {
                        "dialogue_id": "p0-001",
                        "status": "awaiting_independent_annotation",
                        "turns": [{"turn": 1, "speaker": "user", "text": "以后只用邮件"}],
                        "annotations": [],
                        "candidate_truth": [],
                        "distractors": [],
                    }
                ],
            )
            out = root / "sheets"
            manifest = make_sheets(worksheet, out, calibration_ids=["p0-001"], annotator="A")
            self.assertEqual(manifest["sheets"], 1)
            sheet = out / "p0-001.md"
            text = sheet.read_text(encoding="utf-8")
            self.assertIn(ANSWER_START, text)
            sheet.write_text(
                text.replace(ANSWER_START, ANSWER_START + "\n1 | S | global | C | 仅用邮件 |"),
                encoding="utf-8",
            )
            parsed = parse_sheets(out, annotator="A")
            self.assertEqual(parsed["sheet_n"], 1)
            self.assertEqual(parsed["annotated_n"], 1)
            self.assertEqual(parsed["problems"], [])
            # The parsed row carries the units under ``annotations`` — the key
            # the worksheet, the agreement gate and the adjudication gate all
            # read.  It used to be emitted as ``constraints`` only, so the
            # whole chain agreed on zero units.
            row = parsed["annotations"][0]
            self.assertIn("annotations", row)
            record = row["annotations"][0]
            self.assertEqual(record["type"], "standing")
            # The id must satisfy the model-output contract (c<number>), so
            # the annotation can be frozen and scored like a prediction.
            self.assertRegex(record["id"], r"^c\d+$")


if __name__ == "__main__":
    unittest.main()
