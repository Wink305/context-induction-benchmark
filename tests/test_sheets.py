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
from context_benchmark.io import write_jsonl


def wrap(body: str) -> str:
    return f"# sheet\n\n{ANSWER_START}\n{body}\n{ANSWER_END}\n"


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
        # empty dialogue gets the leave-it-blank hint
        self.assertIn("留空", text)

    def test_pipe_in_text_does_not_break_table(self):
        dialogue = {
            "dialogue_id": "p0-998",
            "turns": [{"turn": 1, "speaker": "user", "text": "A | B"}],
            "candidate_truth": [],
            "distractors": [],
        }
        text = render_sheet(dialogue, index=1, total=1, calibration=False, annotator="A")
        self.assertIn("A ｜ B", text)


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
        records = finalize_records(records, "p0-001")
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
            record = parsed["annotations"][0]["constraints"][0]
            self.assertEqual(record["type"], "standing")
            self.assertTrue(record["id"].startswith("p0-001-a"))


if __name__ == "__main__":
    unittest.main()
