from __future__ import annotations

import unittest

from context_benchmark.audit import audit_dataset


def turn(number: int, speaker: str, text: str) -> dict:
    return {"turn": number, "speaker": speaker, "text": text}


def unit(**kwargs) -> dict:
    base = {
        "id": "c001",
        "type": "standing",
        "scope": "global",
        "category": "communication",
        "text": "仅用邮件联系",
        "evidence_turns": [3],
        "valid_from": 3,
        "valid_to": None,
        "superseded_by": None,
        "rule_tags": [],
    }
    base.update(kwargs)
    return base


def dialogue(units, turns=None, distractors=None) -> dict:
    return {
        "dialogue_id": "p0-900",
        "turns": turns
        or [
            turn(1, "user", "我想咨询一下。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "以后都只用邮件联系我，不要打电话。"),
            turn(4, "agent", "明白。"),
        ],
        "candidate_truth": units,
        "distractors": distractors or [],
    }


class AuditTest(unittest.TestCase):
    def test_clean_dataset_has_no_findings(self):
        report = audit_dataset([dialogue([unit()])])
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["clean_rate"], 1.0)

    def test_empty_evidence_flagged(self):
        report = audit_dataset([dialogue([unit(evidence_turns=[])])])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("E0", codes)

    def test_evidence_out_of_range_flagged(self):
        report = audit_dataset([dialogue([unit(evidence_turns=[99])])])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("E1", codes)

    def test_standing_with_only_one_off_wording(self):
        turns = [turn(1, "user", "这次先别提交。"), turn(2, "agent", "好的。")]
        report = audit_dataset([dialogue([unit(evidence_turns=[1], valid_from=1)], turns=turns)])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertTrue({"T1", "C1"} & set(codes))

    def test_implicit_needs_three_evidence_turns(self):
        turns = [
            turn(1, "user", "先别提交。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "这次也别提交。"),
            turn(4, "agent", "好的。"),
        ]
        report = audit_dataset(
            [dialogue([unit(type="implicit", evidence_turns=[1, 3], valid_from=1)], turns=turns)]
        )
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("T5", codes)

    def test_implicit_exempt_from_lexical_overlap(self):
        turns = [
            turn(1, "user", "先别替我提交。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "这次也别提交。"),
            turn(4, "agent", "好的。"),
            turn(5, "user", "还是先别提交。"),
            turn(6, "agent", "好的。"),
        ]
        report = audit_dataset(
            [
                dialogue(
                    [
                        unit(
                            type="implicit",
                            text="提交前应等待用户自行检查",
                            evidence_turns=[1, 3, 5],
                            valid_from=1,
                        )
                    ],
                    turns=turns,
                )
            ]
        )
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertNotIn("C1", codes)
        self.assertNotIn("S2", codes)

    def test_superseded_requires_valid_to(self):
        turns = [
            turn(1, "user", "以后退款都原路退回。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "那条取消。"),
            turn(4, "agent", "已取消。"),
        ]
        report = audit_dataset(
            [dialogue([unit(type="superseded", evidence_turns=[1, 3], valid_from=1)], turns=turns)]
        )
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("T6", codes)

    def test_superseded_dangling_reference(self):
        turns = [
            turn(1, "user", "以后退款都原路退回。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "那条取消。"),
            turn(4, "agent", "已取消。"),
        ]
        report = audit_dataset(
            [
                dialogue(
                    [
                        unit(
                            type="superseded",
                            evidence_turns=[1, 3],
                            valid_from=1,
                            valid_to=2,
                            superseded_by="c999",
                        )
                    ],
                    turns=turns,
                )
            ]
        )
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("T9", codes)

    def test_superseded_scope_read_from_valid_from(self):
        """撤回轮的「这次」不应把原规则的作用域判成局部。"""
        turns = [
            turn(1, "user", "以后退款都原路退回。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "退款原路退回那条取消，这次先退到账户余额。"),
            turn(4, "agent", "已撤回。"),
        ]
        report = audit_dataset(
            [
                dialogue(
                    [
                        unit(
                            type="superseded",
                            text="退款原路退回",
                            evidence_turns=[1, 3],
                            valid_from=1,
                            valid_to=2,
                        )
                    ],
                    turns=turns,
                )
            ]
        )
        self.assertEqual(report["findings"], [])

    def test_distractor_must_appear_in_dialogue(self):
        distractor = {
            "id": "d001",
            "text": "默认使用短信提醒所有工单",
            "evidence_turns": [2],
            "category": "communication",
            "expected": "exclude",
        }
        report = audit_dataset([dialogue([unit()], distractors=[distractor])])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("D1", codes)

    def test_distractor_must_be_spoken_by_agent(self):
        turns = [
            turn(1, "user", "默认使用短信提醒所有工单。"),
            turn(2, "agent", "好的。"),
        ]
        distractor = {
            "id": "d001",
            "text": "默认使用短信提醒所有工单",
            "evidence_turns": [1],
            "category": "communication",
            "expected": "exclude",
        }
        report = audit_dataset([dialogue([unit()], turns=turns, distractors=[distractor])])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("D2", codes)

    def test_empty_dialogue_with_requirement_is_flagged(self):
        turns = [
            turn(1, "user", "以后不要给我打电话。"),
            turn(2, "agent", "好的。"),
        ]
        report = audit_dataset([dialogue([], turns=turns)])
        codes = [issue["code"] for row in report["findings"] for issue in row["issues"]]
        self.assertIn("M1", codes)

    def test_empty_dialogue_with_summary_wording_is_clean(self):
        turns = [
            turn(1, "user", "这次只是来咨询。"),
            turn(2, "agent", "好的。"),
            turn(3, "user", "今天先了解到这里。"),
            turn(4, "agent", "好的。"),
        ]
        report = audit_dataset([dialogue([], turns=turns)])
        self.assertEqual(report["findings"], [])


if __name__ == "__main__":
    unittest.main()
