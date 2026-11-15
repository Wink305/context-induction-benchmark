import copy
import tempfile
import unittest
from pathlib import Path

from context_benchmark.adjudication import freeze_gold
from context_benchmark.agreement import agreement_report
from context_benchmark.io import read_json, read_jsonl


class AdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogues = read_jsonl("data/pilot_v0/candidate_truth.jsonl")

    def annotations(self):
        return [{"dialogue_id": row["dialogue_id"], "annotations": copy.deepcopy(row["candidate_truth"])} for row in self.dialogues]

    def public_annotations(self, rows):
        from context_benchmark.validation import public_constraint
        return [{"dialogue_id": row["dialogue_id"], "annotations": [public_constraint(item) for item in row["annotations"]]} for row in rows]

    def test_freeze_requires_real_adjudication_inputs_and_is_immutable(self):
        rows = self.annotations()
        agreement = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=20)
        with tempfile.TemporaryDirectory() as directory:
            manifest = freeze_gold(self.dialogues, self.public_annotations(rows), agreement, [], directory)
            self.assertEqual(manifest["gold_version"], "v1.0")
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            self.assertEqual(len(gold), 40)
            self.assertTrue(all(row["status"] == "human_adjudicated_gold" for row in gold))
            with self.assertRaises(FileExistsError):
                freeze_gold(self.dialogues, self.public_annotations(rows), agreement, [], directory)

    def test_pending_conflict_without_resolution_cannot_freeze(self):
        rows_a = self.annotations()
        rows_b = self.annotations()
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        agreement = agreement_report(rows_a, rows_b, bootstrap_n=20)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_gold(self.dialogues, self.public_annotations(rows_a), agreement, [], directory)


if __name__ == "__main__":
    unittest.main()
