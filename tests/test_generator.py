import unittest

from context_benchmark.io import read_json, read_jsonl


class GeneratorAcceptanceTests(unittest.TestCase):
    def test_coverage_contract(self):
        report = read_json("data/pilot_v0/coverage.json")
        self.assertTrue(report["accepted_for_annotation_start"])
        self.assertEqual(report["dialogues"], 40)
        self.assertEqual(report["candidate_units"], 420)
        self.assertGreaterEqual(report["dialogues_by_kind"]["empty"], 5)
        self.assertGreaterEqual(report["dialogues_by_kind"]["distractor"], 5)
        self.assertTrue(all(value >= 4 for value in report["rule_dialogue_coverage"].values()))

    def test_annotation_worksheet_has_no_answer_leakage(self):
        rows = read_jsonl("data/pilot_v0/annotation_worksheet.jsonl")
        self.assertEqual(len(rows), 40)
        for row in rows:
            self.assertNotIn("candidate_truth", row)
            self.assertNotIn("distractors", row)
            self.assertNotIn("probes", row)
            self.assertEqual(row["annotations"], [])

    def test_manifest_does_not_claim_gold(self):
        manifest = read_json("data/pilot_v0/manifest.json")
        self.assertIsNone(manifest["gold_version"])
        self.assertFalse(manifest["human_annotation_complete"])
        self.assertEqual(manifest["status"], "generated_unannotated_not_gold")


if __name__ == "__main__":
    unittest.main()
