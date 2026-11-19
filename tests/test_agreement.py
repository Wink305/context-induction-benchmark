import copy
import unittest

from context_benchmark.agreement import agreement_report, cohen_kappa, gwet_ac1
from context_benchmark.io import read_jsonl


class AgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogues = read_jsonl("data/pilot_v0/candidate_truth.jsonl")

    def annotations(self):
        return [{"dialogue_id": row["dialogue_id"], "annotations": copy.deepcopy(row["candidate_truth"])} for row in self.dialogues]

    def test_known_kappa(self):
        result = cohen_kappa(["S", "S", "O", "O"], ["S", "O", "S", "O"])
        self.assertAlmostEqual(result["observed_agreement"], 0.5)
        self.assertAlmostEqual(result["kappa"], 0.0)

    def test_identical_annotations_pass_and_no_conflicts(self):
        rows = self.annotations()
        result = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=50)
        self.assertEqual(result["overall"]["kappa"], 1.0)
        self.assertEqual(result["unresolved_conflict_n"], 0)
        self.assertTrue(result["gold_freeze_allowed"])

    def test_type_conflict_enters_adjudication_queue(self):
        rows_a = self.annotations()
        rows_b = self.annotations()
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        result = agreement_report(rows_a, rows_b, bootstrap_n=20)
        self.assertGreater(result["unresolved_conflict_n"], 0)
        self.assertFalse(result["gold_freeze_allowed"])
        self.assertIn("type_mismatch", result["adjudication_queue"][0]["reasons"])

    def test_extraction_disagreement_is_counted_not_hidden(self):
        """A unit that fails the text matcher must still lower kappa.

        Regression: previously kappa was computed only on matched units, so an
        annotator could erase a disagreement by paraphrasing past the matcher.
        """
        rows_a = self.annotations()
        rows_b = self.annotations()
        # Rewrite text so it cannot match the A unit at all -> unmatched_a/b.
        rows_b[0]["annotations"][0]["text"] = "完全无关的乱码内容甲乙丙丁"
        result = agreement_report(rows_a, rows_b, bootstrap_n=20)
        self.assertGreater(result["unmatched_a_n"], 0)
        self.assertGreater(result["unmatched_b_n"], 0)
        self.assertLess(result["overall"]["kappa"], result["matched_units_kappa_reference"]["kappa"])
        # One rewritten unit does not push the rate below 0.95, but it must be
        # reflected as a fractional rate and, crucially, inside kappa itself.
        self.assertLess(result["extraction_match_rate"]["value"], 1.0)
        self.assertFalse(result["gold_freeze_allowed"])

    def test_many_extraction_disagreements_breach_match_gate(self):
        rows_a = self.annotations()
        rows_b = self.annotations()
        for row in rows_b[:5]:
            for unit in row["annotations"]:
                unit["text"] = "完全无关的乱码内容甲乙丙丁"
        result = agreement_report(rows_a, rows_b, bootstrap_n=20)
        self.assertLess(result["extraction_match_rate"]["value"], 0.95)
        self.assertFalse(result["exit_checks"]["extraction_match_gte_0_95"])

    def test_absent_class_present_when_extraction_differs(self):
        rows_a = self.annotations()
        rows_b = self.annotations()
        rows_b[0]["annotations"][0]["text"] = "完全无关的乱码内容甲乙丙丁"
        result = agreement_report(rows_a, rows_b, bootstrap_n=20)
        self.assertGreater(result["per_type"]["__absent__"]["positive_n_annotator_a"], 0)
        self.assertGreater(result["per_type"]["__absent__"]["positive_n_annotator_b"], 0)

    def test_gwet_is_reported(self):
        result = gwet_ac1(["S", "S", "O"], ["S", "O", "O"])
        self.assertIn("ac1", result)


if __name__ == "__main__":
    unittest.main()
