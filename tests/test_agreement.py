import copy
import unittest

from context_benchmark.agreement import (
    KAPPA_CI_LOWER_MIN,
    KAPPA_POINT_MIN,
    agreement_report,
    cohen_kappa,
    gwet_ac1,
)
from context_benchmark.io import read_jsonl
from context_benchmark.validation import annotation_content_hash


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


class KappaOnlyGateTests(unittest.TestCase):
    """The gate is Cohen's kappa — for every class, rare ones included."""

    @classmethod
    def setUpClass(cls):
        cls.dialogues = read_jsonl("data/pilot_v0/candidate_truth.jsonl")

    def annotations(self):
        return [
            {"dialogue_id": row["dialogue_id"], "annotations": copy.deepcopy(row["candidate_truth"])}
            for row in self.dialogues
        ]

    @staticmethod
    def _unit(text, kind):
        return {
            "id": f"c{text}",
            "type": kind,
            "text": text,
            "scope": "global",
            "category": "process",
            "evidence_turns": [1],
            "valid_from": 1,
            "valid_to": None,
            "superseded_by": None,
            "rule_tags": [],
        }

    @classmethod
    def _dataset(cls, plan):
        """One dialogue per unit; A and B annotate the same text."""
        rows_a = []
        rows_b = []
        for index, (kind_a, kind_b) in enumerate(plan):
            text = f"用例{index:03d}"
            rows_a.append({"dialogue_id": f"d{index:03d}", "annotations": [cls._unit(text, kind_a)]})
            rows_b.append({"dialogue_id": f"d{index:03d}", "annotations": [cls._unit(text, kind_b)]})
        return rows_a, rows_b

    @classmethod
    def _rare_class_dataset(cls):
        """100 units where the rare class disagrees on half of its instances.

        ``superseded`` is used by each annotator 10 times out of 100 units and
        they agree on only 5 of them: kappa collapses to ~0.44 while AC1 sits
        at ~0.88.  The old gate let this class pass on AC1.
        """
        plan = []
        for index in range(100):
            if index < 5:
                plan.append(("superseded", "superseded"))
            elif index < 10:
                plan.append(("superseded", "standing"))
            elif index < 15:
                plan.append(("standing", "superseded"))
            else:
                plan.append(("standing", "standing"))
        return cls._dataset(plan)

    @classmethod
    def _agreed_rare_class_dataset(cls):
        """Same rare class, no disagreement: it must still pass the gate."""
        plan = [
            ("superseded", "superseded") if index < 5 else ("standing", "standing")
            for index in range(100)
        ]
        return cls._dataset(plan)

    def test_gate_metric_is_kappa_for_every_class(self):
        rows = self.annotations()
        result = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=50)
        for ctype, item in result["per_type"].items():
            self.assertEqual(item["threshold_metric"], "cohen_kappa", ctype)
            self.assertEqual(item["threshold_used"], KAPPA_POINT_MIN, ctype)
            self.assertTrue(item["gwet_ac1_report_only"], ctype)

    def test_rare_class_with_low_kappa_fails_even_when_ac1_is_high(self):
        rows_a, rows_b = self._rare_class_dataset()
        result = agreement_report(rows_a, rows_b, bootstrap_n=100)
        item = result["per_type"]["superseded"]
        self.assertTrue(item["rare_class"])
        # AC1 would have let this class through; kappa does not.
        self.assertGreaterEqual(item["gwet_ac1"], 0.75)
        self.assertLess(item["kappa"], KAPPA_POINT_MIN)
        self.assertFalse(item["pass_gate"])
        self.assertTrue(any("cohen_kappa" in reason for reason in item["gate_failures"]))
        self.assertFalse(result["exit_checks"]["per_type_pass"])
        self.assertFalse(result["gold_freeze_allowed"])

    def test_ac1_can_be_near_perfect_while_kappa_is_negative(self):
        """The sharper failure mode: AC1 says "excellent", kappa says "worse than chance".

        Four rare-class units go to different annotators while ninety-six
        are untouched by either.  The rare class is used ten times each
        way, so AC1's chance-corrected agreement lands near 0.96 — a number
        any reader would call outstanding — while Cohen's kappa, which
        charges for the chance overlap, goes negative.  A gate that falls
        back to AC1 would wave this through.
        """
        plan = []
        for index in range(100):
            if index in (3, 11):
                plan.append(("superseded", "standing"))
            elif index in (7, 19):
                plan.append(("standing", "superseded"))
            else:
                plan.append(("standing", "standing"))
        rows_a, rows_b = self._dataset(plan)
        result = agreement_report(rows_a, rows_b, bootstrap_n=200)
        item = result["per_type"]["superseded"]
        self.assertGreater(item["gwet_ac1"], 0.9)
        self.assertLess(item["kappa"], 0.0)
        self.assertFalse(item["pass_gate"])
        self.assertFalse(result["exit_checks"]["per_type_pass"])
        self.assertFalse(result["gold_freeze_allowed"])

    def test_rare_class_that_agrees_still_passes(self):
        rows_a, rows_b = self._agreed_rare_class_dataset()
        result = agreement_report(rows_a, rows_b, bootstrap_n=100)
        item = result["per_type"]["superseded"]
        self.assertTrue(item["rare_class"])
        self.assertTrue(item["pass_gate"], item["gate_failures"])

    def test_class_neither_annotator_used_is_not_a_failure(self):
        rows = self.annotations()
        result = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=50)
        absent = result["per_type"]["__absent__"]
        self.assertTrue(absent["both_annotators_never_used_class"])
        self.assertTrue(absent["pass_gate"])

    def test_ac1_is_reported_for_information_only(self):
        rows = self.annotations()
        result = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=50)
        self.assertTrue(result["gwet_ac1"]["report_only"])
        self.assertIn("kappa", result["gwet_ac1"]["note"].lower())
        self.assertIsNotNone(result["gwet_ac1"]["overall"])

    def test_report_records_annotator_provenance(self):
        rows = self.annotations()
        for row in rows:
            row["annotator"] = "A"
        human = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=20)
        self.assertEqual(human["annotator_type"], "human")
        ai_rows = copy.deepcopy(rows)
        for row in ai_rows:
            row["annotator"] = "AI-precheck"
        ai = agreement_report(ai_rows, copy.deepcopy(ai_rows), bootstrap_n=20)
        self.assertEqual(ai["annotator_type"], "ai_precheck")


    def test_report_binds_the_annotation_sets_by_content_hash(self):
        """A report must name the exact bytes it was computed from.

        This is what stops a hand-written agreement report from being used
        to freeze generator truth: the freezer recomputes both hashes and
        refuses a report that does not match.
        """
        rows_a = self.annotations()
        rows_b = self.annotations()
        rows_b[0]["annotations"][0]["text"] += "（人工改写）"
        report = agreement_report(rows_a, rows_b, bootstrap_n=20)
        self.assertEqual(report["annotations_a_sha256"], annotation_content_hash(rows_a))
        self.assertEqual(report["annotations_b_sha256"], annotation_content_hash(rows_b))
        self.assertNotEqual(report["annotations_a_sha256"], report["annotations_b_sha256"])

    def test_perfect_agreement_is_flagged_but_not_failed(self):
        """A vacuous agreement measurement is visible, yet still not a gate."""
        rows = self.annotations()
        perfect = agreement_report(rows, copy.deepcopy(rows), bootstrap_n=50)
        self.assertEqual(perfect["overall"]["kappa"], 1.0)
        self.assertTrue(perfect["suspect_self_agreement"])
        self.assertIn("labelling twice", perfect["suspect_self_agreement_note"])
        # A flag is not a verdict: the exit checks are unchanged.
        self.assertTrue(perfect["gold_freeze_allowed"])

        rows_b = self.annotations()
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        with_conflict = agreement_report(rows, rows_b, bootstrap_n=50)
        self.assertFalse(with_conflict["suspect_self_agreement"])
        self.assertFalse(with_conflict["gold_freeze_allowed"])


if __name__ == "__main__":
    unittest.main()
