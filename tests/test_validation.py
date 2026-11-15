import copy
import unittest

from context_benchmark.io import read_jsonl
from context_benchmark.validation import parse_output, public_constraint, validate_candidate_dialogue, validate_output

PATH = "data/pilot_v0/candidate_truth.jsonl"


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl(PATH)[0]

    def output(self):
        return {"constraints": [public_constraint(item) for item in self.dialogue["candidate_truth"]]}

    def test_all_generated_candidates_validate(self):
        reports = [validate_candidate_dialogue(row) for row in read_jsonl(PATH)]
        self.assertTrue(all(report.valid for report in reports))

    def test_shape_valid_but_nonexistent_evidence_is_semantic_error(self):
        output = self.output()
        output["constraints"][0]["evidence_turns"] = [999]
        output["constraints"][0]["valid_from"] = 999
        report = validate_output(output, self.dialogue["turns"])
        self.assertFalse(report.valid)
        self.assertIn("semantic_evidence_missing", {issue.code for issue in report.issues})

    def test_agent_turn_cannot_be_user_evidence(self):
        output = self.output()
        output["constraints"][0]["evidence_turns"] = [2]
        output["constraints"][0]["valid_from"] = 2
        report = validate_output(output, self.dialogue["turns"])
        self.assertIn("semantic_evidence_not_user", {issue.code for issue in report.issues})

    def test_implicit_requires_three_turns(self):
        output = self.output()
        implicit = next(item for item in output["constraints"] if item["type"] == "implicit")
        implicit["evidence_turns"] = implicit["evidence_turns"][:2]
        report = validate_output(output, self.dialogue["turns"])
        self.assertIn("semantic_implicit_evidence", {issue.code for issue in report.issues})

    def test_missing_supersession_target(self):
        output = self.output()
        replaced = next(item for item in output["constraints"] if item["superseded_by"])
        replaced["superseded_by"] = "c999"
        report = validate_output(output, self.dialogue["turns"])
        self.assertIn("semantic_superseded_target", {issue.code for issue in report.issues})

    def test_markdown_is_not_silently_accepted(self):
        _, report = parse_output('```json\n{"constraints":[]}\n```')
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()
