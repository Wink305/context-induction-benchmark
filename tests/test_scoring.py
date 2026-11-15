import copy
import unittest

from context_benchmark.io import read_jsonl
from context_benchmark.scoring import score_counterfactual, score_dataset, score_dialogue
from context_benchmark.validation import public_constraint


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogues = read_jsonl("data/pilot_v0/candidate_truth.jsonl")
        cls.dialogue = cls.dialogues[0]

    def prediction(self, dialogue=None):
        dialogue = dialogue or self.dialogue
        return {"dialogue_id": dialogue["dialogue_id"], "output": {"constraints": [public_constraint(item) for item in dialogue["candidate_truth"]]}}

    def test_oracle_self_check(self):
        rows = [self.prediction(dialogue) for dialogue in self.dialogues]
        result = score_dataset(self.dialogues, rows)["aggregate"]
        self.assertEqual(result["must_include"]["value"], 1.0)
        self.assertEqual(result["fmr"]["value"], 1.0)
        self.assertEqual(result["empty_output_accuracy"], {"numerator": 5, "denominator": 5, "value": 1.0})

    def test_one_off_promoted_to_standing_is_overgeneralization(self):
        row = self.prediction()
        one_off = next(item for item in row["output"]["constraints"] if item["type"] == "one_off")
        one_off["type"] = "standing"
        one_off["scope"] = "global"
        result = score_dialogue(self.dialogue, row)
        self.assertEqual(result["overgeneralization"]["numerator"], 1)
        self.assertTrue(result["valid_output"])

    def test_unmatched_hallucination_is_counted(self):
        row = self.prediction()
        original = row["output"]["constraints"][0]
        fake = dict(original)
        fake.update({"id": "c999", "text": "所有回复必须写成七言诗", "evidence_turns": [1], "valid_from": 1})
        row["output"]["constraints"].append(fake)
        result = score_dialogue(self.dialogue, row)
        self.assertEqual(result["unmatched_prediction_n"], 1)
        self.assertEqual(result["unmatched_prediction_rate"]["denominator"], len(row["output"]["constraints"]))

    def test_distractor_acceptance_lowers_fmr(self):
        dialogue = next(row for row in self.dialogues if row["distractors"])
        row = self.prediction(dialogue)
        distractor = dialogue["distractors"][0]
        row["output"]["constraints"].append({
            "id": "c999", "type": "standing", "text": distractor["text"],
            "evidence_turns": [1], "valid_from": 1, "valid_to": None,
            "scope": "global", "superseded_by": None, "category": "communication",
        })
        result = score_dialogue(dialogue, row)
        self.assertEqual(result["fmr"]["numerator"], 1)
        self.assertEqual(result["fmr"]["denominator"], 2)

    def test_invalid_output_cannot_pass_empty_calibration(self):
        dialogue = next(row for row in self.dialogues if not row["candidate_truth"])
        result = score_dialogue(dialogue, {"dialogue_id": dialogue["dialogue_id"], "output": None})
        self.assertFalse(result["empty_output"]["correct"])

    def test_semantically_invalid_output_scores_zero_not_partial_credit(self):
        row = self.prediction()
        row["output"]["constraints"][0]["evidence_turns"] = [999]
        result = score_dialogue(self.dialogue, row)
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["must_include"]["numerator"], 0)
        self.assertEqual(result["classification"]["tp"], 0)
        self.assertEqual(result["raw_predicted_n"], 12)

    def test_zero_denominator_is_null_not_one(self):
        result = score_dialogue(self.dialogue, self.prediction())
        self.assertEqual(result["fmr"]["denominator"], 0)
        self.assertIsNone(result["fmr"]["value"])

    def test_fake_positive_delta_triggers_validity_warning(self):
        rows = []
        for arm, compliant in (("true", 1), ("none", 0), ("fake", 1)):
            rows.append({"probe_id": "p1", "arm": arm, "compliant": compliant, "related": True})
        result = score_counterfactual(rows)
        self.assertTrue(result["validity_warning_fake_positive"])
        self.assertEqual(result["true_delta_mean"]["value"], 1.0)
        self.assertEqual(result["fake_delta_mean"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
