"""Tests for the capability layer: retention, pruning and trap marking.

These are the three behaviours the benchmark is supposed to measure:

* **retention** — were the durable constraints kept, every one of them, and
  were the ones at the end of a long dialogue kept as well as the ones at
  the start?
* **pruning** — was the stuff that must not survive actually removed, without
  the model emptying its memory to look good on it?
* **traps** — did the model notice the planted traps, name them, and still
  handle the affected constraints correctly?

The tests are deliberately written against the *failure* modes the audit
found, not against a happy path: a prediction that copies the gold sentence
but cites the wrong turns, a model that deletes everything, and a prediction
from before the ``trap_flags`` contract existed.

Run: PYTHONPATH=src python -m unittest tests.test_capabilities -v
"""

from __future__ import annotations

import unittest

from context_benchmark.io import read_jsonl
from context_benchmark.ontology import TRAP_KINDS
from context_benchmark.scoring import (
    evidence_iou, score_capabilities, score_pruning, score_retention, score_traps,
)
from context_benchmark.validation import public_constraint, validate_output

PATH = "data/pilot_v0/candidate_truth.jsonl"


def _flag(kind: str = "withdrawn", turns=(15,), reason="第 15 轮撤回了这条规则", ref=None):
    return {"kind": kind, "target_turns": list(turns), "reason": reason, "constraint_ref": ref}


class ContractTests(unittest.TestCase):
    """The output contract now carries an optional ``trap_flags`` array."""

    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl(PATH)[0]
        cls.turns = cls.dialogue["turns"]

    def constraints(self):
        return [public_constraint(item) for item in self.dialogue["candidate_truth"]]

    def test_a_prediction_without_trap_flags_is_still_valid(self):
        report = validate_output({"constraints": self.constraints()}, self.turns)
        self.assertTrue(report.valid)
        self.assertEqual(report.trap_error_n, 0)

    def test_a_well_formed_trap_flag_passes(self):
        output = {"constraints": self.constraints(), "trap_flags": [_flag()]}
        report = validate_output(output, self.turns)
        self.assertTrue(report.valid)
        self.assertEqual(report.trap_error_n, 0)

    def test_an_unknown_kind_is_rejected(self):
        output = {"constraints": self.constraints(), "trap_flags": [_flag(kind="vibes")]}
        report = validate_output(output, self.turns)
        codes = {issue.code for issue in report.trap_issues}
        self.assertIn("schema_trap_kind", codes)
        for kind in TRAP_KINDS:
            self.assertNotIn(kind, {"vibes"})

    def test_a_missing_turn_is_rejected(self):
        output = {"constraints": self.constraints(), "trap_flags": [_flag(turns=(9999,))]}
        report = validate_output(output, self.turns)
        self.assertIn("semantic_trap_turn_missing", {issue.code for issue in report.trap_issues})

    def test_an_agent_turn_is_only_allowed_for_a_non_user_claim(self):
        """The agent's own claim is the classic trap on the other side."""
        agent_turn = next(turn["turn"] for turn in self.turns if turn["speaker"] == "agent")
        allowed = {"constraints": self.constraints(), "trap_flags": [
            _flag(kind="non_user_claim", turns=(agent_turn,), reason="客服单方声明")
        ]}
        self.assertEqual(validate_output(allowed, self.turns).trap_error_n, 0)
        wrong = {"constraints": self.constraints(), "trap_flags": [
            _flag(kind="withdrawn", turns=(agent_turn,))
        ]}
        codes = {issue.code for issue in validate_output(wrong, self.turns).trap_issues}
        self.assertIn("semantic_trap_agent_turn_kind", codes)

    def test_an_empty_reason_is_rejected(self):
        output = {"constraints": self.constraints(), "trap_flags": [_flag(reason="  ")]}
        report = validate_output(output, self.turns)
        self.assertIn("schema_trap_reason", {issue.code for issue in report.trap_issues})

    def test_a_dangling_constraint_ref_is_rejected(self):
        output = {"constraints": self.constraints(), "trap_flags": [_flag(ref="c999")]}
        report = validate_output(output, self.turns)
        self.assertIn("semantic_trap_ref_missing", {issue.code for issue in report.trap_issues})

    def test_a_broken_flag_does_not_invalidate_the_constraints(self):
        """One malformed flag must not zero the whole dialogue on axes A and C."""
        output = {"constraints": self.constraints(), "trap_flags": [_flag(kind="vibes")]}
        report = validate_output(output, self.turns)
        self.assertTrue(report.valid, "the constraints must stay valid")

    def test_an_unknown_root_key_is_still_rejected(self):
        report = validate_output({"constraints": [], "extra": 1}, self.turns)
        self.assertFalse(report.valid)


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl(PATH)[0]
        cls.gold = cls.dialogue["candidate_truth"]
        cls.turns = cls.dialogue["turns"]

    def prediction(self, constraints):
        return {"dialogue_id": self.dialogue["dialogue_id"], "output": {"constraints": constraints}}

    def test_a_perfect_prediction_retains_everything(self):
        row = self.prediction([public_constraint(item) for item in self.gold])
        result = score_retention(self.dialogue, row)
        self.assertEqual(result["unit_recall"]["value"], 1.0)
        self.assertEqual(result["full_set_exact"], 1)

    def test_dropping_one_constraint_breaks_the_full_set(self):
        constraints = [public_constraint(item) for item in self.gold]
        durable = [
            index for index, item in enumerate(self.gold) if item["type"] in {"standing", "implicit"}
        ]
        keep = [item for index, item in enumerate(constraints) if index != durable[0]]
        result = score_retention(self.dialogue, self.prediction(keep))
        self.assertLess(result["unit_recall"]["value"], 1.0)
        self.assertEqual(result["full_set_exact"], 0)
        self.assertEqual(result["retained_n"], result["active_gold_n"] - 1)

    def test_depth_buckets_cover_every_durable_constraint(self):
        row = self.prediction([public_constraint(item) for item in self.gold])
        result = score_retention(self.dialogue, row)
        buckets = result["retention_by_depth"]
        self.assertEqual(sum(bucket["gold_n"] for bucket in buckets.values()), result["active_gold_n"])
        self.assertEqual(sum(bucket["retained_n"] for bucket in buckets.values()), result["retained_n"])

    def test_incomplete_evidence_is_caught_once_the_iou_gate_is_on(self):
        """Citing one of the two turns is not the same as citing the right evidence."""
        constraints = [public_constraint(item) for item in self.gold]
        first = next(index for index, item in enumerate(self.gold) if item["type"] == "standing")
        original = list(self.gold[first]["evidence_turns"])
        self.assertGreater(len(original), 1, "this dialogue needs a multi-turn evidence set")
        constraints[first] = dict(constraints[first])
        constraints[first]["evidence_turns"] = original[:1]
        constraints[first]["valid_from"] = original[0]
        loose = score_retention(self.dialogue, self.prediction(constraints))
        strict = score_retention(
            self.dialogue, self.prediction(constraints),
            {
                "text_min": 0.10, "anchor_coverage_min": 0.08,
                "evidence_iou_min": 0.9, "slot_match_required": False,
            },
        )
        self.assertLess(loose["evidence_iou_mean"], 1.0)
        self.assertEqual(loose["retained_n"], strict["retained_n"] + 1)

    def test_evidence_pointing_at_the_wrong_turn_is_rejected_outright(self):
        """The anchor gate refuses a constraint that cannot be read off the turns it cites."""
        constraints = [public_constraint(item) for item in self.gold]
        first = next(index for index, item in enumerate(self.gold) if item["type"] == "standing")
        gold_turns = set(self.gold[first]["evidence_turns"])
        other_user_turn = next(
            turn["turn"] for turn in self.turns
            if turn["speaker"] == "user" and turn["turn"] not in gold_turns
        )
        constraints[first] = dict(constraints[first])
        constraints[first]["evidence_turns"] = [other_user_turn]
        constraints[first]["valid_from"] = other_user_turn
        loose = score_retention(self.dialogue, self.prediction(constraints))
        self.assertLess(loose["retained_n"], loose["active_gold_n"])

    def test_evidence_iou_edges(self):
        self.assertEqual(evidence_iou([1, 2], [1, 2]), 1.0)
        self.assertEqual(evidence_iou([1, 2], [3, 4]), 0.0)
        self.assertEqual(evidence_iou([], []), 1.0)


class PruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl(PATH)[0]
        cls.gold = cls.dialogue["candidate_truth"]
        cls.turns = cls.dialogue["turns"]

    def prediction(self, constraints):
        return {"dialogue_id": self.dialogue["dialogue_id"], "output": {"constraints": constraints}}

    def test_keeping_a_one_off_as_standing_is_a_false_retain(self):
        constraints = [public_constraint(item) for item in self.gold]
        for index, item in enumerate(self.gold):
            if item["type"] == "one_off":
                constraints[index] = dict(constraints[index])
                constraints[index]["type"] = "standing"
                constraints[index]["scope"] = "global"
                break
        result = score_pruning(self.dialogue, self.prediction(constraints))
        self.assertGreater(result["false_retain_rate"]["value"], 0.0)
        self.assertLess(result["prune_recall"]["value"], 1.0)

    def test_emptying_the_memory_scores_badly_on_both_sides(self):
        """A model that deletes everything must not look good on pruning."""
        result = score_pruning(self.dialogue, self.prediction([]))
        self.assertEqual(result["prune_recall"]["value"], 1.0)
        self.assertEqual(result["over_prune_rate"]["value"], 1.0)
        self.assertTrue(result["degenerate_all_delete"])

    def test_a_clean_prediction_prunes_well_without_over_pruning(self):
        constraints = [public_constraint(item) for item in self.gold]
        result = score_pruning(self.dialogue, self.prediction(constraints))
        self.assertEqual(result["prune_recall"]["value"], 1.0)
        self.assertEqual(result["over_prune_rate"]["value"], 0.0)
        self.assertFalse(result["degenerate_all_delete"])

    def test_failures_are_reported_per_error_kind(self):
        constraints = [public_constraint(item) for item in self.gold]
        for index, item in enumerate(self.gold):
            if item["type"] in {"one_off", "superseded"}:
                constraints[index] = dict(constraints[index])
                constraints[index]["type"] = "standing"
                constraints[index]["scope"] = "global"
                constraints[index]["superseded_by"] = None
        result = score_pruning(self.dialogue, self.prediction(constraints))
        kinds = result["prune_recall_by_kind"]
        self.assertIn("one_off_as_standing", kinds)
        self.assertIn("expired_reuse", kinds)
        for bucket in kinds.values():
            self.assertLess(bucket["recall"]["value"], 1.0)

    def test_an_invalid_output_does_not_look_like_a_clean_prune(self):
        """A broken contract used to score as 'pruned everything correctly'."""
        broken = [{"id": "c001"}]
        result = score_pruning(self.dialogue, self.prediction(broken))
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["prune_recall"]["value"], 0.0)
        self.assertEqual(result["over_prune_rate"]["value"], 1.0)


class TrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl(PATH)[0]
        cls.gold = cls.dialogue["candidate_truth"]
        cls.turns = cls.dialogue["turns"]

    def prediction(self, constraints, flags=None):
        output = {"constraints": constraints}
        if flags is not None:
            output["trap_flags"] = flags
        return {"dialogue_id": self.dialogue["dialogue_id"], "output": output}

    def dialogue_with_traps(self):
        dialogue = dict(self.dialogue)
        withdrawn = next(item for item in self.gold if item["type"] == "superseded")
        dialogue["traps"] = [{
            "trap_id": "t01",
            "kind": "withdrawn",
            "target_turns": withdrawn["evidence_turns"],
            "affected_constraint_ids": [withdrawn["id"]],
            "expected_disposition": "drop",
        }]
        return dialogue, withdrawn

    def test_without_trap_gold_the_axis_is_not_evaluated(self):
        result = score_traps(self.dialogue, self.prediction([]))
        self.assertTrue(result["not_evaluated"])
        self.assertIsNone(result["trap_flag_recall"])

    def test_a_prediction_without_flags_is_not_evaluated_not_zero(self):
        dialogue, _ = self.dialogue_with_traps()
        result = score_traps(dialogue, self.prediction([]))
        self.assertTrue(result["not_evaluated"])
        self.assertIn("no trap_flags", result["not_evaluated_reason"])
        self.assertIsNone(result["trap_flag_recall"])

    def test_a_correctly_named_trap_is_recalled(self):
        dialogue, withdrawn = self.dialogue_with_traps()
        constraints = [public_constraint(item) for item in self.gold]
        flags = [_flag(kind="withdrawn", turns=withdrawn["evidence_turns"])]
        result = score_traps(dialogue, self.prediction(constraints, flags))
        self.assertFalse(result["not_evaluated"])
        self.assertEqual(result["trap_flag_recall"]["value"], 1.0)
        self.assertEqual(result["trap_flag_precision"]["value"], 1.0)

    def test_flagging_something_that_is_not_there_costs_precision(self):
        dialogue, _ = self.dialogue_with_traps()
        constraints = [public_constraint(item) for item in self.gold]
        flags = [_flag(kind="prompt_injection", turns=(1,))]
        result = score_traps(dialogue, self.prediction(constraints, flags))
        self.assertEqual(result["trap_flag_recall"]["value"], 0.0)
        self.assertEqual(result["trap_flag_precision"]["value"], 0.0)

    def test_marking_is_not_the_same_as_handling(self):
        """The trap is named, but the withdrawn rule is still there as active."""
        dialogue, withdrawn = self.dialogue_with_traps()
        constraints = [public_constraint(item) for item in self.gold]
        for index, item in enumerate(self.gold):
            if item["id"] == withdrawn["id"]:
                constraints[index] = dict(constraints[index])
                constraints[index]["type"] = "standing"
                constraints[index]["scope"] = "global"
        flags = [_flag(kind="withdrawn", turns=withdrawn["evidence_turns"])]
        result = score_traps(dialogue, self.prediction(constraints, flags))
        self.assertEqual(result["trap_flag_recall"]["value"], 1.0)
        self.assertLess(result["trap_disposition_accuracy"]["value"], 1.0)


class SchemaConsistencyTests(unittest.TestCase):
    """The published schema and the code must accept the same trap kinds."""

    def test_schema_and_code_list_the_same_trap_kinds(self):
        import json
        from pathlib import Path

        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "constraint-output.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_kinds = set(schema["properties"]["trap_flags"]["items"]["properties"]["kind"]["enum"])
        self.assertEqual(schema_kinds, set(TRAP_KINDS))

    def test_schema_keeps_trap_flags_optional(self):
        import json
        from pathlib import Path

        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "constraint-output.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertNotIn("trap_flags", schema["required"])
        self.assertIn("trap_flags", schema["properties"])


class TrapDataTests(unittest.TestCase):
    """The extension set must actually carry traps, or the trap axis is empty.

    The frozen pilot has none on purpose — it is a baseline and changing it
    would move a published hash. The randomised extension set is where new
    material goes, so the traps live there.
    """

    PATH = "data/contrib/pilot-v1-randomized/candidate_truth.jsonl"

    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(cls.PATH)

    def test_every_non_empty_dialogue_carries_traps(self):
        non_empty = [row for row in self.rows if row.get("candidate_truth")]
        self.assertTrue(non_empty)
        for row in non_empty:
            self.assertTrue(row.get("traps"), f"{row['dialogue_id']} carries no traps")

    def test_all_four_labelled_kinds_are_covered(self):
        kinds = {trap["kind"] for row in self.rows for trap in row.get("traps", [])}
        self.assertEqual(
            kinds,
            {"withdrawn", "replaced", "one_off_disguise", "non_user_claim"},
        )

    def test_traps_are_spread_across_the_dialogue(self):
        depths = {trap["depth_bucket"] for row in self.rows for trap in row.get("traps", [])}
        self.assertEqual(depths, {"early", "mid", "late"})

    def test_trap_turns_exist_in_the_dialogue(self):
        for row in self.rows:
            turns = {turn["turn"] for turn in row["turns"]}
            for trap in row.get("traps", []):
                for turn in trap["target_turns"]:
                    self.assertIn(turn, turns, f"{row['dialogue_id']} trap points at turn {turn}")

    def test_a_perfect_prediction_scores_perfectly_on_the_trap_axis(self):
        row = next(item for item in self.rows if item.get("traps") and item.get("candidate_truth"))
        constraints = [public_constraint(item) for item in row["candidate_truth"]]
        flags = [
            {
                "kind": trap["kind"],
                "target_turns": trap["target_turns"],
                "reason": f"{trap['kind']} at turn {max(trap['target_turns'])}",
                "constraint_ref": (trap["affected_constraint_ids"] or [None])[0],
            }
            for trap in row["traps"]
        ]
        prediction = {"dialogue_id": row["dialogue_id"], "output": {"constraints": constraints, "trap_flags": flags}}
        result = score_traps(row, prediction)
        self.assertFalse(result["not_evaluated"])
        self.assertEqual(result["trap_flag_recall"]["value"], 1.0)
        self.assertEqual(result["trap_flag_precision"]["value"], 1.0)
        self.assertEqual(result["trap_disposition_accuracy"]["value"], 1.0)
        self.assertEqual(result["missed_and_followed_n"], 0)

    def test_saying_nothing_about_the_traps_scores_zero_recall_not_not_evaluated(self):
        row = next(item for item in self.rows if item.get("traps") and item.get("candidate_truth"))
        constraints = [public_constraint(item) for item in row["candidate_truth"]]
        prediction = {"dialogue_id": row["dialogue_id"], "output": {"constraints": constraints, "trap_flags": []}}
        result = score_traps(row, prediction)
        self.assertFalse(result["not_evaluated"], "the gold has traps, so this must be scored")
        self.assertEqual(result["trap_flag_recall"]["value"], 0.0)


class AggregateTests(unittest.TestCase):
    def test_the_dataset_report_carries_all_three_axes(self):
        dialogues = read_jsonl(PATH)
        predictions = [
            {
                "dialogue_id": dialogue["dialogue_id"],
                "output": {"constraints": [public_constraint(item) for item in dialogue["candidate_truth"]]},
            }
            for dialogue in dialogues
        ]
        report = score_capabilities(dialogues, predictions)
        self.assertEqual(report["retention"]["unit_recall"]["value"], 1.0)
        self.assertEqual(report["retention"]["full_set_exact_rate"]["value"], 1.0)
        self.assertEqual(report["pruning"]["prune_recall"]["value"], 1.0)
        self.assertEqual(report["pruning"]["over_prune_rate"]["value"], 0.0)
        # No dialogue carries trap gold yet, so the trap axis must say so
        # rather than reporting a confident zero.
        self.assertEqual(report["traps"]["evaluated_dialogue_n"], 0)
        self.assertIsNone(report["traps"]["trap_flag_recall"])


if __name__ == "__main__":
    unittest.main()
