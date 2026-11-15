import json
import unittest

from context_benchmark.harness import ExecutorRunner, InductionRunner, NoCompressionGuard, make_fake_constraints, planned_conditions
from context_benchmark.io import read_jsonl
from context_benchmark.providers import MockProvider
from context_benchmark.scoring import score_counterfactual
from context_benchmark.validation import public_constraint


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogue = read_jsonl("data/pilot_v0/candidate_truth.jsonl")[0]
        cls.valid_output = json.dumps({"constraints": [public_constraint(item) for item in cls.dialogue["candidate_truth"]]}, ensure_ascii=False)

    def config(self):
        return {
            "model_id": "mock-inducer", "executor_model_id": "mock-executor", "pinned_version": "mock-1", "quant": "na",
            "temperature": 0, "max_tokens": 4096, "seed": 7, "max_retries": 2,
            "prompt_version": "induction-v1", "executor_prompt_version": "executor-v1",
            "budgets_chars": {"tight": 800, "medium": 2000, "generous": 6000},
        }

    def test_invalid_json_is_retried_then_valid(self):
        provider = MockProvider(["not-json", self.valid_output])
        result = InductionRunner(provider, self.config(), "prompt").run(self.dialogue)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["validation"]["schema_error_n"], 1)

    def test_persistent_failure_is_counted_and_logged(self):
        provider = MockProvider(["bad", "bad", "bad"])
        result = InductionRunner(provider, self.config(), "prompt").run(self.dialogue)
        self.assertFalse(result["valid"])
        self.assertTrue(result["failure_counted"])
        self.assertEqual(len(result["attempts"]), 3)

    def test_design_has_four_ablations_and_isolated_three_arms(self):
        conditions = planned_conditions()
        self.assertEqual(len(conditions), 6)
        self.assertEqual({mode for mode, _ in conditions}, {"full_context", "distilled", "raw_turns", "no_history"})
        self.assertIn(("no_history", "fake"), conditions)

    def test_fake_constraints_are_length_aligned(self):
        constraints = [public_constraint(item) for item in self.dialogue["candidate_truth"][:2]]
        fake = make_fake_constraints(constraints)
        for source, control in zip(constraints, fake):
            self.assertEqual(len(source["text"]), len(control["text"]))
            self.assertNotEqual(source["text"], control["text"])
            self.assertTrue(control["fake_control"])

    def test_true_none_fake_executor_flow(self):
        def choose(request):
            content = request.messages[1]["content"]
            if '"constraints":[]' in content or "fake_control" in content:
                return '{"choice":"b2"}'
            return '{"choice":"b1"}'
        provider = MockProvider(callback=choose)
        runner = ExecutorRunner(provider, self.config(), "executor prompt")
        constraints = [public_constraint(item) for item in self.dialogue["candidate_truth"]]
        rows = [runner.run_probe(self.dialogue, self.dialogue["probes"][0], constraints, "no_history", arm, "tight") for arm in ("true", "none", "fake")]
        self.assertEqual([row["compliant"] for row in rows], [1, 0, 0])
        scored = score_counterfactual(rows)
        self.assertEqual(scored["true_delta_mean"]["value"], 1.0)
        self.assertEqual(scored["fake_delta_mean"]["value"], 0.0)

    def test_r2_guard_fails_without_truncating(self):
        guard = NoCompressionGuard(100, len)
        before = guard.before_call("x" * 91, 10)
        self.assertFalse(before["allowed"])
        self.assertFalse(before["truncation"])
        after = guard.after_call(95, 10)
        self.assertFalse(after["task_success"])
        self.assertEqual(after["failure_reason"], "actual_context_overflow")
        missing = guard.after_call(None, None)
        self.assertEqual(missing["failure_reason"], "missing_actual_token_usage")


if __name__ == "__main__":
    unittest.main()
