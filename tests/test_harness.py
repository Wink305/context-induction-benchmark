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

    def test_design_is_the_six_condition_set_not_the_full_factorial(self):
        """The document specifies six conditions; the code used to run 4x3 = 12.

        A full factorial confounds the intervention with the background:
        under ``full_context`` and ``raw_turns`` the target constraint's own
        text is already in the history, so an injected arm cannot be compared
        against anything.
        """
        conditions = planned_conditions()
        self.assertEqual(len(conditions), 6)
        self.assertEqual(
            sorted(conditions),
            sorted([
                ("full_context", "none"), ("distilled", "none"),
                ("raw_turns", "none"), ("no_history", "none"),
                ("distilled", "true"), ("distilled", "fake"),
            ]),
        )

    def test_every_background_keeps_a_none_baseline(self):
        conditions = planned_conditions()
        for mode in ("full_context", "distilled", "raw_turns", "no_history"):
            self.assertIn((mode, "none"), conditions)

    def test_the_intervention_only_runs_where_the_target_is_isolated(self):
        conditions = planned_conditions()
        for background, arm in conditions:
            if arm in {"true", "fake"}:
                self.assertEqual(
                    background, "distilled",
                    f"the {arm} arm leaked into {background}, where the target text is in the history",
                )

    def test_fake_constraints_are_length_aligned_and_unlabeled(self):
        constraints = [public_constraint(item) for item in self.dialogue["candidate_truth"][:2]]
        fake = make_fake_constraints(constraints)
        for source, control in zip(constraints, fake):
            # Length-aligned without truncation: the control arm must be
            # readable Chinese, otherwise the model can identify the arm by the
            # broken sentence alone.
            self.assertLessEqual(abs(len(source["text"]) - len(control["text"])), 24)
            self.assertTrue(control["text"].endswith("。"))
            self.assertNotEqual(source["text"], control["text"])
            # No metadata field may reveal which arm a constraint came from.
            self.assertEqual(set(control), set(source))
            self.assertNotIn("fake_control", control)
            self.assertEqual(control["id"], source["id"])

    def test_fake_constraint_text_is_never_cut_mid_phrase(self):
        """The old implementation truncated to the target length ('以后所有回复都改成五')."""
        constraints = [public_constraint(item) for item in self.dialogue["candidate_truth"][:4]]
        for source, control in zip(constraints, make_fake_constraints(constraints)):
            self.assertNotIn("改成五", control["text"])
            self.assertNotIn("以后所有问题", control["text"][:6])
            self.assertTrue(control["text"].endswith("。"))

    def test_injected_prompt_does_not_leak_arm_metadata(self):
        from context_benchmark.harness import public_injected
        item = dict(public_constraint(self.dialogue["candidate_truth"][0]))
        item["fake_control"] = True
        item["internal_score"] = 99
        self.assertEqual(set(public_injected(item)), set(public_constraint(item)))
        self.assertNotIn("fake_control", public_injected(item))

    def test_true_none_fake_executor_flow(self):
        true_texts = {item["text"] for item in self.dialogue["candidate_truth"]}
        def choose(request):
            content = request.messages[1]["content"]
            # A correct executor follows whatever rule is injected; the none
            # arm has an empty intervention block. It must NOT rely on any
            # explicit "fake" marker, because none exists any more.
            injected_empty = '"constraints":[]' in content.split("干预要点：")[-1]
            follows_injected = any(text in content for text in true_texts)
            if injected_empty or not follows_injected:
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

    def test_missing_induced_target_is_not_scored(self):
        provider = MockProvider(['{"choice":"b1"}'])
        runner = ExecutorRunner(provider, self.config(), "executor prompt")
        row = runner.run_probe(
            self.dialogue, self.dialogue["probes"][0], [], "no_history", "true", "tight"
        )
        self.assertFalse(row["scorable"])
        self.assertEqual(row["error"], "target_missing_from_induction")
        self.assertEqual(provider.requests, [])

    def test_budget_truncation_is_recorded(self):
        from context_benchmark.harness import serialize_history
        dialogue = self.dialogue
        constraints = [public_constraint(item) for item in dialogue["candidate_truth"]]
        text, meta = serialize_history(dialogue, "distilled", constraints, budget_chars=200)
        self.assertTrue(meta["truncated"])
        self.assertGreater(len(meta["dropped_ids"]), 0)
        self.assertGreater(len(meta["selected_ids"]), 0)
        self.assertEqual(set(meta["selected_ids"]) & set(meta["dropped_ids"]), set())

    def test_no_compression_guard_fails_without_truncating(self):
        guard = NoCompressionGuard(100, len)
        before = guard.before_call("x" * 91, 10)
        self.assertFalse(before["allowed"])
        self.assertFalse(before["truncation"])
        after = guard.after_call(95, 10)
        self.assertFalse(after["task_success"])
        self.assertEqual(after["failure_reason"], "actual_context_overflow")
        missing = guard.after_call(None, None)
        self.assertEqual(missing["failure_reason"], "missing_actual_token_usage")

    def test_run_no_compression_is_end_to_end(self):
        from context_benchmark.harness import run_no_compression
        config = self.config()
        config["r2_context_limit_tokens"] = 10 ** 9
        provider = MockProvider([json.dumps({"constraints": []})])
        record = run_no_compression(provider, config, self.dialogue, "prompt")
        self.assertTrue(record["scorable"])
        self.assertTrue(record["task_success"])
        self.assertFalse(record["compression"])
        self.assertFalse(record["truncation"])

    def test_run_no_compression_overflow_fails_closed(self):
        from context_benchmark.harness import run_no_compression
        config = self.config()
        config["r2_context_limit_tokens"] = 10  # tiny window
        provider = MockProvider([json.dumps({"constraints": []})])
        record = run_no_compression(provider, config, self.dialogue, "prompt")
        self.assertFalse(record["task_success"])
        self.assertEqual(record["failure_reason"], "estimated_context_overflow")
        self.assertEqual(provider.requests, [])  # never called the model

    def test_survival_trace_is_end_to_end(self):
        from context_benchmark.harness import run_survival_trace
        from context_benchmark.io import read_jsonl
        truth = read_jsonl("data/pilot_v0/candidate_truth.jsonl")[0]

        def choose(request):
            return '{"choice":"follow"}'

        provider = MockProvider(callback=choose)
        record = run_survival_trace(provider, self.config(), self.dialogue, truth, "prompt")
        self.assertEqual(len(record["compliance"]), len(truth["candidate_truth"]))
        self.assertTrue(all(record["compliance"]))
        # E can feed the survival analyser directly.
        from context_benchmark.survival import survival_report
        report = survival_report([record], bootstrap_n=20)
        self.assertIn("median_survival_turn", report)


if __name__ == "__main__":
    unittest.main()
