"""Tests for the threshold-fitting tool.

The point of these tests is not that the numbers come out — it is that the
tool refuses to do the things that would make the resulting numbers look
better than they are: fitting against generator truth, and declaring a
threshold "calibrated" without an adjudicator.

Run: PYTHONPATH=src python -m unittest tests.test_fit_thresholds -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fit_thresholds_under_test", ROOT / "scripts" / "fit_thresholds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = _load_module()

TURNS = {
    1: {"turn": 1, "speaker": "user", "text": "以后都只用短信通知我，别打手机。"},
    2: {"turn": 2, "speaker": "agent", "text": "好的，以后都用短信，不拨手机。"},
}
GOLD = [{
    "id": "c001", "type": "standing", "text": "以后都只用短信通知我，别打手机",
    "evidence_turns": [1], "valid_from": 1, "valid_to": None,
    "scope": "global", "superseded_by": None, "category": "communication",
}]
CORRECT = [{
    "id": "c001", "type": "standing", "text": "以后都只用短信通知我，别打手机",
    "evidence_turns": [1], "valid_from": 1, "valid_to": None,
    "scope": "global", "superseded_by": None, "category": "communication",
}]
UNRELATED = [{
    "id": "c001", "type": "standing", "text": "今天天气不错适合出门",
    "evidence_turns": [1], "valid_from": 1, "valid_to": None,
    "scope": "global", "superseded_by": None, "category": "communication",
}]
OPPOSED = [{
    "id": "c001", "type": "standing", "text": "以后都只用邮件通知我，别打手机",
    "evidence_turns": [1], "valid_from": 1, "valid_to": None,
    "scope": "global", "superseded_by": None, "category": "communication",
}]


class EvaluateTests(unittest.TestCase):
    def test_an_exact_prediction_is_a_true_positive(self):
        result = fit.evaluate(GOLD, CORRECT, TURNS, 0.10, 0.08)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_an_unrelated_prediction_is_not_credited(self):
        result = fit.evaluate(GOLD, UNRELATED, TURNS, 0.10, 0.08)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["recall"], 0.0)

    def test_a_semantically_opposed_prediction_is_a_false_positive(self):
        """The whole point of the semantic guard: same characters, wrong meaning."""
        result = fit.evaluate(GOLD, OPPOSED, TURNS, 0.10, 0.08)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["precision"], 0.0)

    def test_a_missing_prediction_is_a_false_negative(self):
        result = fit.evaluate(GOLD, [], TURNS, 0.10, 0.08)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["recall"], 0.0)
        self.assertIsNone(result["precision"])


class ThresholdSelectionTests(unittest.TestCase):
    def _grid(self):
        return [
            {"precision": 1.0, "recall": 0.5, "f1": 0.667, "text_min": 0.10, "anchor_coverage_min": 0.08},
            {"precision": 0.90, "recall": 0.90, "f1": 0.90, "text_min": 0.20, "anchor_coverage_min": 0.15},
        ]

    def test_precision_floor_wins_over_raw_recall(self):
        best, rule, degraded = fit.choose_threshold(self._grid(), 0.95)
        self.assertEqual(best["text_min"], 0.10)
        self.assertIn("precision >= 0.95", rule)
        self.assertFalse(degraded)

    def test_falls_back_to_f1_when_the_floor_cannot_be_met(self):
        grid = [
            {"precision": 0.98, "recall": 0.5, "f1": 0.667, "text_min": 0.10, "anchor_coverage_min": 0.08},
            {"precision": 0.90, "recall": 0.90, "f1": 0.90, "text_min": 0.20, "anchor_coverage_min": 0.15},
        ]
        best, rule, degraded = fit.choose_threshold(grid, 0.99)
        self.assertEqual(best["text_min"], 0.20)
        self.assertIn("no threshold reached", rule)
        self.assertTrue(degraded)

    def test_an_empty_grid_is_an_error(self):
        with self.assertRaises(ValueError):
            fit.choose_threshold([], 0.95)


class CrossValidationTests(unittest.TestCase):
    def _pairs(self, n: int = 10):
        pairs = []
        for index in range(n):
            dialogue_id = f"d-{index:03d}"
            pairs.append({
                "dialogue_id": dialogue_id,
                "gold": GOLD,
                "predictions": CORRECT,
                "turns": TURNS,
                "fold": fit.fold_of(dialogue_id, 5),
            })
        return pairs

    def test_folds_are_stable_and_spread(self):
        first = [fit.fold_of(f"d-{index:03d}", 5) for index in range(50)]
        second = [fit.fold_of(f"d-{index:03d}", 5) for index in range(50)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_out_of_fold_summary_is_produced(self):
        _, summary = fit.cross_validate(self._pairs(), (0.10,), (0.08,), 0.95, 5)
        self.assertEqual(summary["out_of_fold"]["tp"], 10)
        self.assertEqual(summary["out_of_fold"]["fp"], 0)
        self.assertEqual(summary["out_of_fold"]["recall"], 1.0)
        self.assertGreaterEqual(summary["folds_n"], 2)

    def test_a_degraded_fold_is_flagged(self):
        pairs = self._pairs()
        for pair in pairs:
            pair["predictions"] = OPPOSED
        _, summary = fit.cross_validate(pairs, (0.10,), (0.08,), 0.95, 5)
        self.assertTrue(summary["degraded_any_fold"])


class ConfigWritingTests(unittest.TestCase):
    """End to end: fit on a small human-labelled set, then satisfy the gate."""

    def _load_check_docs(self):
        spec = importlib.util.spec_from_file_location(
            "check_docs_under_test", ROOT / "scripts" / "check_docs.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_written_config_satisfies_the_repository_gate(self):
        check_docs = self._load_check_docs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialogue_ids = [f"d-{index:03d}" for index in range(10)]
            gold = root / "gold_v1.jsonl"
            gold.write_text(
                "".join(
                    json.dumps({"dialogue_id": d, "annotator": "A", "annotations": GOLD}) + "\n"
                    for d in dialogue_ids
                ),
                encoding="utf-8",
            )
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                "".join(
                    json.dumps({
                        "dialogue_id": d,
                        "turns": [
                            {"turn": 1, "speaker": "user", "text": "以后都只用短信通知我，别打手机。"},
                            {"turn": 2, "speaker": "agent", "text": "好的。"},
                        ],
                    }) + "\n"
                    for d in dialogue_ids
                ),
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps({"dialogue_id": d, "output": {"constraints": CORRECT}}) + "\n"
                    for d in dialogue_ids
                ),
                encoding="utf-8",
            )
            target = root / "calibration.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = fit.main([
                    "--gold", str(gold), "--predictions", str(predictions), "--tasks", str(tasks),
                    "--write", str(target), "--adjudicator", "Reviewer C",
                ])
            self.assertEqual(code, 0)
            config = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(config["state"], "calibrated")
            self.assertEqual(config["provenance"]["adjudicator"], "Reviewer C")
            self.assertEqual(config["provenance"]["sample_n"], 10)
            self.assertIn("out_of_fold", config["provenance"]["split"])
            scoring_source = (ROOT / "src" / "context_benchmark" / "scoring.py").read_text(encoding="utf-8")
            problems = check_docs.gate_10_calibration_provenance(config, scoring_source, root=root)
            self.assertEqual(problems, [])


class RefusalTests(unittest.TestCase):
    def test_generator_truth_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "candidate_truth.jsonl"
            gold.write_text("{}\n", encoding="utf-8")
            tasks = root / "tasks.jsonl"
            tasks.write_text("{}\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                code = fit.main([
                    "--gold", str(gold), "--predictions", str(tasks), "--tasks", str(tasks),
                ])
            self.assertEqual(code, 2)

    def test_writing_without_an_adjudicator_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "gold_v1.jsonl"
            gold.write_text(
                json.dumps({
                    "dialogue_id": "d-000",
                    "annotator": "A",
                    "annotations": GOLD,
                }) + "\n",
                encoding="utf-8",
            )
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps({
                    "dialogue_id": "d-000",
                    "turns": [
                        {"turn": 1, "speaker": "user", "text": "以后都只用短信通知我，别打手机。"},
                        {"turn": 2, "speaker": "agent", "text": "好的。"},
                    ],
                }) + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"dialogue_id": "d-000", "output": {"constraints": CORRECT}}) + "\n",
                encoding="utf-8",
            )
            target = root / "calibration.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = fit.main([
                    "--gold", str(gold), "--predictions", str(predictions), "--tasks", str(tasks),
                    "--write", str(target),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
