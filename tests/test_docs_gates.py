"""Adversarial tests for the documentation-consistency gates.

A gate that cannot fail is worse than no gate: it reads as protection while
protecting nothing. Every gate in ``scripts/check_docs.py`` is therefore
exercised here against a deliberately broken input, and the checks assert
both directions — the gate fires on drift *and* stays quiet on correct input.

Run: PYTHONPATH=src python -m unittest tests.test_docs_gates -v
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_check_docs():
    spec = importlib.util.spec_from_file_location(
        "check_docs_under_test", ROOT / "scripts" / "check_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_docs = _load_check_docs()


class TestCountTestsHelper(unittest.TestCase):
    def test_count_tests_sees_this_file(self) -> None:
        total = check_docs.count_tests()
        self.assertGreater(total, 0)
        source = (ROOT / "tests" / "test_docs_gates.py").read_text(encoding="utf-8")
        self.assertIn("def test_", source)


class TestGate3TestCounts(unittest.TestCase):
    """The gate that was silently dead: it only knew the English word 'tests'."""

    def test_regex_matches_chinese_sentences(self) -> None:
        for sentence in ("197 个测试全绿", "47 项测试通过", "共 197 tests", "197 个单元测试"):
            self.assertIsNotNone(
                check_docs.TEST_COUNT_RE.search(sentence),
                f"the pattern failed to match: {sentence}",
            )

    def test_regex_ignores_unrelated_numbers(self) -> None:
        for sentence in ("40 段对话", "420 个单元", "5 段 48 条", "132/180 人机一致样本"):
            self.assertIsNone(
                check_docs.TEST_COUNT_RE.search(sentence),
                f"the pattern matched an unrelated number: {sentence}",
            )

    def test_stale_count_is_reported(self) -> None:
        problems = check_docs.gate_3_test_counts(
            {"README.md": "当前 197 个测试全绿。"}, tests_n=198
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("197", problems[0])
        self.assertIn("198", problems[0])

    def test_current_count_passes(self) -> None:
        problems = check_docs.gate_3_test_counts(
            {"README.md": "当前 197 个测试全绿。"}, tests_n=197
        )
        self.assertEqual(problems, [])

    def test_english_count_is_caught(self) -> None:
        problems = check_docs.gate_3_test_counts({"README.md": "47 tests passed"}, tests_n=197)
        self.assertEqual(len(problems), 1)

    def test_historical_comparison_is_exempt(self) -> None:
        problems = check_docs.gate_3_test_counts(
            {"README.md": "197 个测试全绿（原 180 + 新增 17）"}, tests_n=197
        )
        self.assertEqual(problems, [])

    def test_every_shipped_document_is_scanned(self) -> None:
        for name in ("P0_预检报告_AI对照.md", "P0_标注手册_v1.md", "任务交接_2026-11-21.md"):
            self.assertIn(name, check_docs.DOC_FILES)


class TestGate1Hashes(unittest.TestCase):
    def test_stale_hash_is_reported(self) -> None:
        stale = "a" * 64
        docs = {"README.md": f"candidate_truth SHA-256 `{stale}`"}
        problems = check_docs.gate_1_hashes(docs, truth_sha="b" * 64, tasks_sha="c" * 64)
        self.assertEqual(len(problems), 1)
        self.assertIn(stale[:12], problems[0])

    def test_current_hash_passes(self) -> None:
        current = "b" * 64
        docs = {"README.md": f"candidate_truth SHA-256 `{current}`"}
        problems = check_docs.gate_1_hashes(docs, truth_sha=current, tasks_sha="c" * 64)
        self.assertEqual(problems, [])

    def test_a_hash_without_a_hash_claim_is_not_a_problem(self) -> None:
        docs = {"README.md": f"参考编号 {('a' * 64)}"}
        problems = check_docs.gate_1_hashes(docs, truth_sha="b" * 64, tasks_sha="c" * 64)
        self.assertEqual(problems, [])


class TestGate2FrozenTasksHash(unittest.TestCase):
    def test_mismatch_is_reported(self) -> None:
        config = json.dumps({"frozen_tasks_sha256": "d" * 64})
        problems = check_docs.gate_2_frozen_tasks_hash("e" * 64, config_text=config)
        self.assertEqual(len(problems), 1)

    def test_match_passes(self) -> None:
        config = json.dumps({"frozen_tasks_sha256": "e" * 64})
        problems = check_docs.gate_2_frozen_tasks_hash("e" * 64, config_text=config)
        self.assertEqual(problems, [])

    def test_missing_key_is_reported(self) -> None:
        problems = check_docs.gate_2_frozen_tasks_hash("e" * 64, config_text="{}")
        self.assertEqual(len(problems), 1)


class TestGate6ImplementationStatus(unittest.TestCase):
    def test_illegal_state_is_reported(self) -> None:
        config = {
            "implementation_status": {
                "entries": [{"id": "x", "promise": "A thing", "state": "done", "evidence": ["README.md"]}]
            }
        }
        problems = check_docs.gate_6_implementation_status({"README.md": "A thing"}, config)
        self.assertTrue(any("not a legal state" in problem for problem in problems))

    def test_promise_missing_from_readme_is_reported(self) -> None:
        config = {
            "implementation_status": {
                "entries": [{"id": "x", "promise": "Not in the README", "state": "partial", "evidence": []}]
            }
        }
        problems = check_docs.gate_6_implementation_status({"README.md": "something else"}, config)
        self.assertTrue(any("not in the README table" in problem for problem in problems))

    def test_missing_evidence_path_is_reported(self) -> None:
        config = {
            "implementation_status": {
                "entries": [{"id": "x", "promise": "A thing", "state": "implemented", "evidence": ["nope.md"]}]
            }
        }
        problems = check_docs.gate_6_implementation_status({"README.md": "A thing"}, config)
        self.assertTrue(any("does not exist" in problem for problem in problems))

    def test_implemented_without_evidence_is_reported(self) -> None:
        config = {
            "implementation_status": {
                "entries": [{"id": "x", "promise": "A thing", "state": "implemented", "evidence": []}]
            }
        }
        problems = check_docs.gate_6_implementation_status({"README.md": "A thing"}, config)
        self.assertTrue(any("no evidence" in problem for problem in problems))


class TestGate7FalseTriggerProbes(unittest.TestCase):
    def test_missing_control_probe_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.jsonl"
            path.write_text(
                json.dumps({"dialogue_id": "d", "probes": [{"probe_id": "p", "related": True}]}) + "\n",
                encoding="utf-8",
            )
            problems = check_docs.gate_7_false_trigger_probes(path)
            self.assertEqual(len(problems), 1)

    def test_control_probe_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.jsonl"
            path.write_text(
                json.dumps({"dialogue_id": "d", "probes": [{"probe_id": "p", "related": False}]}) + "\n",
                encoding="utf-8",
            )
            problems = check_docs.gate_7_false_trigger_probes(path)
            self.assertEqual(problems, [])


class TestGate8SelfCheckVersions(unittest.TestCase):
    def test_stale_matcher_version_is_reported(self) -> None:
        artifact = {"matcher_version": "old", "scorer_version": "scorer-v1"}
        problems = check_docs.gate_8_self_check_versions(
            artifact,
            'MATCHER_VERSION = "new"',
            'SCORER_VERSION = "scorer-v1"',
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("matcher_version", problems[0])


class TestGate10CalibrationProvenance(unittest.TestCase):
    SCORING = 'TEXT_MATCH_MIN = 0.10\nANCHOR_COVERAGE_MIN = 0.08\n'

    def _base(self, state: str = "uncalibrated") -> dict:
        return {
            "state": state,
            "thresholds": {
                "text_min": 0.10,
                "anchor_coverage_min": 0.08,
                "evidence_iou_min": None,
                "slot_match_required": False,
            },
            "provenance": {
                "calibration_set": None,
                "calibration_set_sha256": None,
                "annotators": [],
                "adjudicator": None,
                "fitted_at": None,
                "sample_n": 0,
                "sensitivity": None,
                "specificity": None,
                "split": None,
            },
        }

    def test_shipped_config_is_valid(self) -> None:
        calibration = json.loads((ROOT / "configs" / "calibration.json").read_text(encoding="utf-8"))
        problems = check_docs.gate_10_calibration_provenance(
            calibration, (ROOT / "src" / "context_benchmark" / "scoring.py").read_text(encoding="utf-8")
        )
        self.assertEqual(problems, [])

    def test_threshold_that_disagrees_with_the_code_is_reported(self) -> None:
        calibration = self._base()
        calibration["thresholds"]["text_min"] = 0.42
        problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING)
        self.assertTrue(any("text_min" in problem for problem in problems))

    def test_uncalibrated_with_provenance_is_reported(self) -> None:
        calibration = self._base()
        calibration["provenance"]["sample_n"] = 12
        problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING)
        self.assertTrue(any("provenance carries values" in problem for problem in problems))

    def test_uncalibrated_cannot_claim_p1_thresholds(self) -> None:
        calibration = self._base()
        calibration["thresholds"]["evidence_iou_min"] = 0.5
        problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING)
        self.assertTrue(any("must not claim the P1 thresholds" in problem for problem in problems))

    def test_calibrated_without_provenance_is_reported(self) -> None:
        calibration = self._base(state="calibrated")
        problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING)
        self.assertTrue(any("is empty" in problem for problem in problems))

    def test_calibrated_with_a_missing_calibration_set_is_reported(self) -> None:
        calibration = self._base(state="calibrated")
        calibration["provenance"].update(
            {
                "calibration_set": "data/annotation/gold_v1.jsonl",
                "calibration_set_sha256": "f" * 64,
                "annotators": ["A", "B"],
                "adjudicator": "C",
                "fitted_at": "2026-12-01",
                "sample_n": 48,
                "sensitivity": 0.9,
                "specificity": 0.8,
                "split": "80/20",
            }
        )
        problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING)
        self.assertTrue(any("does not exist" in problem for problem in problems))

    def test_calibrated_with_a_wrong_hash_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "gold.jsonl"
            sample.write_text("{}\n", encoding="utf-8")
            calibration = self._base(state="calibrated")
            calibration["provenance"].update(
                {
                    "calibration_set": "gold.jsonl",
                    "calibration_set_sha256": "f" * 64,
                    "annotators": ["A", "B"],
                    "adjudicator": "C",
                    "fitted_at": "2026-12-01",
                    "sample_n": 48,
                    "sensitivity": 0.9,
                    "specificity": 0.8,
                    "split": "80/20",
                }
            )
            problems = check_docs.gate_10_calibration_provenance(calibration, self.SCORING, root=root)
            self.assertTrue(any("hash does not match" in problem for problem in problems))


class TestCurrentRepositoryPasses(unittest.TestCase):
    def test_repository_is_consistent(self) -> None:
        self.assertEqual(check_docs.main(), 0)


if __name__ == "__main__":
    unittest.main()
