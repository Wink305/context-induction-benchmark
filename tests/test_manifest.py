import shutil
import tempfile
import unittest
from pathlib import Path

from context_benchmark.io import read_json, sha256_file, write_json, write_jsonl
from context_benchmark.manifest import (
    REPO_ROOT, build_dataset_manifest, build_run_manifest, validate_submission,
)

BENCHMARK_CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmark.json"


def _codes(report):
    return [item["code"] for item in report["issues"]]


class RunSubmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.runtime = self.root / "runtime.json"
        self.prompt = self.root / "prompt.txt"
        self.tasks = self.root / "tasks.jsonl"
        self.preds = self.root / "preds.jsonl"
        self.logs = self.root / "calls.jsonl"
        self.package = self.root / "package"
        write_json(self.runtime, {
            "provider": "mock", "model_id": "mock-x", "pinned_version": "pin-1",
            "quant": None, "temperature": 0.0, "seed": 7, "prompt_version": "induction-v1",
        })
        self.prompt.write_text("PROMPT_VERSION: induction-v1\n", encoding="utf-8")
        write_jsonl(self.tasks, [{"dialogue_id": "x-001", "status": "generated_unannotated_not_gold"}])
        write_jsonl(self.preds, [{"dialogue_id": "x-001", "valid": True, "failure_counted": False}])
        write_jsonl(self.logs, [{
            "error": None, "validation": {"valid": True},
            "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 1.0, "cost": 0.0,
        }])
        self.build()

    def build(self, **kwargs):
        options = dict(
            runtime_path=self.runtime, tasks_path=self.tasks, prompt_path=self.prompt,
            predictions_path=self.preds, logs_path=self.logs, out_dir=self.package,
            benchmark_config_path=BENCHMARK_CONFIG, contributor="tester",
        )
        options.update(kwargs)
        return build_run_manifest(**options)

    def validate(self):
        return validate_submission(self.package, benchmark_config_path=BENCHMARK_CONFIG)

    def mutate(self, **changes):
        path = self.package / "submission.json"
        document = read_json(path)
        for key, value in changes.items():
            document[key] = value
        write_json(path, document)

    def test_fresh_package_is_valid_and_stamps_versions(self):
        report = self.validate()
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["kind"], "run_result")
        self.assertEqual(report["checked_files"], 2)
        document = read_json(self.package / "submission.json")
        self.assertEqual(document["benchmark_version"], "0.4.0")
        self.assertEqual(document["primary_dimension"], "A")
        self.assertFalse(document["scores_declared"])
        self.assertEqual(document["run"]["usage"]["call_n"], 1)
        self.assertEqual(document["run"]["denominator_disclosure"]["scored_n"], 1)
        self.assertTrue(document["evidence"]["src/context_benchmark/scoring.py"])
        self.assertEqual(document["files"]["predictions.jsonl"], sha256_file(self.package / "predictions.jsonl"))

    def test_tampered_evidence_hash_is_detected(self):
        """Changing scorer code hashes or the declared value must fail closed."""
        document = read_json(self.package / "submission.json")
        document["evidence"]["src/context_benchmark/scoring.py"] = "0" * 64
        write_json(self.package / "submission.json", document)
        self.assertIn("evidence_hash_mismatch", _codes(self.validate()))

    def test_missing_required_field_is_detected(self):
        document = read_json(self.package / "submission.json")
        for key in ("provenance", "status", "declared_limitations", "evidence"):
            document.pop(key, None)
        write_json(self.package / "submission.json", document)
        codes = _codes(self.validate())
        self.assertIn("missing_required_field", codes)

    def test_scorer_version_must_match_scoring_py(self):
        document = read_json(self.package / "submission.json")
        document["run"]["scorer_version"] = "scoring.py@000000000000"
        write_json(self.package / "submission.json", document)
        self.assertIn("scorer_version_mismatch", _codes(self.validate()))

    def test_human_gold_claim_needs_version_and_hash(self):
        document = read_json(self.package / "submission.json")
        document["dataset"]["evidence_grade"] = "human_gold"
        document["dataset"]["human_annotation_complete"] = True
        write_json(self.package / "submission.json", document)
        codes = _codes(self.validate())
        self.assertIn("human_gold_without_version", codes)
        self.assertIn("human_gold_without_hash", codes)

    def test_tampered_artifact_is_detected(self):
        with (self.package / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        self.assertIn("hash_mismatch", _codes(self.validate()))

    def test_missing_artifact_is_detected(self):
        self.mutate(files={"predictions.jsonl": "0" * 64, "ghost.jsonl": "0" * 64})
        codes = _codes(self.validate())
        self.assertIn("missing_artifact", codes)

    def test_unsupported_or_mismatched_version_is_rejected(self):
        self.mutate(benchmark_version="9.9.9")
        self.assertIn("unsupported_benchmark_version", _codes(self.validate()))
        self.build()
        self.mutate(dimension_set_version="three-axis-v0")
        self.assertIn("dimension_set_mismatch", _codes(self.validate()))

    def test_unpinned_model_fails_closed(self):
        document = read_json(self.package / "submission.json")
        document["run"]["model"]["model_id"] = ""
        write_json(self.package / "submission.json", document)
        self.assertIn("unpinned_model", _codes(self.validate()))

    def test_fake_human_gold_is_rejected(self):
        document = read_json(self.package / "submission.json")
        document["dataset"]["evidence_grade"] = "human_gold"
        document["dataset"]["human_annotation_complete"] = False
        write_json(self.package / "submission.json", document)
        self.assertIn("fake_human_gold", _codes(self.validate()))

    def test_scores_declared_requires_primary_dimension(self):
        self.mutate(scores_declared=True, dimensions_reported=["B"])
        codes = _codes(self.validate())
        self.assertIn("primary_dimension_A_missing_from_ranked_report", codes)
        self.build(scores_declared=True, dimensions_reported=["A", "C", "D"])
        self.assertTrue(self.validate()["valid"])

    def test_declared_scores_without_verification_are_rejected(self):
        """A package that claims scores must say how they were obtained."""
        self.build(scores_declared=True, dimensions_reported=["A"])
        document = read_json(self.package / "submission.json")
        document.pop("scores_verification", None)
        write_json(self.package / "submission.json", document)
        self.assertIn("missing_scores_verification", _codes(self.validate()))

    def test_unverifiable_scores_must_say_so(self):
        """No dataset on this machine => the package has to declare it."""
        self.build(scores_declared=True, dimensions_reported=["A"])
        document = read_json(self.package / "submission.json")
        # The temp dataset has no candidate_truth.jsonl, so nothing can be
        # recomputed and the package must not pretend otherwise.
        self.assertEqual(document["scores_verification"], "not_reproducible_locally")
        self.assertEqual(document["scores"], {})
        # ... and because it declared that honestly, the validator accepts it.
        self.assertNotIn("scores_not_reproducible_locally", _codes(self.validate()))
        report = self.validate()
        self.assertTrue(report["valid"], report["issues"])
        self.assertFalse(report["scores_verified"])

    def test_silent_scores_are_rejected(self):
        """A package claiming scores but not declaring how is refused."""
        self.build(scores_declared=True, dimensions_reported=["A"])
        document = read_json(self.package / "submission.json")
        document.pop("scores_verification", None)
        write_json(self.package / "submission.json", document)
        self.assertIn("missing_scores_verification", _codes(self.validate()))

    def test_claimed_reproduction_without_scores_is_rejected(self):
        self.build(scores_declared=True, dimensions_reported=["A"])
        document = read_json(self.package / "submission.json")
        document["scores_verification"] = "reproduced_locally"
        document["scores"] = {}
        write_json(self.package / "submission.json", document)
        self.assertIn("scores_without_values", _codes(self.validate()))

    def test_unverifiable_scores_must_mention_the_limitation(self):
        """Saying 'not reproducible' is not enough; the limitations must say so."""
        self.build(scores_declared=True, dimensions_reported=["A"])
        document = read_json(self.package / "submission.json")
        document["declared_limitations"] = ["No human gold and no inter-annotator agreement."]
        write_json(self.package / "submission.json", document)
        self.assertIn("scores_limitation_missing", _codes(self.validate()))

    def test_denominator_must_add_up(self):
        document = read_json(self.package / "submission.json")
        document["run"]["denominator_disclosure"]["scored_n"] = 99
        write_json(self.package / "submission.json", document)
        self.assertIn("denominator_mismatch", _codes(self.validate()))

    def test_real_data_requires_compliance_note(self):
        document = read_json(self.package / "submission.json")
        document["dataset"]["source_type"] = "real_redacted"
        write_json(self.package / "submission.json", document)
        self.assertIn("missing_compliance_note", _codes(self.validate()))

    def test_repetition_index_changes_run_id(self):
        first = read_json(self.package / "submission.json")["run"]["run_id"]
        self.build(repetition_index=2)
        second = read_json(self.package / "submission.json")["run"]["run_id"]
        self.assertNotEqual(first, second)


class DatasetExtensionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "contrib-x"
        self.root.mkdir()

    def write_generated(self, tasks_text='{"dialogue_id":"x-001"}\n'):
        (self.root / "tasks.jsonl").write_text(tasks_text, encoding="utf-8")
        write_json(self.root / "manifest.json", {
            "dataset_version": "contrib-x-v1", "status": "generated_unannotated_not_gold",
            "seed": 999, "generation_method": "deterministic_template_v0",
            "human_annotation_complete": False, "gold_version": None, "files": {},
        })

    def test_extension_is_stamped_and_valid(self):
        self.write_generated()
        built = build_dataset_manifest(self.root, benchmark_config_path=BENCHMARK_CONFIG, contributor="tester")
        self.assertEqual(built["file_n"], 2)
        report = validate_submission(self.root, benchmark_config_path=BENCHMARK_CONFIG)
        self.assertTrue(report["valid"], report["issues"])
        document = read_json(self.root / "dataset.json")
        self.assertEqual(document["benchmark_version"], "0.4.0")
        self.assertEqual(document["split"], "public")

    def test_byte_identical_copy_of_frozen_pilot_is_refused(self):
        self.write_generated()
        shutil.copyfile(REPO_ROOT / "data/pilot_v0/tasks.jsonl", self.root / "tasks.jsonl")
        with self.assertRaises(ValueError):
            build_dataset_manifest(self.root, benchmark_config_path=BENCHMARK_CONFIG)

    def test_validator_also_rejects_declared_duplicate(self):
        self.write_generated()
        frozen = read_json(REPO_ROOT / "data/pilot_v0/manifest.json")["files"]["tasks.jsonl"]
        from context_benchmark.io import sha256_file
        write_json(self.root / "dataset.json", {
            "benchmark_version": "0.4.0", "dimension_set_version": "six-axis-v1",
            "dataset_version": "contrib-x-v1", "provenance": {"compliance_note": None},
            "files": {"tasks.jsonl": frozen},
        })
        self.assertIn("duplicate_of_frozen_pilot", _codes(
            validate_submission(self.root, benchmark_config_path=BENCHMARK_CONFIG)))


class VersionConsistencyTests(unittest.TestCase):
    def test_package_version_matches_benchmark_config(self):
        from context_benchmark import __version__
        self.assertEqual(read_json(BENCHMARK_CONFIG)["benchmark_version"], __version__)

    def test_pyproject_version_matches_package(self):
        import re
        from context_benchmark import __version__
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), __version__)

    def test_frozen_tasks_hash_is_stamped_in_config(self):
        config = read_json(BENCHMARK_CONFIG)
        stamped = config["dataset_policy"]["frozen_tasks_sha256"]
        actual = sha256_file(REPO_ROOT / "data/pilot_v0/tasks.jsonl")
        self.assertEqual(stamped, actual)

    def test_configured_dimension_set_matches_registry(self):
        from context_benchmark.dimensions import DIMENSION_SET_VERSION, attribution_ids, rankable_ids
        config = read_json(BENCHMARK_CONFIG)
        self.assertEqual(config["dimension_set_version"], DIMENSION_SET_VERSION)
        self.assertEqual(config["rankable_dimensions"], rankable_ids())
        self.assertEqual(config["attribution_dimensions"], attribution_ids())


if __name__ == "__main__":
    unittest.main()
