import tempfile
import unittest
from pathlib import Path

from context_benchmark.io import read_json, read_jsonl, write_json
from context_benchmark.generator import generate_dataset


class GeneratorAcceptanceTests(unittest.TestCase):
    def test_coverage_contract(self):
        report = read_json("data/pilot_v0/coverage.json")
        self.assertTrue(report["accepted_for_annotation_start"])
        self.assertEqual(report["dialogues"], 40)
        self.assertEqual(report["candidate_units"], 420)
        self.assertGreaterEqual(report["dialogues_by_kind"]["empty"], 5)
        self.assertGreaterEqual(report["dialogues_by_kind"]["distractor"], 5)
        self.assertTrue(all(value >= 4 for value in report["rule_dialogue_coverage"].values()))

    def test_annotation_worksheet_has_no_answer_leakage(self):
        rows = read_jsonl("data/pilot_v0/annotation_worksheet.jsonl")
        self.assertEqual(len(rows), 40)
        for row in rows:
            self.assertNotIn("candidate_truth", row)
            self.assertNotIn("distractors", row)
            self.assertNotIn("probes", row)
            self.assertEqual(row["annotations"], [])

    def test_unrelated_probes_are_opt_in_and_leave_pilot_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = read_json("configs/generator.json")
            config["dataset_version"] = "p0-extension-unrelated-probe"
            config["seed"] = config["seed"] + 1
            config["add_unrelated_probes"] = True
            config_path = root / "generator.json"
            write_json(config_path, config)
            output = root / "ext"
            generate_dataset(str(config_path), output_dir=output, contributor="test", split="public")
            added = [
                probe
                for row in read_jsonl(output / "candidate_truth.jsonl")
                for probe in row["probes"]
                if probe["probe_id"] == "p003"
            ]
            self.assertGreater(len(added), 0)
            self.assertTrue(all(probe["related"] is False for probe in added))
            frozen = [
                probe
                for row in read_jsonl("data/pilot_v0/candidate_truth.jsonl")
                for probe in row["probes"]
                if probe["probe_id"] == "p003"
            ]
            self.assertEqual(frozen, [])

    def test_manifest_does_not_claim_gold(self):
        manifest = read_json("data/pilot_v0/manifest.json")
        self.assertIsNone(manifest["gold_version"])
        self.assertFalse(manifest["human_annotation_complete"])
        self.assertEqual(manifest["status"], "generated_unannotated_not_gold")

    def test_lexical_variants_defeat_single_template_answer_key(self):
        """Memorising one dialogue's gold must not score full marks on the rest."""
        from context_benchmark.scoring import score_dataset
        from context_benchmark.validation import public_constraint
        dialogues = read_jsonl("data/pilot_v0/candidate_truth.jsonl")
        standard = [d for d in dialogues if d["dialogue_kind"] == "standard"]
        # Build a fixed "answer key" from the first standard dialogue.
        key = [public_constraint(item) for item in standard[0]["candidate_truth"]]
        predictions = [{"dialogue_id": d["dialogue_id"], "output": {"constraints": key}} for d in standard]
        result = score_dataset(standard, predictions)["aggregate"]
        self.assertLess(
            result["must_include"]["value"], 0.7,
            "a single memorised answer key still scores too high; variants are not diverse enough",
        )

    def test_key_slots_have_multiple_surface_forms(self):
        rows = read_jsonl("data/pilot_v0/candidate_truth.jsonl")
        standard = [r for r in rows if r["dialogue_kind"] == "standard"]
        for cid in ("c001", "c004", "c005"):
            texts = {u["text"] for r in standard for u in r["candidate_truth"] if u["id"] == cid}
            self.assertGreaterEqual(len(texts), 2, f"{cid} has only one surface form across dialogues")


if __name__ == "__main__":
    unittest.main()
