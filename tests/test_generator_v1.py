"""Acceptance tests for the randomised v1 candidate set.

The v1 set exists because the frozen pilot set could be answered from its
structure instead of from the dialogue: one slot template for all 40
dialogues, a fixed slot-to-turn mapping, empty dialogues that were shorter
than everything else, and a compliant option that was always first.

These tests pin the properties that make the set un-memorisable, and they
pin the version lock: the frozen pilot set must still regenerate
byte-for-byte.
"""

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from context_benchmark.audit import audit_dataset
from context_benchmark.generator import bigram_overlap, generate_dataset
from context_benchmark.io import read_json, read_jsonl, sha256_file, write_json
from context_benchmark.manifest import validate_submission
from context_benchmark.validation import validate_candidate_dialogue

FROZEN_SHA = "111bbc622ee8e38b451fd2e892ac68c92f5c7adc55d231e6ddc8031824e5a7a0"
V1_CONFIG = "configs/generator.v1.json"
V1_DIR = "data/contrib/pilot-v1-randomized"
# Slots whose surface form must not be memorisable from a single sentence.
KEY_SLOTS = ("c001", "c002", "c003", "c004", "c005", "c006", "c007", "c008", "c009", "c010", "c011", "c012")


def _dialogue_text(dialogue):
    return "\n".join(turn["text"] for turn in dialogue["turns"])


class FrozenPilotVersionLockTests(unittest.TestCase):
    """The frozen pilot set is a version lock; the v1 work must not move it."""

    def test_frozen_config_regenerates_the_frozen_bytes(self):
        config = read_json("configs/generator.json")
        with tempfile.TemporaryDirectory() as tmp:
            config["output_dir"] = (Path(tmp) / "pilot_v0").as_posix()
            config_path = Path(tmp) / "generator.json"
            write_json(config_path, config)
            generate_dataset(str(config_path))
            regenerated = sha256_file(Path(tmp) / "pilot_v0" / "candidate_truth.jsonl")
        self.assertEqual(regenerated, FROZEN_SHA)
        self.assertEqual(sha256_file("data/pilot_v0/candidate_truth.jsonl"), FROZEN_SHA)

    def test_frozen_pilot_tree_has_no_extra_files(self):
        names = sorted(path.name for path in Path("data/pilot_v0").iterdir())
        self.assertNotIn("dataset.json", names)

    def test_frozen_config_does_not_opt_into_the_new_behaviour(self):
        config = read_json("configs/generator.json")
        self.assertFalse(config.get("randomize_layout", False))
        self.assertNotIn("length_tier", config)


class RandomizedV1DatasetTests(unittest.TestCase):
    """Properties of the generated pilot-v1-randomized set."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls._tmp.name) / "pilot-v1-randomized"
        generate_dataset(V1_CONFIG, output_dir=cls.output, contributor="tests", split="public")
        cls.rows = read_jsonl(cls.output / "candidate_truth.jsonl")
        cls.standard = [row for row in cls.rows if row["dialogue_kind"] != "empty"]
        cls.empty = [row for row in cls.rows if row["dialogue_kind"] == "empty"]
        cls.by_id = {row["dialogue_id"]: row for row in cls.rows}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_config_declares_the_new_dataset_version(self):
        config = read_json(V1_CONFIG)
        self.assertEqual(config["dataset_version"], "pilot-v1-randomized")
        self.assertTrue(config["randomize_layout"])
        self.assertTrue(config["add_unrelated_probes"])
        self.assertEqual(config["length_tier"], "medium")

    def test_committed_artifact_matches_a_fresh_generation(self):
        """The checked-in data set is the one the current code produces."""
        self.assertEqual(
            sha256_file(self.output / "candidate_truth.jsonl"),
            sha256_file(Path(V1_DIR) / "candidate_truth.jsonl"),
        )

    def test_same_seed_regenerates_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            second = Path(tmp) / "pilot-v1-randomized"
            generate_dataset(V1_CONFIG, output_dir=second, contributor="tests", split="public")
            self.assertEqual(
                sha256_file(self.output / "candidate_truth.jsonl"),
                sha256_file(second / "candidate_truth.jsonl"),
            )

    def test_slot_layouts_are_not_shared_across_dialogues(self):
        """A slot's label must not be readable from the turn number."""
        layouts = [tuple(sorted(row["slot_layout"].items())) for row in self.standard]
        counts = Counter(layouts)
        self.assertGreaterEqual(
            len(counts), len(layouts) / 2,
            "more than half the dialogues share one slot-to-turn mapping",
        )
        self.assertGreater(len(counts), 1)

    def test_withdrawn_rule_always_precedes_its_replacement(self):
        for row in self.standard:
            truth = {item["id"]: item for item in row["candidate_truth"]}
            if truth["c008"].get("superseded_by"):
                self.assertLess(truth["c008"]["valid_from"], truth["c009"]["valid_from"])
            if truth["c004"]["type"] == "superseded":
                self.assertLess(truth["c004"]["valid_from"], truth["c004"]["valid_to"])
                self.assertEqual(truth["c004"]["valid_to"] + 1, truth["c004"]["evidence_turns"][1])

    def test_every_key_slot_has_at_least_five_surface_forms(self):
        surfaces = defaultdict(set)
        for row in self.standard:
            for item in row["candidate_truth"]:
                surfaces[item["id"]].add(item["text"])
        for slot in KEY_SLOTS:
            self.assertGreaterEqual(
                len(surfaces[slot]), 5,
                f"{slot} has only {len(surfaces[slot])} distinct surface forms",
            )

    def test_each_scenario_exercises_its_own_surface_forms(self):
        """A scene-specific slot (allergy, accommodation, ...) needs carriers.

        If the empty dialogues could take a scenario's slots, that scenario
        would end up with too few non-empty dialogues to exercise its own
        wording, and the "at least five surface forms" contract would hold
        only in the aggregate.
        """
        by_scenario = defaultdict(list)
        for row in self.standard:
            by_scenario[row["scenario"]].append(row)
        self.assertGreaterEqual(len(by_scenario), 5)
        for scenario, rows in by_scenario.items():
            self.assertGreaterEqual(len(rows), 5, f"{scenario} has too few non-empty dialogues")
            for slot in ("c010", "c011", "c012"):
                forms = {
                    item["text"]
                    for row in rows
                    for item in row["candidate_truth"]
                    if item["id"] == slot
                }
                self.assertGreaterEqual(len(forms), 5, f"{scenario}/{slot} is under-covered")

    def test_compliant_option_is_not_always_first(self):
        """'Always answer b1' must not be a winning strategy."""
        first_is_compliant = Counter()
        for row in self.standard:
            for probe in row["probes"]:
                first_is_compliant[probe["behaviors"][0]["is_compliant"]] += 1
        self.assertIn(True, first_is_compliant)
        self.assertIn(False, first_is_compliant)
        self.assertGreater(min(first_is_compliant.values()), 0)

    def test_unrelated_control_probes_exist(self):
        unrelated = [
            probe for row in self.rows for probe in row["probes"] if probe.get("related") is False
        ]
        self.assertGreater(len(unrelated), 0)
        self.assertEqual(len(unrelated), len(self.standard))
        for probe in unrelated:
            self.assertEqual(len(probe["behaviors"]), 2)
            self.assertIn("follows_injection", probe)

    def test_probe_queries_stay_below_the_lexical_overlap_ceiling(self):
        ceiling = read_json(V1_CONFIG)["max_probe_bigram_overlap"]
        for row in self.rows:
            truth = {item["id"]: item for item in row["candidate_truth"]}
            turns = {turn["turn"]: turn for turn in row["turns"]}
            for probe in row["probes"]:
                target = truth[probe["target_constraint_id"]]
                overlap = bigram_overlap(probe["query"], target["text"])
                self.assertLess(
                    overlap, ceiling,
                    f"{row['dialogue_id']}:{probe['probe_id']} overlaps its target constraint",
                )
                for turn in target["evidence_turns"]:
                    if turns[turn]["speaker"] == "user":
                        self.assertLess(bigram_overlap(probe["query"], turns[turn]["text"]), ceiling)
                # The recorded value is the strictest reading: the larger of
                # the overlap with the normalised truth and with the wording
                # the constraint was read from.
                self.assertLess(probe["lexical_overlap"], ceiling)
                self.assertGreaterEqual(probe["lexical_overlap"] + 1e-6, overlap)

    def test_probe_ids_are_stable(self):
        for row in self.rows:
            ids = [probe["probe_id"] for probe in row["probes"]]
            self.assertEqual(ids, sorted(ids))
            self.assertLessEqual(len(set(ids)), 3)

    def test_no_two_dialogues_are_identical(self):
        texts = [_dialogue_text(row) for row in self.rows]
        self.assertEqual(len(set(texts)), len(texts))
        self.assertGreater(len(set(texts)), 1)

    def test_empty_and_non_empty_turn_distributions_overlap(self):
        """An empty answer must not be decidable from the length."""
        standard_turns = [len(row["turns"]) for row in self.standard]
        empty_turns = [len(row["turns"]) for row in self.empty]
        self.assertTrue(empty_turns)
        self.assertGreater(len(set(empty_turns)), 1, "the empty dialogues are all the same length")
        self.assertGreaterEqual(min(empty_turns), min(standard_turns))
        self.assertLessEqual(max(empty_turns), max(standard_turns))
        self.assertTrue(set(empty_turns) & set(standard_turns))
        spread = max(standard_turns) - min(standard_turns)
        self.assertLess(
            abs(sum(empty_turns) / len(empty_turns) - sum(standard_turns) / len(standard_turns)),
            0.2 * spread + 1,
        )

    def test_empty_dialogues_carry_no_constraint_like_user_turn(self):
        cues = ("不要", "别", "必须", "不能", "不准", "以后", "每次", "任何", "都要", "只能用")
        for row in self.empty:
            for turn in row["turns"]:
                if turn["speaker"] != "user":
                    continue
                for cue in cues:
                    self.assertNotIn(cue, turn["text"], f"{row['dialogue_id']} turn {turn['turn']}")

    def test_length_profile_is_recorded(self):
        for row in self.rows:
            profile = row["length_profile"]
            self.assertEqual(profile["turns"], len(row["turns"]))
            self.assertEqual(profile["length_tier"], "medium")
            self.assertGreater(profile["estimated_chars"], 0)

    def test_dialogues_pass_the_shape_validator(self):
        for row in self.rows:
            report = validate_candidate_dialogue(row)
            self.assertTrue(report.valid, f"{row['dialogue_id']}: {report.to_dict()}")

    def test_rule_audit_finds_nothing(self):
        report = audit_dataset(self.rows)
        self.assertEqual(report["severity_counts"], {})
        self.assertEqual(report["code_counts"], {})

    def test_dataset_extension_validates(self):
        report = validate_submission(self.output)
        self.assertTrue(report["valid"], report)


if __name__ == "__main__":
    unittest.main()
