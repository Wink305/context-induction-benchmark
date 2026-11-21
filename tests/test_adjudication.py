import copy
import tempfile
import unittest
from pathlib import Path

from context_benchmark.adjudication import freeze_gold
from context_benchmark.agreement import agreement_report
from context_benchmark.io import read_json, read_jsonl
from context_benchmark.validation import public_constraint


def human_units(units):
    """Turn generator units into what a human annotator would have written.

    The content is kept — the machinery under test is the freeze, not the
    labels — but every generator marker is removed: ``candidate_only`` is
    dropped, and the generator's notes (「候选 I，待人工确认」) are replaced
    by a plain human note.  The result satisfies the model-output contract,
    exactly like a sheet parsed by ``parse_sheets``.
    """
    out = []
    for unit in units:
        item = public_constraint(unit)
        item["note"] = "human annotation"
        # The boundary tags are the annotator's own judgement and travel
        # with the unit into the agreement gate.
        item["rule_tags"] = list(unit.get("rule_tags") or [])
        out.append(item)
    return out


class HumanAnnotationFixture:
    """Two independent annotators who agreed on every unit.

    Used to exercise the freeze machinery end to end.  The dialogue source
    is the **truth-free worksheet**, never ``candidate_truth.jsonl``: the
    freeze must refuse generator output.
    """

    @classmethod
    def setUpClass(cls):
        cls.dialogues = read_jsonl("data/pilot_v0/annotation_worksheet.jsonl")
        cls.truth = read_jsonl("data/pilot_v0/candidate_truth.jsonl")

    def annotations(self, annotator="A"):
        return [
            {
                "dialogue_id": row["dialogue_id"],
                "annotator": annotator,
                "status": "annotated",
                "provenance": "human",
                "annotations": human_units(row["candidate_truth"]),
            }
            for row in self.truth
        ]

    def agreement(self, rows_a=None, rows_b=None):
        rows_a = rows_a if rows_a is not None else self.annotations("A")
        rows_b = rows_b if rows_b is not None else self.annotations("B")
        return agreement_report(rows_a, rows_b, bootstrap_n=40)


class AdjudicationTests(HumanAnnotationFixture, unittest.TestCase):
    def test_freeze_requires_real_adjudication_inputs_and_is_immutable(self):
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertEqual(manifest["gold_version"], "v1.0")
            self.assertEqual(manifest["gold_provenance"], "human")
            self.assertEqual(manifest["annotator_n"], 2)
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            self.assertEqual(len(gold), 40)
            self.assertTrue(all(row["status"] == "human_adjudicated_gold" for row in gold))
            self.assertTrue(all(row["gold_provenance"] == "human" for row in gold))
            with self.assertRaises(FileExistsError):
                freeze_gold(self.dialogues, rows, agreement, [], directory)

    def test_frozen_gold_is_readable_by_the_scorer(self):
        """The frozen file must be scoreable by the benchmark's own scorer."""
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            freeze_gold(self.dialogues, rows, agreement, [], directory)
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            # scoring.py reads dialogue["candidate_truth"]; without it the
            # frozen gold could not be scored.
            self.assertTrue(all(isinstance(row.get("candidate_truth"), list) for row in gold))
            self.assertTrue(all(row["candidate_truth"] == row["constraints"] for row in gold))

    def test_pending_conflict_without_resolution_cannot_freeze(self):
        rows_a = self.annotations("A")
        rows_b = self.annotations("B")
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        agreement = self.agreement(rows_a, rows_b)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_gold(self.dialogues, rows_a, agreement, [], directory)

    def test_resolution_decision_b_is_applied(self):
        """A resolution saying 'b' must put B's unit into the frozen gold."""
        rows_a = self.annotations("A")
        rows_b = self.annotations("B")
        target_index = 0
        rows_b[0]["annotations"][target_index]["type"] = "one_off"
        rows_b[0]["annotations"][target_index]["scope"] = "task"
        agreement = self.agreement(rows_a, rows_b)
        conflict = next(
            c for c in agreement["adjudication_queue"]
            if c.get("a_index") == target_index and "type_mismatch" in c["reasons"]
        )
        resolution = {
            "conflict_id": conflict["conflict_id"], "decision": "b",
            "reviewer": "third", "rationale": "B's scope is correct",
        }
        with tempfile.TemporaryDirectory() as directory:
            freeze_gold(
                self.dialogues, rows_a, agreement,
                [resolution], directory,
            )
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            frozen_unit = next(
                u for u in gold[0]["constraints"]
                if u["type"] == "one_off" and u["scope"] == "task"
            )
            self.assertEqual(frozen_unit["text"], rows_b[0]["annotations"][target_index]["text"])

    def test_resolution_exclude_removes_unit(self):
        rows_a = self.annotations("A")
        rows_b = self.annotations("B")
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        agreement = self.agreement(rows_a, rows_b)
        conflict = next(
            c for c in agreement["adjudication_queue"]
            if c.get("a_index") == 0 and "type_mismatch" in c["reasons"]
        )
        resolution = {
            "conflict_id": conflict["conflict_id"], "decision": "exclude",
            "reviewer": "third", "rationale": "unit is not a real constraint",
        }
        with tempfile.TemporaryDirectory() as directory:
            freeze_gold(
                self.dialogues, rows_a, agreement,
                [resolution], directory,
            )
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            removed_text = conflict["annotator_a"]["text"]
            self.assertNotIn(removed_text, [u["text"] for u in gold[0]["constraints"]])


class FreezeProvenanceTests(HumanAnnotationFixture, unittest.TestCase):
    """Gold may only be frozen when it can prove it came from people."""

    def test_freeze_records_the_declared_provenance(self):
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertEqual(manifest["gold_provenance"], "human")
            gold = read_jsonl(Path(directory) / "gold_v1.jsonl")
            self.assertTrue(all(row["gold_provenance"] == "human" for row in gold))

    def test_generator_truth_cannot_be_frozen_as_human_gold(self):
        """The shipped generator truth must never back a human gold claim."""
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        generator_truth = read_jsonl("data/pilot_v0/candidate_truth.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(generator_truth, rows, agreement, [], directory)
            message = str(caught.exception)
            self.assertIn("generator provenance", message)
            # The dialogue-level markers must be named: the truth container
            # and the generator status.
            self.assertIn("candidate_truth", message)
            self.assertIn("generated_unannotated_not_gold", message)
            self.assertFalse((Path(directory) / "gold_v1.jsonl").exists())

    def test_generator_truth_is_refused_even_when_its_unit_markers_are_stripped(self):
        """Stripping the per-unit markers must not let generator truth through.

        The dialogue-level markers are independent of the unit-level ones:
        the truth container and the generator status survive a strip of
        ``candidate_only``.
        """
        generator_truth = read_jsonl("data/pilot_v0/candidate_truth.jsonl")
        stripped = []
        for row in generator_truth:
            dialogue = dict(row)
            dialogue["status"] = "annotated"
            for unit in dialogue.get("candidate_truth", []):
                unit.pop("candidate_only", None)
                unit["note"] = "human annotation"
            stripped.append(dialogue)
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(stripped, rows, agreement, [], directory)
            self.assertIn("generator provenance", str(caught.exception))
            self.assertFalse((Path(directory) / "gold_v1.jsonl").exists())

    def test_generator_units_inside_the_annotations_are_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            for unit in row["annotations"]:
                unit["candidate_only"] = True
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("generator provenance", str(caught.exception))

    def test_a_generator_note_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            for unit in row["annotations"]:
                unit["note"] = "三次独立交互模式；候选 I，待人工确认"
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("generator provenance", str(caught.exception))

    def test_one_annotator_cannot_certify_agreement(self):
        """The same person labelling twice is not an agreement measurement."""
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("A"))
        self.assertEqual(agreement["annotator_n"], 1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("independent annotators", str(caught.exception))

    def test_unknown_annotator_origin_is_not_promoted_to_human(self):
        rows = self.annotations("A")
        for row in rows:
            row.pop("annotator", None)
        agreement = self.agreement(rows, self.annotations("B"))
        agreement["annotator_type"] = "unknown"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("annotator_type", str(caught.exception))

    def test_boundary_gate_is_not_assumed_to_have_passed(self):
        """No boundary tags means the R1-R8 gate was never exercised."""
        rows = self.annotations("A")
        for row in rows:
            for unit in row["annotations"]:
                unit["rule_tags"] = []
        rows_b = self.annotations("B")
        for row in rows_b:
            for unit in row["annotations"]:
                unit["rule_tags"] = []
        agreement = self.agreement(rows, rows_b)
        self.assertFalse(agreement["boundary_assessed"])
        self.assertFalse(agreement["exit_checks"]["boundary_gte_0_75"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("boundary_gte_0_75", str(caught.exception))

    def test_an_undeclared_origin_is_not_stamped_as_human(self):
        """A payload with no provenance claim must not be frozen as human gold."""
        rows = self.annotations("A")
        for row in rows:
            row.pop("provenance", None)
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("does not declare a human origin", str(caught.exception))
            self.assertFalse((Path(directory) / "gold_v1.jsonl").exists())

    def test_an_annotator_cannot_review_their_own_conflicts(self):
        """The third reviewer must not be one of the two annotators."""
        rows_a = self.annotations("A")
        rows_b = self.annotations("B")
        rows_b[0]["annotations"][0]["type"] = "one_off"
        rows_b[0]["annotations"][0]["scope"] = "task"
        agreement = self.agreement(rows_a, rows_b)
        conflict = next(
            c for c in agreement["adjudication_queue"]
            if c.get("a_index") == 0 and "type_mismatch" in c["reasons"]
        )
        resolution = {
            "conflict_id": conflict["conflict_id"], "decision": "a",
            "reviewer": "A", "rationale": "my own annotation is correct",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows_a, agreement, [resolution], directory)
            self.assertIn("third reviewer", str(caught.exception))

    def test_non_human_gold_provenance_is_rejected(self):
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(
                    self.dialogues, rows, agreement, [], directory,
                    gold_provenance="ai_precheck",
                )
            self.assertIn("gold_provenance", str(caught.exception))
            self.assertFalse((Path(directory) / "gold_v1.jsonl").exists())

    def test_ai_annotator_in_the_payload_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            row["annotator"] = "AI-precheck"
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("non-human provenance", str(caught.exception))

    def test_ai_note_inside_an_annotation_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            for unit in row["annotations"]:
                unit["note"] = "AI pre-check; not a human annotation"
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("note", str(caught.exception))

    def test_chinese_ai_note_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            for unit in row["annotations"]:
                unit["note"] = "由模型生成"
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_gold(self.dialogues, rows, agreement, [], directory)

    def test_incomplete_human_annotation_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            row["human_annotation_complete"] = False
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("human_annotation_complete", str(caught.exception))

    def test_non_human_evidence_grade_is_rejected(self):
        rows = self.annotations("A")
        for row in rows:
            row["evidence_grade"] = "ai_precheck_only"
        agreement = self.agreement(rows, self.annotations("B"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [], directory)
            self.assertIn("evidence_grade", str(caught.exception))

    def test_ai_annotator_in_the_agreement_report_is_rejected(self):
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        agreement["annotator_type"] = "ai_precheck"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_gold(self.dialogues, rows, agreement, [], directory)

    def test_ai_reviewer_in_the_resolutions_is_rejected(self):
        rows = self.annotations("A")
        agreement = self.agreement(rows, self.annotations("B"))
        resolution = {
            "conflict_id": "adj-0001", "decision": "a",
            "reviewer": "model", "rationale": "because",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                freeze_gold(self.dialogues, rows, agreement, [resolution], directory)
            self.assertIn("reviewer", str(caught.exception))

    @unittest.skipUnless(
        Path("data/annotation/precheck_ai_annotations.jsonl").is_file(),
        "the AI pre-check artefacts are local working artifacts and are not published",
    )
    def test_repository_ai_precheck_artifact_cannot_be_frozen(self):
        """The shipped AI pre-check must never be freezable as human gold."""
        rows = []
        for row in read_jsonl("data/annotation/precheck_ai_annotations.jsonl"):
            rows.append({
                "dialogue_id": row["dialogue_id"],
                "annotations": copy.deepcopy(row.get("constraints", [])),
                "annotator": row.get("annotator"),
                "provenance": row.get("provenance"),
                "human_annotation_complete": row.get("human_annotation_complete"),
            })
        self.assertTrue(any(row["annotator"] == "AI-precheck" for row in rows))
        self.assertTrue(any(row["provenance"] == "ai_precheck" for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_gold(self.dialogues, rows, self.agreement(), [], directory)


if __name__ == "__main__":
    unittest.main()
