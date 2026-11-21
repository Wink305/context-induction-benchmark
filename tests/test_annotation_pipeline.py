"""End-to-end acceptance test for the human annotation chain.

The chain is: render answer-free sheets -> a human fills the answer block ->
parse the sheets -> measure agreement -> adjudicate the conflicts -> freeze
gold -> read the frozen gold back with the benchmark's own scorer.

Every earlier version of this pipeline was broken at some link: the parsed
rows carried the units under a key nobody downstream read, the parsed units
had no ``text`` field, the boundary tags never reached the agreement gate,
the frozen file used a key the scorer does not read, and the freeze accepted
generator truth as if it were human gold.  This module exercises the whole
path so those breaks cannot come back quietly.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from context_benchmark.adjudication import freeze_gold
from context_benchmark.agreement import agreement_report
from context_benchmark.io import read_jsonl
from context_benchmark.scoring import score_dataset
from context_benchmark.sheets import ANSWER_END, ANSWER_START, make_sheets, parse_sheets

TYPE_CODES = {"standing": "S", "one_off": "O", "implicit": "I", "superseded": "T"}
CATEGORY_CODES = {"communication": "C", "process": "P", "domain": "D"}

WORKSHEET = "data/pilot_v0/annotation_worksheet.jsonl"
TRUTH = "data/pilot_v0/candidate_truth.jsonl"


def sheet_rows(units: list[dict]) -> str:
    """Write a human's answers back in the sheet's own row format.

    The annotation is re-rendered from the truth only so the test has
    something to fill in; the point of the test is the path, not the labels.
    """
    rows = []
    for unit in units:
        turns = ",".join(str(turn) for turn in unit["evidence_turns"])
        note = ""
        if unit["type"] == "superseded":
            note = (
                f"被 {unit['valid_to'] + 1} 轮替代"
                if unit.get("superseded_by")
                else str(unit["valid_to"] + 1)
            )
        tags = ",".join(unit.get("rule_tags") or [])
        rows.append(
            f"{turns} | {TYPE_CODES[unit['type']]} | {unit['scope']} | "
            f"{CATEGORY_CODES[unit['category']]} | {unit['text']} | {note} | {tags}"
        )
    return "\n".join(rows)


class AnnotationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = read_jsonl(TRUTH)

    def _fill(self, root: Path, annotator: str, drift: bool = False) -> Path:
        """Render the sheets for one annotator and fill in the answer blocks."""
        out = root / f"sheets_{annotator}"
        make_sheets(WORKSHEET, out, annotator=annotator)
        for row in self.truth:
            units = row["candidate_truth"]
            if drift and units:
                units = copy.deepcopy(units)
                # One systematic disagreement: the first unit read as a
                # one-off request scoped to this task.
                units[0]["type"] = "one_off"
                units[0]["scope"] = "task"
                units[0]["rule_tags"] = []
            path = out / f"{row['dialogue_id']}.md"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(ANSWER_START, ANSWER_START + "\n" + sheet_rows(units) + "\n"),
                encoding="utf-8",
            )
        return out

    def test_the_whole_chain_runs_and_the_frozen_gold_is_scoreable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sheets_a = self._fill(root, "A")
            sheets_b = self._fill(root, "B", drift=True)

            parsed_a = parse_sheets(sheets_a, annotator="A")
            parsed_b = parse_sheets(sheets_b, annotator="B")

            # The parse must be clean and must actually carry units.
            self.assertEqual(parsed_a["problems"], [])
            self.assertEqual(parsed_b["problems"], [])
            units = [unit for row in parsed_a["annotations"] for unit in row["annotations"]]
            self.assertEqual(len(units), 420)
            self.assertTrue(all(unit["id"].startswith("c") for unit in units))
            self.assertTrue(all(unit["text"] for unit in units))
            self.assertTrue(any(unit["rule_tags"] for unit in units))

            report = agreement_report(
                parsed_a["annotations"], parsed_b["annotations"], bootstrap_n=200
            )
            # Two independent annotators, and a real disagreement between them.
            self.assertEqual(report["annotator_n"], 2)
            self.assertEqual(report["annotator_type"], "human")
            self.assertLess(report["overall"]["kappa"], 1.0)
            self.assertGreater(report["overall"]["kappa"], 0.70)
            self.assertTrue(report["boundary_assessed"])
            self.assertGreater(report["boundary_overall"]["value"], 0.75)
            self.assertTrue(report["adjudication_queue"])

            resolutions = [
                {
                    "conflict_id": conflict["conflict_id"],
                    "decision": "b",
                    "reviewer": "third reviewer",
                    "rationale": "B's reading follows the manual",
                }
                for conflict in report["adjudication_queue"]
            ]
            manifest = freeze_gold(
                read_jsonl(WORKSHEET),
                parsed_a["annotations"],
                report,
                resolutions,
                root / "gold",
            )
            self.assertEqual(manifest["gold_provenance"], "human")
            self.assertEqual(manifest["annotator_n"], 2)
            self.assertTrue(manifest["boundary_assessed"])

            gold = read_jsonl(root / "gold" / "gold_v1.jsonl")
            self.assertEqual(len(gold), 40)
            self.assertEqual(sum(len(row["candidate_truth"]) for row in gold), 420)
            # The frozen file must be readable by the benchmark's own scorer,
            # which reads dialogue["candidate_truth"].
            scored = score_dataset(
                gold,
                [
                    {"dialogue_id": row["dialogue_id"], "output": {"constraints": row["candidate_truth"]}}
                    for row in gold
                ],
            )
            self.assertEqual(scored["aggregate"]["fama_macro"]["value"], 0.875)
            self.assertEqual(scored["aggregate"]["must_include"]["value"], 1.0)

    def test_the_sheets_handed_to_annotators_carry_no_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sheets"
            make_sheets(WORKSHEET, out)
            for path in sorted(out.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                answer = text.split(ANSWER_START, 1)[1].split(ANSWER_END, 1)[0]
                self.assertEqual(answer.strip(), "", f"{path.name} already contains answers")

    def test_a_single_annotator_cannot_freeze_gold(self):
        """One person labelling twice is not an agreement measurement."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sheets_a = self._fill(root, "A")
            parsed_a = parse_sheets(sheets_a, annotator="A")
            report = agreement_report(
                parsed_a["annotations"], copy.deepcopy(parsed_a["annotations"]), bootstrap_n=50
            )
            self.assertEqual(report["annotator_n"], 1)
            with self.assertRaises(ValueError) as caught:
                freeze_gold(
                    read_jsonl(WORKSHEET), parsed_a["annotations"], report, [], root / "gold"
                )
            self.assertIn("independent annotators", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
