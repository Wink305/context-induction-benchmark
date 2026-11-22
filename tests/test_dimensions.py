import unittest
from pathlib import Path

from context_benchmark.dimensions import (
    DIMENSION_SET_VERSION, IMPLEMENTED_METRICS, PRIMARY_DIMENSION, attribution_ids,
    dimension_ids, implemented_metric_names, rankable_ids, registry,
    validate_dimension_set,
)


class MetricImplementationTests(unittest.TestCase):
    """A registered metric must have an implementation behind it.

    The registry used to list ``hit_rate``, ``FAR``, ``live_constraint_accuracy``,
    ``hierarchy_S_U_T``, ``task_completion`` and ``quality_decay`` as if they
    were measured. No scorer ever produced them, so a reader of the registry
    would believe six metrics exist that do not.
    """

    def test_registered_metrics_all_exist_in_a_scorer(self):
        implemented = implemented_metric_names()
        for item in registry()["dimensions"]:
            for metric in item["metrics"]:
                self.assertIn(metric, implemented, f"dimension {item['id']} registers {metric}")

    def test_every_implemented_metric_name_appears_in_its_module(self):
        root = Path(__file__).resolve().parents[1]
        for relative_path, names in IMPLEMENTED_METRICS.items():
            source = (root / relative_path).read_text(encoding="utf-8")
            for name in names:
                for part in name.split("."):
                    self.assertIn(part, source, f"{relative_path} does not define {name}")

    def test_promised_metrics_are_not_claimed_as_implemented(self):
        implemented = implemented_metric_names()
        for item in registry()["dimensions"]:
            for metric in item.get("promised_metrics", []):
                self.assertNotIn(metric, implemented, f"dimension {item['id']} promises {metric}")

    def test_metrics_state_agrees_with_promised_metrics(self):
        for item in registry()["dimensions"]:
            promised = item.get("promised_metrics", [])
            expected = "partial" if promised else "implemented"
            self.assertEqual(item["metrics_state"], expected, item["id"])

    def test_the_known_dangling_names_are_promised_not_implemented(self):
        registry_by_id = {item["id"]: item for item in registry()["dimensions"]}
        self.assertEqual(registry_by_id["B"]["promised_metrics"], ["hit_rate", "FAR"])
        self.assertEqual(
            registry_by_id["C"]["promised_metrics"],
            ["live_constraint_accuracy", "hierarchy_S_U_T"],
        )
        self.assertEqual(registry_by_id["F"]["promised_metrics"], ["quality_decay"])
        for axis_id in ("A", "D", "E"):
            self.assertEqual(registry_by_id[axis_id]["promised_metrics"], [], axis_id)


class DimensionRegistryTests(unittest.TestCase):
    def test_six_axes_with_one_primary(self):
        self.assertEqual(dimension_ids(), ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(len(dimension_ids()), 6)
        self.assertEqual(PRIMARY_DIMENSION, "A")
        report = registry()
        self.assertEqual(report["dimension_n"], 6)
        self.assertTrue(all(item["role"] == "primary" for item in report["dimensions"] if item["id"] == "A"))
        self.assertEqual(sum(1 for item in report["dimensions"] if item["role"] == "primary"), 1)
        self.assertFalse(report["weighted_total"])
        self.assertEqual(report["dimension_set_version"], DIMENSION_SET_VERSION)

    def test_ranking_and_attribution_are_disjoint(self):
        self.assertEqual(rankable_ids(), ["A", "B", "C", "D"])
        self.assertEqual(attribution_ids(), ["E", "F"])
        self.assertFalse(set(rankable_ids()) & set(attribution_ids()))

    def test_every_rankable_axis_needs_an_orthogonality_check(self):
        for item in registry()["dimensions"]:
            if item["rankable"]:
                self.assertTrue(item["orthogonality_required"], item["id"])
                self.assertTrue(item["metrics"], item["id"])

    def test_validation_rejects_unknown_duplicate_and_missing_primary(self):
        self.assertEqual(validate_dimension_set(["A", "B", "C"]), [])
        self.assertIn("unknown_dimension:Z", validate_dimension_set(["Z"]))
        self.assertIn("duplicate_dimension_id", validate_dimension_set(["A", "A"]))
        issues = validate_dimension_set(["B"], scores_declared=True)
        self.assertIn("primary_dimension_A_missing_from_ranked_report", issues)
        self.assertIn("scores_declared_but_no_dimension", validate_dimension_set([], scores_declared=True))
        self.assertIn("no_rankable_dimension_declared",
                      validate_dimension_set(["E", "F"], scores_declared=True))
        self.assertEqual(validate_dimension_set(["A", "E"], scores_declared=True), [])

    def test_attribution_axes_can_be_reported_but_not_only_ranked(self):
        self.assertEqual(validate_dimension_set(["E", "F"]), [])


if __name__ == "__main__":
    unittest.main()
