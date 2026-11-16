import unittest

from context_benchmark.dimensions import (
    DIMENSION_SET_VERSION, PRIMARY_DIMENSION, attribution_ids, dimension_ids,
    rankable_ids, registry, validate_dimension_set,
)


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
