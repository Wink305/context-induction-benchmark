import unittest

from context_benchmark.survival import kaplan_meier, survival_report


class SurvivalTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"dialogue_id": "a", "compliance": [True, True, False, True]},
            {"dialogue_id": "b", "compliance": [True, True, True, True]},
        ]

    def test_kaplan_meier_with_right_censoring(self):
        result = kaplan_meier(self.records)
        self.assertEqual(result["median_survival_turn"], 3)
        self.assertAlmostEqual(result["curve"][3]["survival"], 0.5)
        self.assertEqual(result["curve"][4]["censored_n"], 1)

    def test_rec_and_rob_are_separate(self):
        result = survival_report(self.records, bootstrap_n=20)
        self.assertEqual(result["recovery_rec"]["value"], 1.0)
        self.assertAlmostEqual(result["robustness_rob_macro"]["value"], 0.875)
        self.assertIn("survival_ci_95_dialogue_bootstrap", result)


if __name__ == "__main__":
    unittest.main()
