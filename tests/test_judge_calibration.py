import unittest

from context_benchmark.scoring import judge_calibration


class JudgeCalibrationTests(unittest.TestCase):
    def test_two_error_rates_and_unextractable_are_separate(self):
        rows = [
            {"gold": "yes", "predicted": "yes"},
            {"gold": "yes", "predicted": "no"},
            {"gold": "no", "predicted": "no"},
            {"gold": "no", "predicted": "unextractable"},
        ]
        result = judge_calibration(rows)
        self.assertEqual(result["gold_yes_error_rate"]["value"], 0.5)
        self.assertEqual(result["gold_no_error_rate"]["value"], 0.5)
        self.assertEqual(result["unextractable_rate"]["value"], 0.25)
        self.assertEqual(result["overall_error_rate"]["denominator"], 4)

    def test_unextractable_is_in_kappa_gate_not_excluded(self):
        """Abstentions must lower the gate kappa, not vanish from its denominator."""
        rows = [
            {"gold": "yes", "predicted": "yes"},
            {"gold": "no", "predicted": "no"},
            {"gold": "yes", "predicted": "unextractable"},
            {"gold": "no", "predicted": "unextractable"},
        ]
        result = judge_calibration(rows)
        self.assertIn("human_machine_kappa_includes_unextractable", result)
        self.assertEqual(
            result["human_machine_kappa_includes_unextractable"]["n"], 4,
        )
        # scorable-only sees just 2 rows; the gate sees all 4.
        self.assertEqual(result["human_machine_kappa_scorable_only"]["n"], 2)
        self.assertFalse(result["pass_kappa_0_60"])


if __name__ == "__main__":
    unittest.main()
