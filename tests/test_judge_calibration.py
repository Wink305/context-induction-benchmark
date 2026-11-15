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


if __name__ == "__main__":
    unittest.main()
