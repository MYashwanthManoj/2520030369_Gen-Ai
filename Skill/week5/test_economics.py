import unittest
import sys
import os

# Ensure week5 directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from token_economics import FeatureTrafficSpec, TokenEconomicsCalculator, PRICING_CATALOG

class TestTokenEconomics(unittest.TestCase):
    def setUp(self):
        self.spec = FeatureTrafficSpec(
            feature_name="Test Feature",
            expected_daily_active_users=1000,
            queries_per_user_per_day=5,
            avg_input_tokens_per_call=1000,
            avg_output_tokens_per_call=200,
            peak_concurrency_multiplier=2.0
        )
        self.calc = TokenEconomicsCalculator(self.spec)

    def test_daily_and_monthly_calls(self):
        self.assertEqual(self.calc.daily_calls, 5000)
        self.assertEqual(self.calc.monthly_calls, 150000)

    def test_token_totals(self):
        self.assertEqual(self.calc.daily_input_tokens, 5_000_000)
        self.assertEqual(self.calc.daily_output_tokens, 1_000_000)
        self.assertEqual(self.calc.total_daily_tokens, 6_000_000)

    def test_cost_calculation_gpt4o_mini(self):
        res = self.calc.calculate_cost_for_model("gpt-4o-mini")
        # 5M * $0.15/1M = $0.75 input
        # 1M * $0.60/1M = $0.60 output
        # daily = $1.35
        self.assertAlmostEqual(res["daily_total_cost_usd"], 1.35, places=2)
        self.assertAlmostEqual(res["monthly_total_cost_usd"], 1.35 * 30, places=2)

    def test_full_report_generation(self):
        report = self.calc.generate_full_comparison_report()
        self.assertIn("feature_spec", report)
        self.assertIn("model_cost_evaluations", report)
        self.assertEqual(len(report["model_cost_evaluations"]), len(PRICING_CATALOG))

if __name__ == "__main__":
    unittest.main()
