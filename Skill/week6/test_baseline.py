import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_evaluator import compute_token_overlap_metrics, BaselineEvaluator

class TestBaselineEvaluator(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.golden_path = os.path.join(self.base_dir, "golden_dataset.json")

    def test_golden_dataset_exists(self):
        self.assertTrue(os.path.exists(self.golden_path))

    def test_overlap_metric_exact(self):
        text = "Hello world this is a test"
        metrics = compute_token_overlap_metrics(text, text)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_overlap_metric_partial(self):
        ref = "Transformers rely on attention mechanisms"
        cand = "Transformers utilize self attention mechanisms"
        metrics = compute_token_overlap_metrics(ref, cand)
        self.assertGreater(metrics["f1_score"], 0.5)

    def test_evaluator_run(self):
        evaluator = BaselineEvaluator(self.golden_path, release_tag="v0.1-test")
        report = evaluator.run_evaluation()
        self.assertEqual(report["summary"]["total_questions"], 5)
        self.assertGreater(report["summary"]["pass_rate_pct"], 80.0)

if __name__ == "__main__":
    unittest.main()

