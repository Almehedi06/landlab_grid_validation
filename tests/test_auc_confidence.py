"""The AUC confidence interval: does it agree with a bootstrap, and fail honestly."""

import unittest

import numpy as np

from landlab_grid_validation import roc_auc, roc_auc_ci
from landlab_grid_validation.metrics import roc_auc_variance, score_vs_binary_metrics


def bootstrap_ci(scores, observed, n_boot=4000, level=0.95, seed=0):
    """Percentile interval from resampling each class with replacement."""
    rng = np.random.default_rng(seed)
    y = observed.astype(bool)
    positives, negatives = scores[y], scores[~y]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(positives, positives.size, replace=True)
        n = rng.choice(negatives, negatives.size, replace=True)
        draws[i] = roc_auc(np.r_[p, n], np.r_[np.ones(p.size), np.zeros(n.size)])
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


class TestAucConfidenceInterval(unittest.TestCase):
    def _sample(self, n_pos=40, n_neg=160, separation=1.0, seed=1):
        rng = np.random.default_rng(seed)
        scores = np.r_[rng.normal(separation, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
        observed = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        return scores, observed

    def test_matches_a_bootstrap(self):
        scores, observed = self._sample()
        low, high = roc_auc_ci(scores, observed)
        boot_low, boot_high = bootstrap_ci(scores, observed)
        self.assertAlmostEqual(low, boot_low, delta=0.03)
        self.assertAlmostEqual(high, boot_high, delta=0.03)

    def test_brackets_the_auc_and_stays_in_range(self):
        scores, observed = self._sample()
        auc = roc_auc(scores, observed)
        low, high = roc_auc_ci(scores, observed)
        self.assertLess(low, auc)
        self.assertLess(auc, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_fewer_points_give_a_wider_interval(self):
        wide = roc_auc_ci(*self._sample(n_pos=8, n_neg=32))
        narrow = roc_auc_ci(*self._sample(n_pos=200, n_neg=800))
        self.assertGreater(wide[1] - wide[0], 3 * (narrow[1] - narrow[0]))

    def test_undefined_cases_are_nan_not_a_false_certainty(self):
        perfect = (np.array([3.0, 4.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0, 0.0]))
        self.assertEqual(roc_auc(*perfect), 1.0)
        self.assertTrue(all(np.isnan(v) for v in roc_auc_ci(*perfect)))  # zero variance

        all_tied = (np.full(10, 0.5), np.r_[np.ones(5), np.zeros(5)])
        self.assertTrue(all(np.isnan(v) for v in roc_auc_ci(*all_tied)))

        one_class = (np.arange(6.0), np.ones(6))
        self.assertTrue(np.isnan(roc_auc(*one_class)))
        self.assertTrue(all(np.isnan(v) for v in roc_auc_ci(*one_class)))

        self.assertTrue(np.isnan(roc_auc_variance(np.arange(4.0), np.r_[1, 0, 0, 0])))

    def test_the_interval_is_reported_with_every_score_comparison(self):
        scores, observed = self._sample()
        metrics = score_vs_binary_metrics(scores, observed, threshold=0.5, probability=False)
        self.assertEqual(metrics["roc_auc_ci_level"], 0.95)
        self.assertLess(metrics["roc_auc_ci_low"], metrics["roc_auc"])
        self.assertLess(metrics["roc_auc"], metrics["roc_auc_ci_high"])


if __name__ == "__main__":
    unittest.main()
