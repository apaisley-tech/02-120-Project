"""
Tests for wright_fisher.py.

Run with: python3 -m unittest -v test_wright_fisher
"""

import math
import random
import unittest

from wright_fisher import estimate_fixation_probability, run_until_fixed_or_lost


class TestFixationProbability(unittest.TestCase):
    def setUp(self):
        random.seed(120)

    def test_fixation_probability_equals_starting_frequency(self):
        """
        A neutral allele starting at frequency p0 should fix in about p0 of trials.
        Each trial fixes (1) or not (0), so the estimate has standard error
        sqrt(p0 * (1 - p0) / num_trials); we allow 4 standard errors.
        """
        num_trials = 2000
        print()
        print(f"{'N':>4} {'p0':>5} {'fixed':>7} {'allowed range':>17}")
        for pop_size in [10, 50]:
            for p0 in [0.1, 0.25, 0.5, 0.75, 0.9]:
                with self.subTest(pop_size=pop_size, p0=p0):
                    estimate = estimate_fixation_probability(pop_size, p0, num_trials)
                    tolerance = 4 * math.sqrt(p0 * (1 - p0) / num_trials)
                    print(f"{pop_size:>4} {p0:>5.2f} {estimate:>7.3f} "
                          f"   {p0 - tolerance:.3f} to {p0 + tolerance:.3f}")
                    self.assertAlmostEqual(estimate, p0, delta=tolerance)

    def test_absent_allele_never_fixes(self):
        self.assertFalse(run_until_fixed_or_lost(pop_size=50, p0=0.0))

    def test_fixed_allele_stays_fixed(self):
        self.assertTrue(run_until_fixed_or_lost(pop_size=50, p0=1.0))


if __name__ == "__main__":
    unittest.main()
