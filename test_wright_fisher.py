"""
Tests for wright_fisher.py.

Run with: python3 -m unittest -v test_wright_fisher
"""

import math
import unittest

from wright_fisher import (
    average_heterozygosity,
    estimate_fixation_probability,
    make_crash_history,
    make_growth_history,
    run_until_fixed_or_lost,
    set_seed,
    simulate_drift,
)


class TestFixationProbability(unittest.TestCase):
    def setUp(self):
        set_seed(120)

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


class TestChangingPopulationSize(unittest.TestCase):
    def setUp(self):
        set_seed(120)

    def test_simulate_drift_returns_one_frequency_per_generation_plus_start(self):
        freqs = simulate_drift([100, 5, 100], p0=0.3)
        self.assertEqual(len(freqs), 4)
        self.assertEqual(freqs[0], 0.3)

    def test_crash_history(self):
        sizes = make_crash_history(normal_size=100, crash_size=5,
                                   gens_before=2, gens_crash=3, gens_after=1)
        self.assertEqual(sizes, [100, 100, 5, 5, 5, 100])

    def test_growth_history(self):
        sizes = make_growth_history(start_size=10, end_size=1000,
                                    gens_before=1, gens_growth=2, gens_after=1)
        self.assertEqual(sizes, [10, 100, 1000, 1000])

    def test_heterozygosity_follows_theory_through_a_crash(self):
        """
        Each Wright-Fisher generation of size N multiplies the expected
        heterozygosity by (1 - 1/2N), so after a history of sizes
        H_t = H_0 * product of (1 - 1/2N) over those generations.
        """
        sizes = make_crash_history(normal_size=200, crash_size=5,
                                   gens_before=5, gens_crash=5, gens_after=5)
        p0 = 0.5
        simulated = average_heterozygosity(sizes, p0, num_trials=2000)

        expected = 2 * p0 * (1 - p0)
        print()
        print(f"{'gen':>4} {'N':>4} {'simulated H':>12} {'theory H':>9}")
        for t, pop_size in enumerate(sizes, start=1):
            expected *= 1 - 1 / (2 * pop_size)
            print(f"{t:>4} {pop_size:>4} {simulated[t]:>12.3f} {expected:>9.3f}")
            with self.subTest(generation=t):
                self.assertAlmostEqual(simulated[t], expected, delta=0.02)


if __name__ == "__main__":
    unittest.main()
