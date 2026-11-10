"""
Tests for diversity.py.

Run with: python3 -m unittest -v test_diversity
"""

import math
import unittest

import numpy as np

from diversity import (
    average_sample_stats,
    nucleotide_diversity,
    sample_counts,
    tajimas_d,
    tajimas_d_from_summary,
    watterson_theta,
)
from wright_fisher import make_crash_history, make_growth_history, set_seed


class TestStatistics(unittest.TestCase):
    """
    Hand-worked example: n = 4 sequences with three segregating sites carried by
    1, 1 and 2 of the sequences.
    """

    counts = np.array([1, 1, 2])

    def test_nucleotide_diversity(self):
        # Pairwise differences 1*3 + 1*3 + 2*2 = 10 over 6 pairs.
        self.assertAlmostEqual(nucleotide_diversity(self.counts, 4), 10 / 6)

    def test_watterson_theta(self):
        # a1 = 1 + 1/2 + 1/3 = 11/6.
        self.assertAlmostEqual(watterson_theta(self.counts, 4), 3 / (11 / 6))

    def test_tajimas_d(self):
        # With exact fractions, e1 = 2/363 and e2 = 83/30855, so
        # D = (5/3 - 18/11) / sqrt(3 * e1 + 6 * e2).
        expected = (5 / 3 - 18 / 11) / math.sqrt(3 * 2 / 363 + 6 * 83 / 30855)
        self.assertAlmostEqual(tajimas_d(self.counts, 4), expected)

    def test_tajimas_d_from_summary_on_arrays(self):
        d = tajimas_d_from_summary(np.array([10 / 6, 0.0]), np.array([3, 0]), 4)
        self.assertAlmostEqual(d[0], tajimas_d(self.counts, 4))
        self.assertTrue(math.isnan(d[1]))

    def test_tajimas_d_undefined_without_variation(self):
        self.assertTrue(math.isnan(tajimas_d(np.empty(0, dtype=int), 10)))


class TestSampling(unittest.TestCase):
    def setUp(self):
        set_seed(120)

    def test_counts_are_segregating_in_sample(self):
        freqs = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        counts = sample_counts(freqs, pop_size=50, sample_size=10)
        self.assertTrue(np.all((counts > 0) & (counts < 10)))

    def test_sampling_everyone_gives_exact_counts(self):
        freqs = np.array([0.01, 0.25, 0.5])
        counts = sample_counts(freqs, pop_size=50, sample_size=100)
        self.assertEqual(list(counts), [1, 25, 50])

    def test_sample_size_must_fit_population(self):
        with self.assertRaises(ValueError):
            sample_counts(np.array([0.5]), pop_size=5, sample_size=11)


class TestNeutralTheory(unittest.TestCase):
    """
    Checks against theory, using N = 100 and mu = 0.025 so that theta = 4N * mu = 10,
    with samples of n = 20 sequences.
    """

    pop_size = 100
    mu = 0.025
    sample_size = 20
    burn_in = 10 * pop_size

    def setUp(self):
        set_seed(120)

    def test_steady_population(self):
        """
        At equilibrium E[pi] = theta and E[S] = theta * a1, so E[theta_W] = theta,
        and Tajima's D averages about 0. Run-to-run spread gives standard errors
        of about 2.5% for pi and S and 0.05 for D, so we allow 10% and 0.2.
        """
        theta = 4 * self.pop_size * self.mu
        a1 = sum(1 / i for i in range(1, self.sample_size))
        stats = average_sample_stats([self.pop_size] * self.burn_in, self.mu,
                                     self.sample_size, num_trials=400,
                                     record_gens=[self.burn_in])
        print()
        print(f"pi = {stats['pi'][0]:.2f}, theta_W = {stats['theta_w'][0]:.2f} "
              f"(theory {theta:.2f}); S = {stats['S'][0]:.1f} (theory {theta * a1:.1f}); "
              f"D = {stats['D'][0]:.3f} (theory about 0)")
        self.assertAlmostEqual(stats["pi"][0], theta, delta=0.1 * theta)
        self.assertAlmostEqual(stats["theta_w"][0], theta, delta=0.1 * theta)
        self.assertAlmostEqual(stats["S"][0], theta * a1, delta=0.1 * theta * a1)
        self.assertAlmostEqual(stats["D"][0], 0, delta=0.2)

    def test_crash_makes_d_positive(self):
        """
        A crash wipes out rare variants faster than common ones, so pi exceeds
        theta_W and D turns positive.
        """
        sizes = [self.pop_size] * self.burn_in + make_crash_history(
            normal_size=self.pop_size, crash_size=10, gens_before=0, gens_crash=10,
            gens_after=5)
        stats = average_sample_stats(sizes, self.mu, self.sample_size, num_trials=200,
                                     record_gens=[len(sizes)])
        print(f"\nD just after a crash = {stats['D'][0]:.3f} (theory: positive)")
        self.assertGreater(stats["D"][0], 0.3)

    def test_growth_makes_d_negative(self):
        """
        Rapid growth adds many new mutations, which are all rare, so theta_W
        exceeds pi and D turns negative.
        """
        sizes = [self.pop_size] * self.burn_in + make_growth_history(
            start_size=self.pop_size, end_size=20 * self.pop_size, gens_before=0,
            gens_growth=20, gens_after=80)
        stats = average_sample_stats(sizes, self.mu, self.sample_size, num_trials=200,
                                     record_gens=[len(sizes)])
        print(f"\nD after rapid growth = {stats['D'][0]:.3f} (theory: negative)")
        self.assertLess(stats["D"][0], -0.3)


if __name__ == "__main__":
    unittest.main()
