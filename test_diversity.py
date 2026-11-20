"""
Tests for diversity.py.

Run with: python3 -m unittest -v test_diversity
"""

import math
import unittest

import numpy as np

from diversity import (
    average_sample_stats,
    control_limits,
    expected_sfs_shares,
    nucleotide_diversity,
    sample_counts,
    sample_stats_and_sfs,
    sample_stats_per_trial,
    sfs_ratio,
    site_frequency_spectrum,
    summarize_trials,
    tajima_critical_values,
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


class TestPerTrialStats(unittest.TestCase):
    """
    Checks that per-population statistics have the right shape and that
    averaging them gives the same answer as average_sample_stats.
    """

    def test_shape_and_matches_average(self):
        sizes, mu, n, trials, gens = [50] * 300, 0.02, 10, 30, [100, 200, 300]
        set_seed(7)
        per_trial = sample_stats_per_trial(sizes, mu, n, trials, gens)
        set_seed(7)
        averaged = average_sample_stats(sizes, mu, n, trials, gens)
        for name in ("S", "pi", "theta_w", "D"):
            self.assertEqual(per_trial[name].shape, (trials, len(gens)))
        self.assertEqual(summarize_trials(per_trial), averaged)

    def test_summarize_skips_undefined_d(self):
        """
        Hand-worked: D values 1 and 3 plus one undefined value average to 2,
        with standard error sd / sqrt(2) = sqrt(2) / sqrt(2) = 1.
        """
        stats = {"D": np.array([[1.0], [3.0], [np.nan]])}
        summary = summarize_trials(stats)
        self.assertAlmostEqual(summary["D"][0], 2.0)
        self.assertAlmostEqual(summary["D_se"][0], 1.0)


class TestCriticalValues(unittest.TestCase):
    """
    Checks the limits of Tajima's test from the beta distribution.
    """

    def test_limits_straddle_zero(self):
        for n in (4, 10, 20, 50):
            lower, upper = tajima_critical_values(n)
            self.assertLess(lower, 0)
            self.assertGreater(upper, 0)

    def test_close_to_normal_for_large_samples(self):
        """
        D is scaled to have mean 0 and variance 1, so for large n the 5% limits
        are near the normal ones, +-1.96, skewed to the right (Tajima 1989, Table 2).
        """
        lower, upper = tajima_critical_values(50)
        self.assertAlmostEqual(lower, -1.96, delta=0.3)
        self.assertAlmostEqual(upper, 1.96, delta=0.3)
        self.assertGreater(abs(upper), abs(lower))

    def test_stricter_level_widens_limits(self):
        lower5, upper5 = tajima_critical_values(20, 0.05)
        lower1, upper1 = tajima_critical_values(20, 0.01)
        self.assertLess(lower1, lower5)
        self.assertGreater(upper1, upper5)

    def test_sample_too_small(self):
        with self.assertRaises(ValueError):
            tajima_critical_values(3)


class TestControlLimits(unittest.TestCase):
    """
    Checks the limits taken from simulated steady populations.
    """

    def test_hand_worked_quantiles(self):
        """
        For the values 0, 1, ..., 100 (plus undefined values, which are ignored),
        the 10% limits are the 5th and 95th percentiles: 5 and 95.
        """
        d = np.append(np.arange(101.0), [np.nan, np.nan])
        lower, upper = control_limits(d, level=0.1)
        self.assertAlmostEqual(lower, 5)
        self.assertAlmostEqual(upper, 95)

    def test_all_undefined(self):
        lower, upper = control_limits(np.full((3, 2), np.nan))
        self.assertTrue(math.isnan(lower) and math.isnan(upper))

    def test_new_steady_populations_flagged_about_five_percent(self):
        """
        Limits from one set of steady populations should flag about 5% of a
        second, independent set. With 400 populations the standard error of that
        share is about 1.1%, so we allow 1.5% to 9%.
        """
        set_seed(120)
        sizes, gens = [100] * 1000, [1000]
        control = sample_stats_per_trial(sizes, 0.025, 20, 400, gens)["D"]
        fresh = sample_stats_per_trial(sizes, 0.025, 20, 400, gens)["D"][:, 0]
        lower, upper = control_limits(control)
        share = np.mean((fresh < lower) | (fresh > upper))
        print(f"\nShare of new steady samples outside the control limits = {share:.3f} "
              f"(target 0.05)")
        self.assertGreater(share, 0.015)
        self.assertLess(share, 0.09)


class TestSiteFrequencySpectrum(unittest.TestCase):
    def test_hand_worked_spectrum(self):
        # n = 5: two sites carried by 1 sequence, one by 3, none by 2 or 4.
        self.assertEqual(list(site_frequency_spectrum(np.array([1, 3, 1]), 5)), [2, 0, 1, 0])

    def test_empty_sample(self):
        self.assertEqual(list(site_frequency_spectrum(np.empty(0, dtype=int), 4)), [0, 0, 0])

    def test_expected_shares(self):
        # n = 4: a1 = 11/6, so the shares are (6/11) * (1, 1/2, 1/3).
        shares = expected_sfs_shares(4)
        self.assertTrue(np.allclose(shares, [6 / 11, 3 / 11, 2 / 11]))
        self.assertAlmostEqual(shares.sum(), 1)

    def test_ratio_is_one_for_a_steady_shape(self):
        # Any multiple of theta / i has exactly the steady shape.
        spectrum = 7.5 / np.arange(1, 20)
        self.assertTrue(np.allclose(sfs_ratio(spectrum), 1))
        self.assertTrue(np.all(np.isnan(sfs_ratio(np.zeros(5)))))

    def test_same_statistics_with_or_without_spectrum(self):
        """
        Counting the spectrum draws no random numbers, so with the same seed both
        functions give exactly the same statistics.
        """
        args = ([30] * 200, 0.05, 8, 20, [100, 200])
        set_seed(5)
        stats_only = sample_stats_per_trial(*args)
        set_seed(5)
        stats, _ = sample_stats_and_sfs(*args)
        for name in stats_only:
            self.assertTrue(np.array_equal(stats_only[name], stats[name], equal_nan=True))

    def test_singletons_match_first_class_of_spectrum(self):
        set_seed(5)
        stats, sfs = sample_stats_and_sfs([30] * 200, 0.05, 8, 20, [100, 200])
        self.assertTrue(np.allclose(stats["singletons"].mean(axis=0), sfs[:, 0]))
        self.assertTrue(np.all(stats["singletons"] <= stats["S"]))

    def test_spectrum_adds_up_to_average_s(self):
        # Every segregating site falls in exactly one class, so the classes add up to S.
        set_seed(5)
        stats, sfs = sample_stats_and_sfs([30] * 200, 0.05, 8, 20, [100, 200])
        self.assertEqual(sfs.shape, (2, 7))
        self.assertTrue(np.allclose(sfs.sum(axis=1), stats["S"].mean(axis=0)))

    def test_steady_spectrum_matches_theory(self):
        """
        In a steady population, the expected number of sites carried by i of the
        n sampled sequences is theta / i, with theta = 4N * mu (here 10). So
        i * sfs[i] / theta should be about 1 in every class. Averaging 400
        populations at 3 generations, the standard error is at most about 4%
        per class, so we allow 15%.
        """
        set_seed(120)
        pop_size, mu, n = 100, 0.025, 10
        theta = 4 * pop_size * mu
        _, sfs = sample_stats_and_sfs([pop_size] * 1500, mu, n, num_trials=400,
                                      record_gens=[1000, 1250, 1500])
        scaled = np.arange(1, n) * sfs.mean(axis=0) / theta
        print("\ni * sfs[i] / theta in a steady population:", np.round(scaled, 2))
        for i, value in enumerate(scaled, start=1):
            with self.subTest(i=i):
                self.assertAlmostEqual(value, 1, delta=0.15)

    def test_crash_loses_rare_variants_and_growth_adds_them(self):
        """
        Just after a crash, rare variants are missing (fewer singletons than a steady
        shape); during growth, new rare variants pile up (more singletons).
        """
        set_seed(120)
        burn_in = [100] * 1000
        crash = burn_in + make_crash_history(100, 5, 0, 10, 0)
        growth = burn_in + make_growth_history(100, 2000, 0, 50, 50)
        _, crash_sfs = sample_stats_and_sfs(crash, 0.025, 10, 200, [len(crash)])
        _, growth_sfs = sample_stats_and_sfs(growth, 0.025, 10, 200, [len(growth)])
        self.assertLess(sfs_ratio(crash_sfs)[0, 0], 0.8)
        self.assertGreater(sfs_ratio(growth_sfs)[0, 0], 1.2)


if __name__ == "__main__":
    unittest.main()
