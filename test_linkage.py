"""
Tests for linkage.py, and for the linked options of recovery.py and inference.py.

Run with: python3 -m unittest -v test_linkage
"""

import unittest
from unittest import mock

import numpy as np

import inference
import wright_fisher
from diversity import harmonic_sums, sample_stats_per_trial
from linkage import linked_sample_stats
from recovery import compare_histories
from test_inference import SMALL
from wright_fisher import make_crash_history

# N = 50 and mu = 0.05 give theta = 4N * mu = 10, as in the rest of the project.
N, MU, n = 50, 0.05, 10
BURN_IN = [N] * (10 * N)


class TestLinkedSampleStats(unittest.TestCase):
    def test_shape_matches_unlinked(self):
        wright_fisher.set_seed(1)
        stats = linked_sample_stats(BURN_IN, MU, n, 7, [100, 300, 500])
        self.assertEqual(set(stats), {"S", "pi", "theta_w", "D", "singletons"})
        for values in stats.values():
            self.assertEqual(values.shape, (7, 3))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            linked_sample_stats(BURN_IN, MU, n, 0, [100])
        with self.assertRaises(ValueError):
            linked_sample_stats(BURN_IN, MU, n, 5, [len(BURN_IN) + 1])
        with self.assertRaises(ValueError):
            linked_sample_stats([2] * 10, MU, n, 5, [10])  # 4 gene copies, n = 10

    def test_no_mutation_no_variation(self):
        stats = linked_sample_stats(BURN_IN, 0.0, n, 20, [len(BURN_IN)])
        self.assertTrue(np.all(stats["S"] == 0))
        self.assertTrue(np.all(np.isnan(stats["D"])))

    def test_first_generation_has_only_singletons(self):
        # Generation 0 has no variation, so every mutation in a sample from generation 1
        # arose in that generation, on one sampled sequence.
        wright_fisher.set_seed(2)
        stats = linked_sample_stats([N] * 5, 0.5, n, 200, [1])
        self.assertTrue(np.all(stats["S"] == stats["singletons"]))
        # Expected S = n * mu per sample.
        self.assertAlmostEqual(stats["S"].mean(), n * 0.5, delta=0.3)

    def test_steady_averages_match_theory(self):
        # E[pi] = E[theta_W] = 4N * mu, and E[singletons] = theta, as for unlinked sites.
        wright_fisher.set_seed(3)
        stats = linked_sample_stats(BURN_IN, MU, n, 2000, [len(BURN_IN)])
        theta = 4 * N * MU
        self.assertAlmostEqual(stats["pi"].mean(), theta, delta=0.5)
        self.assertAlmostEqual(stats["theta_w"].mean(), theta, delta=0.5)
        self.assertAlmostEqual(stats["singletons"].mean(), theta, delta=0.5)
        a1, _ = harmonic_sums(n)
        self.assertAlmostEqual(stats["S"].mean(), theta * a1, delta=1.5)

    def test_linked_d_varies_more(self):
        # Linked sites share one tree, so D varies far more from sample to sample,
        # close to the spread of about 1 that Tajima's test assumes.
        gens = [len(BURN_IN) + t for t in range(0, 101, 10)]
        wright_fisher.set_seed(4)
        linked = linked_sample_stats(BURN_IN + [N] * 100, MU, n, 300, gens)["D"]
        wright_fisher.set_seed(4)
        unlinked = sample_stats_per_trial(BURN_IN + [N] * 100, MU, n, 300, gens)["D"]
        self.assertGreater(np.nanstd(linked), 0.7)
        self.assertGreater(np.nanstd(linked), 1.8 * np.nanstd(unlinked))

    def test_crash_average_matches_unlinked(self):
        # Just after a crash, D is positive and pi low, with linked or unlinked sites:
        # the averages do not depend on linkage.
        sizes = BURN_IN + make_crash_history(N, 5, 0, 10, 0)
        gen = [len(sizes)]
        wright_fisher.set_seed(5)
        linked = linked_sample_stats(sizes, MU, n, 3000, gen)
        wright_fisher.set_seed(5)
        unlinked = sample_stats_per_trial(sizes, MU, n, 3000, gen)
        self.assertGreater(np.nanmean(linked["D"]), 0.3)
        self.assertAlmostEqual(linked["pi"].mean(), unlinked["pi"].mean(),
                               delta=0.1 * unlinked["pi"].mean())


class TestLinkedOptions(unittest.TestCase):
    def test_compare_histories_with_linked_sites(self):
        wright_fisher.set_seed(6)
        results = compare_histories(N, {"crash": make_crash_history(N, 5, 0, 10, 290)}, MU, n,
                                    num_trials=100, record_every=10,
                                    simulate=linked_sample_stats)
        self.assertIsNotNone(results["crash"]["recovery"]["D"])
        self.assertEqual(results["crash"]["event_trials"]["D"].shape, (100, 31))

    def test_linked_table_differs_and_validates_across_tables(self):
        unlinked = inference.build_reference_table(**SMALL, workers=1)
        linked = inference.build_reference_table(**SMALL, workers=1, linked=True)
        self.assertEqual(len(linked["kind"]), len(unlinked["kind"]))
        self.assertGreater(np.std(linked["pi"]), np.std(unlinked["pi"]))
        # Test rows come from the linked table; the true kinds are its kinds.
        results = inference.validate(unlinked, num_tests=30, num_keep=50, tests=linked)
        self.assertEqual(sorted(np.unique(results["kind"])), [0, 1, 2])
        rates = inference.validation_rates(results)
        self.assertTrue(all(0 <= value <= 1 for value in rates.values() if not np.isnan(value)))

    def test_cache_name_unchanged_for_unlinked_tables(self):
        # The included table must still load without being rebuilt.
        with mock.patch.object(inference, "build_reference_table", autospec=True,
                               side_effect=AssertionError("cache name changed")):
            table = inference.cached_reference_table(normal_size=200, mu=0.0125,
                                                     sample_size=20)
        self.assertEqual(len(table["kind"]), 424_200)


if __name__ == "__main__":
    unittest.main()
