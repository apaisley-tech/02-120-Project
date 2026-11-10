"""
Tests for infinite_sites.py.

Run with: python3 -m unittest -v test_infinite_sites
"""

import unittest

import numpy as np

from infinite_sites import (
    average_total_heterozygosity,
    simulate_sites,
    sites_generation,
    sites_generation_many,
    total_heterozygosity,
)
from wright_fisher import make_crash_history, set_seed


class TestSitesGeneration(unittest.TestCase):
    def setUp(self):
        set_seed(120)

    def test_no_mutation_means_no_variation(self):
        freqs = simulate_sites([50] * 100, mu=0.0)
        self.assertEqual(len(freqs), 0)

    def test_frequencies_are_strictly_between_zero_and_one(self):
        freqs = simulate_sites([50] * 500, mu=0.05)
        self.assertGreater(len(freqs), 0)
        self.assertTrue(np.all((freqs > 0) & (freqs < 1)))

    def test_frequencies_are_whole_numbers_of_copies(self):
        freqs = simulate_sites([50] * 500, mu=0.05)
        counts = freqs * 100
        self.assertTrue(np.allclose(counts, np.round(counts)))

    def test_new_mutations_start_as_one_copy(self):
        freqs = sites_generation(pop_size=20, freqs=np.empty(0), mu=0.5)
        self.assertTrue(np.all(freqs == 1 / 40))

    def test_total_heterozygosity(self):
        self.assertAlmostEqual(total_heterozygosity(np.array([0.5, 0.1])), 0.5 + 0.18)
        self.assertEqual(total_heterozygosity(np.empty(0)), 0.0)


class TestSitesGenerationMany(unittest.TestCase):
    def setUp(self):
        set_seed(120)

    def test_trial_ids_and_frequencies_are_valid(self):
        freqs, trial_ids = np.empty(0), np.empty(0, dtype=int)
        for _ in range(200):
            freqs, trial_ids = sites_generation_many(50, freqs, trial_ids, num_trials=7, mu=0.05)
        self.assertEqual(len(freqs), len(trial_ids))
        self.assertTrue(np.all((trial_ids >= 0) & (trial_ids < 7)))
        self.assertTrue(np.all((freqs > 0) & (freqs < 1)))
        # Every population should have collected some variation by now.
        self.assertEqual(len(np.unique(trial_ids)), 7)

    def test_heterozygosity_matches_theory_in_every_population(self):
        """
        After 1000 generations at N = 50 with mu = 0.05, the expected total
        heterozygosity is close to 4N * mu - 2 * mu = 9.9 in each population.
        Averaged over 500 populations, the standard error is about 0.08, so we allow 0.4.
        """
        num_trials = 500
        freqs, trial_ids = np.empty(0), np.empty(0, dtype=int)
        for _ in range(1000):
            freqs, trial_ids = sites_generation_many(50, freqs, trial_ids, num_trials, mu=0.05)
        per_trial = np.bincount(trial_ids, weights=2 * freqs * (1 - freqs), minlength=num_trials)
        self.assertAlmostEqual(np.mean(per_trial), 9.9, delta=0.4)


class TestHeterozygosityTheory(unittest.TestCase):
    def setUp(self):
        set_seed(120)

    def test_heterozygosity_follows_theory_through_a_crash(self):
        """
        Drift in a generation of size N multiplies the expected total heterozygosity
        H by (1 - 1/2N). On average 2N * mu new sites appear, each at frequency
        1/2N, and each adds 2 * (1/2N) * (1 - 1/2N) to H. Together:
            H_next = (1 - 1/2N) * (H + 2 * mu)
        With a steady size this levels off at 4N * mu - 2 * mu.

        Run-to-run spread gives a standard error of about 2% of H here,
        so we allow 10%.
        """
        sizes = make_crash_history(normal_size=100, crash_size=5,
                                   gens_before=300, gens_crash=10, gens_after=50)
        mu = 0.02
        simulated = average_total_heterozygosity(sizes, mu, num_trials=300)

        expected = 0.0
        print()
        print(f"{'gen':>4} {'N':>4} {'simulated H':>12} {'theory H':>9}")
        for t, pop_size in enumerate(sizes, start=1):
            expected = (1 - 1 / (2 * pop_size)) * (expected + 2 * mu)
            if t % 10 == 0 or 300 < t <= 310:
                print(f"{t:>4} {pop_size:>4} {simulated[t]:>12.3f} {expected:>9.3f}")
                with self.subTest(generation=t):
                    self.assertAlmostEqual(simulated[t], expected, delta=0.1 * expected)


if __name__ == "__main__":
    unittest.main()
