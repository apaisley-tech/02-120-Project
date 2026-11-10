"""
Tests for recovery.py.

Run with: python3 -m unittest -v test_recovery
"""

import unittest

from recovery import (
    compare_histories,
    compare_to_steady,
    recovery_time,
    recovery_times_over_seeds,
)
from wright_fisher import make_crash_history, set_seed


class TestRecoveryTime(unittest.TestCase):
    gens = [0, 10, 20, 30, 40]
    baseline = [0.0] * 5
    tolerances = [0.1] * 5

    def test_never_affected(self):
        values = [0.05, -0.05, 0.0, 0.05, 0.0]
        self.assertEqual(recovery_time(self.gens, values, self.baseline, self.tolerances), 0)

    def test_recovers_after_last_departure(self):
        # The values return to the baseline at generation 20, leave it again at 30,
        # and only stay close from generation 40 on.
        values = [0.0, 0.5, 0.05, 0.3, 0.05]
        self.assertEqual(recovery_time(self.gens, values, self.baseline, self.tolerances), 40)

    def test_not_recovered(self):
        values = [0.0, 0.5, 0.4, 0.3, 0.2]
        self.assertIsNone(recovery_time(self.gens, values, self.baseline, self.tolerances))

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            recovery_time(self.gens, [0.0], self.baseline, self.tolerances)


class TestCompareToSteady(unittest.TestCase):
    """
    Uses a small population (N = 50, theta = 4N * mu = 10) so the tests run quickly.
    """

    normal_size = 50
    mu = 0.05
    sample_size = 10

    def setUp(self):
        set_seed(120)

    def test_steady_population_is_never_different(self):
        result = compare_to_steady(self.normal_size, [self.normal_size] * 100, self.mu,
                                   self.sample_size, num_trials=50, record_every=20)
        self.assertEqual(result["gens"], [0, 20, 40, 60, 80, 100])
        for stat, gen in result["recovery"].items():
            with self.subTest(stat=stat):
                self.assertEqual(gen, 0)

    def test_histories_must_have_same_length(self):
        with self.assertRaises(ValueError):
            compare_histories(self.normal_size, {"a": [50] * 10, "b": [50] * 20}, self.mu,
                              self.sample_size, num_trials=5, record_every=5)

    def test_crash_is_detected_and_recovers(self):
        sizes = make_crash_history(normal_size=self.normal_size, crash_size=5, gens_before=0,
                                   gens_crash=10, gens_after=400)
        result = compare_to_steady(self.normal_size, sizes, self.mu, self.sample_size,
                                   num_trials=100, record_every=10)
        print(f"\nRecovery after a crash from 50 to 5: {result['recovery']}")
        for stat, gen in result["recovery"].items():
            with self.subTest(stat=stat):
                self.assertIsNotNone(gen)
                self.assertGreater(gen, 0)

    def test_recovery_times_over_seeds_shape(self):
        times = recovery_times_over_seeds(self.normal_size, crash_sizes=[25, 5], gens_crash=5,
                                          gens_after=50, mu=self.mu,
                                          sample_size=self.sample_size, num_trials=20,
                                          record_every=10, seeds=[1, 2, 3])
        self.assertEqual(set(times), {"S", "pi", "theta_w", "D"})
        for stat, values in times.items():
            with self.subTest(stat=stat):
                self.assertEqual(values.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
