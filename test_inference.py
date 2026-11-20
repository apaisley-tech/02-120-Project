"""
Tests for inference.py.

Run with: python3 -m unittest -v test_inference
"""

import unittest

import numpy as np

from inference import (
    KINDS,
    STATS,
    build_reference_table,
    infer,
    make_histories,
    observe,
    summarize,
    validate,
    weighted_quantiles,
)
from wright_fisher import make_crash_history, make_growth_history

# A small table that builds in a few seconds: N = 50, theta = 4N * mu = 10, n = 10.
SMALL = dict(normal_size=50, mu=0.05, sample_size=10, crash_sizes=(5, 25),
             crash_lengths=(5, 20), growth_sizes=(500,), growth_length=10, max_time=200,
             record_every=10, num_trials=40, seed=3)


class TestReferenceTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = build_reference_table(**SMALL, workers=1)

    def test_histories(self):
        histories = make_histories(50, [5, 25], [5, 20], [500], 10, 200)
        self.assertEqual([h["kind"] for h in histories], ["steady"] + ["crash"] * 4 + ["growth"])
        self.assertTrue(all(len(h["sizes"]) == 200 for h in histories))

    def test_shape_and_columns(self):
        # 6 histories x 40 populations x 21 times (0, 10, ..., 200).
        rows = 6 * 40 * 21
        for name in ["kind", "crash_size", "crash_length", "end_size", "time", "population"] + STATS:
            self.assertEqual(len(self.table[name]), rows, name)
        self.assertEqual(len(np.unique(self.table["population"])), 6 * 40)

    def test_steady_rows_have_no_time(self):
        steady = self.table["kind"] == KINDS.index("steady")
        self.assertTrue(np.all(np.isnan(self.table["time"][steady])))
        self.assertFalse(np.any(np.isnan(self.table["time"][~steady])))

    def test_each_population_keeps_its_own_kind(self):
        for population in [0, 41, 239]:
            rows = self.table["population"] == population
            self.assertEqual(len(np.unique(self.table["kind"][rows])), 1)

    def test_crash_too_small_to_sample(self):
        with self.assertRaises(ValueError):
            build_reference_table(**{**SMALL, "crash_sizes": (4,)}, workers=1)

    def test_parallel_table_is_identical(self):
        parallel = build_reference_table(**SMALL, workers=3)
        for name in STATS:
            self.assertTrue(np.array_equal(self.table[name], parallel[name]), name)

    def test_exact_match_is_kept_first(self):
        row = 1234
        observed = {name: self.table[name][row] for name in STATS}
        kept = infer(observed, self.table, num_keep=50)
        self.assertEqual(kept["distance"].min(), 0)
        self.assertAlmostEqual(kept["weight"].sum(), 1)

    def test_excluded_rows_are_never_kept(self):
        row = 1234
        observed = {name: self.table[name][row] for name in STATS}
        exclude = self.table["population"] == self.table["population"][row]
        kept = infer(observed, self.table, num_keep=50, exclude=exclude)
        self.assertNotIn(self.table["population"][row], kept["population"])

    def test_kinds_have_equal_prior_odds(self):
        """
        Keeping every row returns the prior: 1/3 for each kind, even though there
        are four crash histories and only one steady one.
        """
        kept = infer({name: 0.0 for name in STATS}, self.table,
                     num_keep=len(self.table["kind"]) - 1)
        probability = summarize(kept)["probability"]
        for kind in KINDS:
            self.assertAlmostEqual(probability[kind], 1 / 3, places=3)

    def test_validate_shapes(self):
        results = validate(self.table, num_tests=30, num_keep=50)
        self.assertEqual(len(results["kind"]), 30)
        self.assertTrue(np.all((results["p_true"] >= 0) & (results["p_true"] <= 1)))


class TestWeightedQuantiles(unittest.TestCase):
    def test_hand_worked(self):
        # Half the weight is on 1, so the median is 1; 3 carries the top quarter.
        values, weights = np.array([3.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.0])
        self.assertEqual(weighted_quantiles(values, weights, [0.25, 0.5, 0.9]), [1.0, 1.0, 3.0])

    def test_no_weight(self):
        self.assertTrue(np.isnan(weighted_quantiles(np.array([1.0]), np.zeros(1), [0.5])[0]))


class TestInferenceWorks(unittest.TestCase):
    """
    End to end: a recent severe crash and big growth should be recognized from one
    sample. These populations use seeds the table does not use.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = build_reference_table(**{**SMALL, "num_trials": 100})

    def probability_of(self, kind: str, sizes: list[int], seed: int) -> float:
        observed = observe(sizes, 50, 0.05, 10, seed)
        return summarize(infer(observed, self.table, num_keep=200))["probability"][kind]

    def test_recent_severe_crash(self):
        # Crash to N = 5 for 20 generations: strength 2, sampled 30 generations after it began.
        sizes = make_crash_history(50, 5, 0, 20, 10)
        average = np.mean([self.probability_of("crash", sizes, 500 + i) for i in range(5)])
        print(f"\nAverage P(crash) after a severe crash = {average:.2f}")
        self.assertGreater(average, 0.7)

    def test_big_growth(self):
        sizes = make_growth_history(50, 500, 0, 10, 140)
        average = np.mean([self.probability_of("growth", sizes, 600 + i) for i in range(5)])
        print(f"\nAverage P(growth) after 10x growth = {average:.2f}")
        self.assertGreater(average, 0.7)


if __name__ == "__main__":
    unittest.main()
