"""
Tests for animation.py, on small made-up data.

Run with: python3 -m unittest -v test_animation
"""

import unittest

import numpy as np

from animation import animated_histogram_figure, animated_sfs_figure, histogram_edges

GENS = [0, 10, 20, 30]
SIZES = [10] * 20 + [100] * 10  # A crash for 20 generations, then back to 100.


class TestAnimatedSFS(unittest.TestCase):
    def setUp(self):
        # Spectra for n = 5 (4 classes), one row per recorded generation.
        self.event = np.array([[4, 2, 1.3, 1], [1, 2, 3, 4], [2, 2, 2, 2], [4, 2, 1.3, 1]])
        self.steady = np.tile([4, 2, 4 / 3, 1], (4, 1))
        self.fig = animated_sfs_figure(GENS, SIZES, 100, self.event, self.steady, "crash", 20,
                                       start=2)

    def test_one_frame_per_generation(self):
        self.assertEqual([frame.name for frame in self.fig.frames], ["0", "10", "20", "30"])

    def test_starts_at_the_chosen_generation(self):
        slider = self.fig.layout.sliders[0]
        self.assertEqual(slider.steps[slider.active].label, "20")
        self.assertEqual(list(self.fig.data[2].x), [20, 20])  # The time marker.

    def test_frame_shows_that_generations_spectrum(self):
        # Generation 10 has the spectrum 1, 2, 3, 4: its share of singletons is 0.1,
        # against a steady share of 1 / a1 = 1 / (1 + 1/2 + 1/3 + 1/4) = 0.48.
        frame = self.fig.frames[1]
        self.assertEqual(list(frame.data[0].x), [10, 10])
        self.assertAlmostEqual(frame.data[1].y[0], 0.1 * (1 + 1 / 2 + 1 / 3 + 1 / 4))

    def test_axis_is_fixed_and_covers_every_frame(self):
        low, high = self.fig.layout.yaxis2.range
        for frame in self.fig.frames:
            for trace in frame.data[1:]:
                values = np.log10(np.array(trace.y, dtype=float))
                self.assertTrue(np.all((values >= low) & (values <= high)))


class TestAnimatedHistogram(unittest.TestCase):
    def test_counts_and_edges(self):
        rng = np.random.default_rng(1)
        event_d = rng.normal(0.5, 0.4, size=(50, 4))
        event_d[0, 1] = np.nan  # D undefined for one population at generation 10.
        steady_d = rng.normal(0, 0.3, size=(50, 4))
        fig = animated_histogram_figure(GENS, SIZES, 100, event_d, steady_d, -0.6, 0.6,
                                        "crash", 20, start=0)
        self.assertEqual(len(fig.frames), 4)
        # Generation 10: 49 defined event values and 50 steady ones, all counted.
        self.assertEqual(sum(fig.frames[1].data[2].y), 49)
        self.assertEqual(sum(fig.frames[1].data[1].y), 50)
        tallest = max(max(trace.y) for frame in fig.frames for trace in frame.data[1:])
        self.assertGreaterEqual(fig.layout.yaxis2.range[1], tallest)

    def test_edges_cover_values_and_limits(self):
        edges = histogram_edges([np.array([0.1, np.nan, 0.9])], lower=-0.3, upper=0.4)
        self.assertTrue(np.allclose(edges, [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]))

    def test_edges_with_no_defined_values(self):
        edges = histogram_edges([np.full(3, np.nan)], lower=0.0, upper=0.0)
        self.assertEqual(len(edges), 2)

    def test_steady_history(self):
        d = np.zeros((5, 4))
        fig = animated_histogram_figure(GENS, [100] * 30, 100, d, d, -1, 1, "steady", 0, 0)
        self.assertEqual(len(fig.frames), 4)


if __name__ == "__main__":
    unittest.main()
