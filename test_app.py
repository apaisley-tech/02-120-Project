"""
Tests for app.py: runs the real app without a browser, using Streamlit's AppTest.

Run with: python3 -m unittest -v test_app
"""

import unittest

from streamlit.testing.v1 import AppTest


def app_with(kind: str, normal_size: int, crash_size: int | None = None) -> AppTest:
    """
    Opens the app in "Explore an event" mode with small, fast settings.
    Parameters:
    - kind (str): "crash", "growth" or "steady".
    - normal_size (int): The normal population size.
    - crash_size (int | None): For a crash, the size to crash to (None keeps the default).
    Returns:
    - AppTest: The app, ready to press Run simulation.
    """
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    # The first radio is the mode ("Explore an event" by default); the second is the event.
    at.sidebar.radio[1].set_value(kind).run()
    at.sidebar.select_slider[0].set_value(normal_size).run()
    if crash_size is not None:
        at.sidebar.slider(key=f"crash_{normal_size}").set_value(crash_size).run()
    slider_labels = [s.label for s in at.sidebar.slider]
    at.sidebar.slider[slider_labels.index("Generations to follow")].set_value(200).run()
    populations = [s.label for s in at.sidebar.select_slider].index("Populations to simulate")
    at.sidebar.select_slider[populations].set_value(50).run()
    return at


def sample_slider(at: AppTest):
    """
    Finds the sample size slider.
    Parameters:
    - at (AppTest): The running app.
    Returns:
    - The "Sequences sampled per population (n)" slider.
    """
    return next(s for s in at.sidebar.slider if s.label.startswith("Sequences sampled"))


class TestSampleSizeDuringCrash(unittest.TestCase):
    def test_default_crash_at_n_100_runs(self):
        """
        N = 100 crashes to N = 5 by default, which has only 10 gene copies. This used
        to stop the app, because it still tried to sample 20 sequences.
        """
        at = app_with("crash", 100)
        slider = sample_slider(at)
        self.assertEqual(slider.max, 10)
        self.assertLessEqual(slider.value, 10)
        self.assertTrue(any("only 2N = 10 gene copies" in c.value for c in at.sidebar.caption))
        at.button[0].click().run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertIn("Single samples", [t.label for t in at.tabs])

    def test_smallest_crash_runs(self):
        """
        A crash to N = 2 leaves 4 gene copies, the smallest sample allowed, so there is
        nothing to choose: no slider is drawn and n is 4.
        """
        at = app_with("crash", 50, crash_size=2)
        self.assertFalse(any(s.label.startswith("Sequences sampled") for s in at.sidebar.slider))
        self.assertTrue(any("At most 4 sequences" in c.value for c in at.sidebar.caption))
        at.button[0].click().run()
        self.assertEqual([e.value for e in at.exception], [])

    def test_growth_keeps_full_range(self):
        at = app_with("growth", 50)
        self.assertEqual(sample_slider(at).max, 50)
        self.assertFalse(any("gene copies" in c.value for c in at.sidebar.caption))


if __name__ == "__main__":
    unittest.main()
