"""
Works backwards: from one sample of DNA sequences, infers what happened to the
population it came from (steady, a crash or growth), how severe the event was and
how long ago it began.

There is no formula for the chance of a sample's statistics after a crash, but the
simulation can produce examples. So we use Approximate Bayesian Computation (ABC)
by rejection:
1. Build a reference table: simulate many populations under a grid of possible
   histories, and record the statistics of one sample from each population every
   few generations, next to the history and time that produced them.
2. Put the observed statistics and every row of the table on a common scale and
   measure how far apart they are.
3. Keep the rows closest to the observation. The histories in those rows show
   which histories could have produced the sample, and how likely each is.

We compare three statistics of a sample: pi (driven by variants at middle
frequencies), S (all variable sites) and the number of singletons (sites carried
by one sampled sequence, the rarest variants). Together they summarize the shape
of the site frequency spectrum, which is where crashes and growth leave their mark.
Tajima's D is a function of pi and S, so it adds nothing new.

Steady, crash and growth start with equal prior odds (1/3 each). Within crashes
and growth, every grid point and every recorded time is equally likely.

Pseudocode

BuildReferenceTable(normalSize, mu, n, crashSizes, crashLengths, growthSizes,
                    growthLength, maxTime, recordEvery, numTrials, seed)
    histories ← steady, every crash (size, length), every growth (end size)
    for every history h (all at once, on different CPU cores)
        SetSeed(seed + h)
        stats ← SampleStatsPerTrial(burnIn + history h, ..., at times 0, recordEvery, ... maxTime)
        for every population and every time
            add a row: kind, crash size, length, end size, time, pi, S, singletons,
                       and which population it came from
    return the rows

Infer(observed, table, numKeep)
    scale ← the standard deviation of each statistic among steady rows
    for every row r
        distance[r] ← sqrt of the sum over statistics of ((r - observed) / scale)^2
    keep the numKeep rows with the smallest distance
    weight each kept row by its kind's prior weight (1/3 split over that kind's rows)
    return the kept rows and their weights

Summarize(kept, weights)
    P(kind) ← the share of weight on each kind
    for crashes: the weighted median and 90% range of strength = length / (2 * size)
                 and of the time since the crash began
    for growth: the same for the size after growth and the time since growth began

Validate(table, numTests)
    for numTests random rows
        remove every row from the same simulated population (so no near-copy remains)
        Infer from the rest, and compare the answer with the row's true history
    report how often the kind is right and how close strength and time are
"""

import inspect
import os
import sys

import numpy as np

import wright_fisher
from diversity import sample_stats_per_trial
from linkage import linked_sample_stats
from recovery import run_in_parallel

STATS = ["pi", "S", "singletons"]
KINDS = ["steady", "crash", "growth"]
CACHE_DIR = "inference_cache"
# Crash strength classes for reporting: (lowest, highest, name).
STRENGTH_CLASSES = [(0, 0.1, "weak (strength < 0.1)"), (0.1, 0.4, "moderate (0.1-0.4)"),
                    (0.4, 1.01, "strong (0.4-1)"), (1.01, np.inf, "severe (> 1)")]


def make_histories(normal_size: int, crash_sizes: list[int], crash_lengths: list[int],
                   growth_sizes: list[int], growth_length: int, max_time: int) -> list[dict]:
    """
    Lists the possible histories of the reference table.
    Parameters:
    - normal_size (int): The population size before any event.
    - crash_sizes (list[int]): The population sizes a crash can drop to.
    - crash_lengths (list[int]): How many generations a crash can last.
    - growth_sizes (list[int]): The population sizes growth can end at.
    - growth_length (int): How many generations growth takes.
    - max_time (int): How many generations to follow after each event begins.
    Returns:
    - list[dict]: One dict per history with "kind", "crash_size", "crash_length",
      "end_size" (0 where they do not apply) and "sizes" (the population size in
      each generation after the burn-in).
    """
    histories = [{"kind": "steady", "crash_size": 0, "crash_length": 0, "end_size": 0,
                  "sizes": [normal_size] * max_time}]
    for size in crash_sizes:
        for length in crash_lengths:
            histories.append({
                "kind": "crash", "crash_size": size, "crash_length": length, "end_size": 0,
                "sizes": wright_fisher.make_crash_history(normal_size, size, 0, length,
                                                          max_time - length)})
    for end_size in growth_sizes:
        histories.append({
            "kind": "growth", "crash_size": 0, "crash_length": 0, "end_size": end_size,
            "sizes": wright_fisher.make_growth_history(normal_size, end_size, 0, growth_length,
                                                       max_time - growth_length)})
    return histories


def _simulate_history(task: tuple) -> dict[str, np.ndarray]:
    """
    Simulates one history of the reference table (run by run_in_parallel).
    Parameters:
    - task (tuple): seed, normal_size, sizes, mu, sample_size, num_trials, times, linked.
    Returns:
    - dict[str, np.ndarray]: For each statistic in STATS, one row per population
      and one column per time.
    """
    seed, normal_size, sizes, mu, sample_size, num_trials, times, linked = task
    wright_fisher.set_seed(seed)
    burn_in = [normal_size] * (10 * normal_size)
    simulate = linked_sample_stats if linked else sample_stats_per_trial
    stats = simulate(burn_in + sizes, mu, sample_size, num_trials,
                     [len(burn_in) + t for t in times])
    return {name: stats[name] for name in STATS}


def build_reference_table(normal_size: int = 200, mu: float = 0.0125, sample_size: int = 20,
                          crash_sizes: tuple = (10, 20, 40, 100),
                          crash_lengths: tuple = (5, 10, 20, 40),
                          growth_sizes: tuple = (400, 1000, 2000, 4000),
                          growth_length: int = 20, max_time: int = 1000,
                          record_every: int = 10, num_trials: int = 200, seed: int = 1,
                          workers: int | None = None,
                          linked: bool = False) -> dict[str, np.ndarray]:
    """
    Simulates every history in the grid and records one sample per population at
    every recorded time.
    Parameters:
    - normal_size (int): The population size before any event (and of steady rows).
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - crash_sizes, crash_lengths, growth_sizes, growth_length: The grid of histories
      (see make_histories). Crash sizes must be at least n / 2, so that a sample of
      n sequences can be taken during the crash.
    - max_time (int): How many generations after the event began to follow.
    - record_every (int): How many generations apart to sample.
    - num_trials (int): Populations simulated per history.
    - seed (int): History h uses seed + h, so the table is reproducible.
    - workers (int | None): As in run_in_parallel.
    - linked (bool): Simulate completely linked sites (linkage.linked_sample_stats)
      instead of independent ones. Each recorded time then has its own family tree.
    Returns:
    - dict[str, np.ndarray]: One entry per row for "kind" (index into KINDS),
      "crash_size", "crash_length", "end_size", "time" (generations since the event
      began; nan for steady rows), "population" (which simulated population the row
      came from) and every statistic in STATS. "settings" holds the arguments.
    """
    if any(2 * size < sample_size for size in crash_sizes):
        raise ValueError("Every crash size must be at least sample_size / 2.")
    if any(length >= max_time for length in crash_lengths) or growth_length >= max_time:
        raise ValueError("Events must be shorter than max_time.")

    histories = make_histories(normal_size, list(crash_sizes), list(crash_lengths),
                               list(growth_sizes), growth_length, max_time)
    times = list(range(0, max_time + 1, record_every))
    tasks = [(seed + h, normal_size, history["sizes"], mu, sample_size, num_trials, times,
              linked) for h, history in enumerate(histories)]
    results = run_in_parallel(_simulate_history, tasks, workers)

    rows_per_history = num_trials * len(times)
    table = {}
    for name in ["kind", "crash_size", "crash_length", "end_size"]:
        values = [KINDS.index(h["kind"]) if name == "kind" else h[name] for h in histories]
        table[name] = np.repeat(values, rows_per_history)
    time = np.tile(np.array(times, dtype=float), num_trials * len(histories))
    table["time"] = np.where(table["kind"] == KINDS.index("steady"), np.nan, time)
    table["population"] = np.repeat(np.arange(num_trials * len(histories)), len(times))
    for name in STATS:
        # Each result has one row per population; ravel keeps a population's times together.
        table[name] = np.concatenate([result[name].ravel() for result in results])
    table["settings"] = np.array([normal_size, mu, sample_size, max_time, record_every,
                                  num_trials, seed], dtype=float)
    return table


def cached_reference_table(**kwargs) -> dict[str, np.ndarray]:
    """
    Loads a reference table from inference_cache/ if one was built with the same
    arguments, and otherwise builds it and saves it there.
    Parameters:
    - kwargs: As in build_reference_table (except workers, which does not change it).
    Returns:
    - dict[str, np.ndarray]: The table.
    """
    workers = kwargs.pop("workers", None)
    # Include the defaults too, so changing a default never loads a stale table.
    settings = inspect.signature(build_reference_table).bind(**kwargs)
    settings.apply_defaults()
    settings = {name: value for name, value in settings.arguments.items() if name != "workers"}
    # Leave linked out of the name when False, so tables saved before it existed still load.
    if not settings["linked"]:
        del settings["linked"]
    key = "_".join(f"{name}-{settings[name]}" for name in sorted(settings))
    key = key.replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")
    path = os.path.join(CACHE_DIR, f"table_{key}.npz")
    if os.path.exists(path):
        with np.load(path) as saved:
            return dict(saved)
    table = build_reference_table(**settings, workers=workers)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **table)
    return table


def steady_scale(table: dict[str, np.ndarray]) -> np.ndarray:
    """
    Finds how much each statistic varies among steady populations, used to put
    the statistics on a common scale.
    Parameters:
    - table (dict): A reference table.
    Returns:
    - np.ndarray: The standard deviation of each statistic in STATS among steady rows.
    """
    steady = table["kind"] == KINDS.index("steady")
    return np.array([np.std(table[name][steady]) for name in STATS])


def infer(observed: dict[str, float], table: dict[str, np.ndarray], num_keep: int = 500,
          exclude: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """
    Keeps the reference rows whose statistics are closest to the observed sample.
    Parameters:
    - observed (dict[str, float]): The observed value of every statistic in STATS.
    - table (dict): A reference table.
    - num_keep (int): How many of the closest rows to keep.
    - exclude (np.ndarray | None): True for rows that may not be kept (used in
      validation to remove the tested population).
    Returns:
    - dict[str, np.ndarray]: The kept rows (every column of the table except
      "settings"), their "distance" and "weight": the prior weight of the row's kind,
      normalized so the weights add up to 1.
    """
    scale = steady_scale(table)
    distance = np.sqrt(sum(((table[name] - observed[name]) / s) ** 2
                           for name, s in zip(STATS, scale)))
    if exclude is not None:
        distance = np.where(exclude, np.inf, distance)
    keep = np.argpartition(distance, num_keep)[:num_keep]
    kept = {name: values[keep] for name, values in table.items() if name != "settings"}
    kept["distance"] = distance[keep]

    # Equal prior odds for each kind: each kind's prior is spread over its rows.
    rows_of_kind = np.bincount(table["kind"], minlength=len(KINDS))
    with np.errstate(divide="ignore"):
        prior = np.where(rows_of_kind > 0, 1 / rows_of_kind, 0)
    weight = prior[kept["kind"]]
    kept["weight"] = weight / weight.sum()
    return kept


def weighted_quantiles(values: np.ndarray, weights: np.ndarray,
                       quantiles: list[float]) -> list[float]:
    """
    Finds quantiles of values in which each value counts with its weight.
    Parameters:
    - values (np.ndarray): The values.
    - weights (np.ndarray): A non-negative weight for each value.
    - quantiles (list[float]): The quantiles wanted, each between 0 and 1.
    Returns:
    - list[float]: One value per quantile, or nan for each if the weights add to 0.
    """
    if len(values) == 0 or weights.sum() <= 0:
        return [np.nan] * len(quantiles)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return [float(values[order][min(np.searchsorted(cumulative, q), len(values) - 1)])
            for q in quantiles]


def summarize(kept: dict[str, np.ndarray]) -> dict:
    """
    Summarizes the kept rows into probabilities and estimates.
    Parameters:
    - kept (dict): The output of infer.
    Returns:
    - dict: "probability" (of each kind), and for "crash" and "growth" (if they
      have any weight) the median and 90% range of the estimates: for a crash,
      "strength" (length / (2 * size)), "crash_size", "crash_length" and "time";
      for growth, "end_size" and "time". Each estimate is (low, median, high).
    """
    weight = kept["weight"]
    summary = {"probability": {kind: float(weight[kept["kind"] == k].sum())
                               for k, kind in enumerate(KINDS)}}
    for kind, names in [("crash", ["strength", "crash_size", "crash_length", "time"]),
                        ("growth", ["end_size", "time"])]:
        rows = kept["kind"] == KINDS.index(kind)
        if weight[rows].sum() <= 0:
            continue
        estimates = {}
        for name in names:
            if name == "strength":
                values = kept["crash_length"][rows] / (2 * kept["crash_size"][rows])
            else:
                values = kept[name][rows]
            estimates[name] = tuple(weighted_quantiles(values.astype(float), weight[rows],
                                                       [0.05, 0.5, 0.95]))
        summary[kind] = estimates
    return summary


def describe(summary: dict) -> str:
    """
    Writes a summary in words.
    Parameters:
    - summary (dict): The output of summarize.
    Returns:
    - str: A few lines of text.
    """
    p = summary["probability"]
    lines = [f"P(steady) = {p['steady']:.2f}, P(crash) = {p['crash']:.2f}, "
             f"P(growth) = {p['growth']:.2f}"]
    if "crash" in summary and p["crash"] > 0.05:
        c = summary["crash"]
        lines.append(f"  If a crash: strength {c['strength'][1]:.2f} "
                     f"(90%: {c['strength'][0]:.2f} to {c['strength'][2]:.2f}), "
                     f"began {c['time'][1]:.0f} generations ago "
                     f"(90%: {c['time'][0]:.0f} to {c['time'][2]:.0f})")
    if "growth" in summary and p["growth"] > 0.05:
        g = summary["growth"]
        lines.append(f"  If growth: to N = {g['end_size'][1]:.0f} "
                     f"(90%: {g['end_size'][0]:.0f} to {g['end_size'][2]:.0f}), "
                     f"began {g['time'][1]:.0f} generations ago "
                     f"(90%: {g['time'][0]:.0f} to {g['time'][2]:.0f})")
    return "\n".join(lines)


def observe(sizes: list[int], normal_size: int, mu: float, sample_size: int,
            seed: int) -> dict[str, float]:
    """
    Simulates one population with a known history and takes one sample at the end,
    as a test observation.
    Parameters:
    - sizes (list[int]): The population size in each generation after the burn-in.
    - normal_size (int): The burn-in population size.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - seed (int): The random seed.
    Returns:
    - dict[str, float]: The sample's value of every statistic in STATS.
    """
    wright_fisher.set_seed(seed)
    burn_in = [normal_size] * (10 * normal_size)
    stats = sample_stats_per_trial(burn_in + sizes, mu, sample_size, 1,
                                   [len(burn_in) + len(sizes)])
    return {name: float(stats[name][0, 0]) for name in STATS}


def validate(table: dict[str, np.ndarray], num_tests: int = 1000, num_keep: int = 500,
             seed: int = 0, tests: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """
    Tests the inference on rows of the table whose history is known. Every row from
    the tested population is removed first, so no near-copy of the test is left.
    Parameters:
    - table (dict): A reference table.
    - num_tests (int): How many rows to test.
    - num_keep (int): As in infer.
    - seed (int): Chooses the test rows.
    - tests (dict | None): A second table to take the test rows from, for example one
      simulated under a different model than the reference table. Its populations
      are separate from the reference table's, so nothing is removed. Defaults to table.
    Returns:
    - dict[str, np.ndarray]: One entry per test: the true "kind", "time", "strength"
      (nan unless a crash) and "end_size" (0 unless growth), the most probable kind
      ("guess"), the probability given to the true kind ("p_true"), and for crashes
      and growth the median estimates "strength_estimate", "time_estimate" and
      "end_size_estimate" and whether the truth was in the 90% range
      ("strength_covered", "time_covered").
    """
    source = table if tests is None else tests
    rng = np.random.default_rng(seed)
    # Equal numbers of steady, crash and growth tests.
    rows = np.concatenate([rng.choice(np.flatnonzero(source["kind"] == k),
                                      num_tests // len(KINDS), replace=False)
                           for k in range(len(KINDS))])
    names = ["kind", "guess", "p_true", "time", "strength", "end_size", "strength_estimate",
             "time_estimate", "end_size_estimate", "strength_covered", "time_covered"]
    results = {name: np.full(len(rows), np.nan) for name in names}
    for i, row in enumerate(rows):
        observed = {name: source[name][row] for name in STATS}
        exclude = None if tests is not None else table["population"] == table["population"][row]
        summary = summarize(infer(observed, table, num_keep, exclude=exclude))
        kind = KINDS[source["kind"][row]]
        probabilities = summary["probability"]
        results["kind"][i] = source["kind"][row]
        results["guess"][i] = KINDS.index(max(probabilities, key=probabilities.get))
        results["p_true"][i] = probabilities[kind]
        results["time"][i] = source["time"][row]
        results["end_size"][i] = source["end_size"][row]
        if kind == "crash":
            strength = source["crash_length"][row] / (2 * source["crash_size"][row])
            results["strength"][i] = strength
            if "crash" in summary:
                low, median, high = summary["crash"]["strength"]
                results["strength_estimate"][i] = median
                results["strength_covered"][i] = low <= strength <= high
        if kind in ("crash", "growth") and kind in summary:
            low, median, high = summary[kind]["time"]
            results["time_estimate"][i] = median
            results["time_covered"][i] = low <= source["time"][row] <= high
            if kind == "growth":
                results["end_size_estimate"][i] = summary["growth"]["end_size"][1]
    return results


def validation_rates(results: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Picks the headline numbers out of validate's results.
    Parameters:
    - results (dict): The output of validate.
    Returns:
    - dict[str, float]: Shares from 0 to 1: "steady", "crash" and "growth" (the right
      kind was most probable), "severe_early" (severe crashes, strength above 1,
      in the first 300 generations, called crashes), "strength_covered",
      "crash_time_covered" and "growth_time_covered" (the truth was inside the 90% range).
    """
    rates = {kind: float(np.mean(results["guess"][results["kind"] == k] == k))
             for k, kind in enumerate(KINDS)}
    severe = (results["kind"] == 1) & (results["strength"] > 1) & (results["time"] < 300)
    rates["severe_early"] = float(np.mean(results["guess"][severe] == 1))
    crash = results["kind"] == 1
    rates["strength_covered"] = float(np.nanmean(results["strength_covered"][crash]))
    rates["crash_time_covered"] = float(np.nanmean(results["time_covered"][crash]))
    rates["growth_time_covered"] = float(np.nanmean(results["time_covered"][results["kind"] == 2]))
    return rates


def describe_validation(results: dict[str, np.ndarray]) -> str:
    """
    Summarizes validate's results in words.
    Parameters:
    - results (dict): The output of validate.
    Returns:
    - str: A few lines of text.
    """
    lines = []
    for k, kind in enumerate(KINDS):
        rows = results["kind"] == k
        lines.append(f"{kind:>7}: right kind {np.mean(results['guess'][rows] == k):.0%} of the "
                     f"time (average probability on the truth {np.mean(results['p_true'][rows]):.2f})")
    # Mild crashes look almost exactly like steady populations, so crash detection
    # is shown separately for each strength, early and late.
    crash_rows = results["kind"] == 1
    for low, high, name in STRENGTH_CLASSES:
        cells = []
        for start, end in [(0, 300), (300, 600), (600, 1001)]:
            window = (crash_rows & (results["strength"] >= low) & (results["strength"] < high)
                      & (results["time"] >= start) & (results["time"] < end))
            right = f"{np.mean(results['guess'][window] == 1):.0%}" if window.any() else "-"
            cells.append(f"{start}-{min(end, 1000)}: {right:>4}")
        lines.append(f"  {name:>27} crash, right kind by generations ago: " + ", ".join(cells))
    growth_rows = results["kind"] == 2
    cells = []
    for start, end in [(0, 300), (300, 600), (600, 1001)]:
        window = growth_rows & (results["time"] >= start) & (results["time"] < end)
        cells.append(f"{start}-{min(end, 1000)}: {np.mean(results['guess'][window] == 2):.0%}")
    lines.append(f"  {'growth':>27}, right kind by generations ago: " + ", ".join(cells))
    crash = (results["kind"] == 1) & ~np.isnan(results["strength_estimate"])
    log_error = np.abs(np.log2(results["strength_estimate"][crash] / results["strength"][crash]))
    lines.append(f"Crash strength: median estimate within a factor of "
                 f"{2 ** np.median(log_error):.2f} of the truth; truth inside the 90% range "
                 f"{np.nanmean(results['strength_covered'][crash]):.0%} of the time")
    for kind, k in [("crash", 1), ("growth", 2)]:
        rows = (results["kind"] == k) & ~np.isnan(results["time_estimate"])
        error = np.abs(results["time_estimate"][rows] - results["time"][rows])
        lines.append(f"{kind.capitalize()} time: median error {np.median(error):.0f} generations; "
                     f"truth inside the 90% range {np.nanmean(results['time_covered'][rows]):.0%} "
                     "of the time")
    return "\n".join(lines)


def main():
    normal_size, mu, sample_size = 200, 0.0125, 20
    print("Building the reference table (cached in inference_cache/ after the first run)...")
    table = cached_reference_table(normal_size=normal_size, mu=mu, sample_size=sample_size)
    print(f"  {len(table['kind'])} reference samples")

    mysteries = {
        "steady": [normal_size] * 300,
        "crash to N = 20 for 20 generations, 200 generations ago":
            wright_fisher.make_crash_history(normal_size, 20, 0, 20, 180),
        "crash to N = 10 for 10 generations, 50 generations ago":
            wright_fisher.make_crash_history(normal_size, 10, 0, 10, 40),
        "growth to N = 2000 over 20 generations, 300 generations ago":
            wright_fisher.make_growth_history(normal_size, 2000, 0, 20, 280),
    }
    print("\nOne sample from each mystery population (seeds not used in the table):")
    for i, (label, sizes) in enumerate(mysteries.items()):
        observed = observe(sizes, normal_size, mu, sample_size, seed=10_000 + i)
        print(f"\nTruth: {label}")
        print(f"  observed pi = {observed['pi']:.2f}, S = {observed['S']:.0f}, "
              f"singletons = {observed['singletons']:.0f}")
        print("  " + describe(summarize(infer(observed, table))).replace("\n", "\n  "))

    print("\nValidation on 1500 samples with known histories:")
    print(describe_validation(validate(table, num_tests=1500)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("usage: python3 inference.py")
    main()
