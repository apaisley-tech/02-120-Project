"""
Measures how long a population size event leaves a mark on genetic variation.

Every scenario starts with a burn-in at a steady size so that variation reaches
equilibrium, then follows an event history (a crash or growth). We track the
average of a statistic (Tajima's D or pi) over many runs, and compare it with
the same statistic in a steady control population. The recovery time is the
first generation after which the event population can no longer be told apart
from the control: the difference stays within a few standard errors from then on.
We use 4 standard errors by default: with about 100 sampled generations, a 3
standard error cutoff would flag a false "not yet recovered" point by chance in
roughly 1 in 4 scenarios.

Because the cutoff uses standard errors, running more trials can detect smaller
leftover effects, so recovery times should be compared at the same num_trials.

Pseudocode

RecoveryTime(gens, values, baseline, tolerances)
    for i from the last recorded generation down to the first
        if |values[i] - baseline[i]| > tolerances[i]
            if i is the last recorded generation
                return not recovered
            return gens[i + 1]
    return gens[0]

CompareHistories(normalSize, histories, mu, n, numTrials, recordEvery, numSE)
    (Every history must have the same length; the steady control is simulated once.)
    burnIn ← 10 * normalSize copies of normalSize
    gens ← 0, recordEvery, 2 * recordEvery, ... up to the history length
    steady ← AverageSampleStats(burnIn + normalSize copies, ...) at the end of burnIn + gens
    for every label, eventSizes in histories
        event ← AverageSampleStats(burnIn + eventSizes, ...) at the same generations
        for every statistic (S, pi, theta_W, D)
            tolerances ← numSE * sqrt(event standard error^2 + steady standard error^2)
            recovery[statistic] ← RecoveryTime(gens, event, steady, tolerances)
        results[label] ← gens, event, steady, recovery
    return results

RecoveryTimesOverSeeds(normalSize, crashSizes, gensCrash, gensAfter, mu, n, numTrials,
                       recordEvery, seeds)
    for every seed in seeds
        SetSeed(seed)
        histories ← a crash history for every crashSize in crashSizes
        results ← CompareHistories(normalSize, histories, ...)
        for every statistic and crashSize
            times[statistic][seed][crashSize] ← the recovery time in results
                                                (undefined if not recovered)
    return times

CompareToSteady(normalSize, eventSizes, mu, n, numTrials, recordEvery, numSE)
    return CompareHistories(normalSize, {one label: eventSizes}, ...) for that label
"""

import numpy as np

import wright_fisher
from diversity import average_sample_stats


def recovery_time(gens: list[int], values: list[float], baseline: list[float],
                  tolerances: list[float]) -> int | None:
    """
    Finds the first generation after which values stay close to the baseline.
    Parameters:
    - gens (list[int]): The recorded generations, in increasing order.
    - values (list[float]): The statistic in the event population at each generation.
    - baseline (list[float]): The statistic in the control population at each generation.
    - tolerances (list[float]): How far values may be from baseline at each generation
      and still count as recovered.
    Returns:
    - int | None: The first generation from which every later value is within
      tolerance, or None if the last value is still outside it (not recovered yet).
    """
    if not len(gens) == len(values) == len(baseline) == len(tolerances):
        raise ValueError("gens, values, baseline and tolerances must have the same length.")

    for i in range(len(gens) - 1, -1, -1):
        if abs(values[i] - baseline[i]) > tolerances[i]:
            if i == len(gens) - 1:
                return None
            return gens[i + 1]
    return gens[0]


def compare_to_steady(normal_size: int, event_sizes: list[int], mu: float,
                      sample_size: int, num_trials: int, record_every: int,
                      num_se: float = 4) -> dict:
    """
    Simulates an event and a steady control population, both after a burn-in,
    and measures how long the event's effect on each statistic lasts.
    Parameters:
    - normal_size (int): The population size during the burn-in and in the control.
    - event_sizes (list[int]): The population size in each generation after the burn-in.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - num_trials (int): The number of independent simulations per population.
    - record_every (int): How many generations apart to sample.
    - num_se (float): How many standard errors apart the event and control may be
      and still count as recovered.
    Returns:
    - dict: "gens" (generations after the burn-in that were sampled), "event" and
      "steady" (the full averages from average_sample_stats), and "recovery"
      (for each of "S", "pi", "theta_w" and "D", the recovery time, or None
      if it has not recovered).
    """
    results = compare_histories(normal_size, {"event": event_sizes}, mu, sample_size,
                                num_trials, record_every, num_se)
    return results["event"]


def compare_histories(normal_size: int, histories: dict[str, list[int]], mu: float,
                      sample_size: int, num_trials: int, record_every: int,
                      num_se: float = 4) -> dict[str, dict]:
    """
    Runs compare_to_steady for several events that share one steady control,
    so the control only needs to be simulated once.
    Parameters:
    - normal_size (int): The population size during the burn-in and in the control.
    - histories (dict): A label and the population size in each
      generation after the burn-in, for every event. All must have the same length.
    - mu, sample_size, num_trials, record_every, num_se: As in compare_to_steady.
    Returns:
    - dict[str, dict]: The compare_to_steady result for each label.
    """
    lengths = {len(sizes) for sizes in histories.values()}
    if len(lengths) != 1:
        raise ValueError("Every history must have the same number of generations.")
    num_gens = lengths.pop()

    burn_in = [normal_size] * (10 * normal_size)
    gens = list(range(0, num_gens + 1, record_every))
    record_gens = [len(burn_in) + t for t in gens]
    steady = average_sample_stats(burn_in + [normal_size] * num_gens, mu, sample_size,
                                  num_trials, record_gens)

    results = {}
    for label, event_sizes in histories.items():
        event = average_sample_stats(burn_in + event_sizes, mu, sample_size, num_trials,
                                     record_gens)
        recovery = {}
        for stat in ["S", "pi", "theta_w", "D"]:
            tolerances = list(num_se * np.hypot(event[stat + "_se"], steady[stat + "_se"]))
            recovery[stat] = recovery_time(gens, event[stat], steady[stat], tolerances)
        results[label] = {"gens": gens, "event": event, "steady": steady, "recovery": recovery}
    return results


def recovery_times_over_seeds(normal_size: int, crash_sizes: list[int], gens_crash: int,
                              gens_after: int, mu: float, sample_size: int, num_trials: int,
                              record_every: int, seeds: list[int]) -> dict[str, np.ndarray]:
    """
    Measures recovery times after crashes of several sizes, repeating the whole
    experiment with several random seeds to show how much the times vary.
    Parameters:
    - normal_size (int): The population size before and after each crash.
    - crash_sizes (list[int]): The population sizes during the crash to compare.
    - gens_crash (int): The number of generations each crash lasts.
    - gens_after (int): The number of generations simulated after the crash ends.
    - mu, sample_size, num_trials, record_every: As in compare_to_steady.
    - seeds (list[int]): The random seeds, one per repeat of the experiment.
    Returns:
    - dict[str, np.ndarray]: For each of "S", "pi", "theta_w" and "D", an array with
      one row per seed and one column per crash size, holding the recovery time in
      generations since the crash began, or nan if it had not recovered by the end.
    """
    times = {stat: np.full((len(seeds), len(crash_sizes)), np.nan)
             for stat in ["S", "pi", "theta_w", "D"]}
    for i, seed in enumerate(seeds):
        wright_fisher.set_seed(seed)
        histories = {crash_size: wright_fisher.make_crash_history(
                         normal_size, crash_size, gens_before=0, gens_crash=gens_crash,
                         gens_after=gens_after)
                     for crash_size in crash_sizes}
        results = compare_histories(normal_size, histories, mu, sample_size, num_trials,
                                    record_every)
        for j, crash_size in enumerate(crash_sizes):
            for stat, gen in results[crash_size]["recovery"].items():
                if gen is not None:
                    times[stat][i, j] = gen
    return times


def main():
    wright_fisher.set_seed(120)

    normal_size, mu, sample_size = 200, 0.0125, 20
    gens_after = 2000

    print(f"N = {normal_size}, theta = 4N * mu = {4 * normal_size * mu:.0f}, n = {sample_size}, "
          f"200 runs each; crashes last 20 generations")
    print(f"{'crash size':>10} {'peak D':>7} {'lowest D':>9} {'D recovers':>11} {'pi recovers':>12}")
    histories = {crash_size: wright_fisher.make_crash_history(normal_size, crash_size,
                                                              gens_before=0, gens_crash=20,
                                                              gens_after=gens_after)
                 for crash_size in [100, 50, 20, 10]}
    results = compare_histories(normal_size, histories, mu, sample_size, num_trials=200,
                                record_every=20)
    for crash_size, result in results.items():
        d = result["event"]["D"]
        print(f"{crash_size:>10} {max(d):>7.2f} {min(d):>9.2f} "
              f"{str(result['recovery']['D']):>11} {str(result['recovery']['pi']):>12}")


if __name__ == "__main__":
    main()
