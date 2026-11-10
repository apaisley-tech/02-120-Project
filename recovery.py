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

CompareToSteady(normalSize, eventSizes, mu, n, numTrials, recordEvery, numSE)
    burnIn ← 10 * normalSize copies of normalSize
    gens ← 0, recordEvery, 2 * recordEvery, ... up to len(eventSizes)
    event ← AverageSampleStats(burnIn + eventSizes, ...) at the end of burnIn + gens
    steady ← AverageSampleStats(burnIn + normalSize copies, ...) at the same generations
    for every statistic (S, pi, theta_W, D)
        tolerances ← numSE * sqrt(event standard error^2 + steady standard error^2)
        recovery[statistic] ← RecoveryTime(gens, event, steady, tolerances)
    return gens, event, steady, recovery
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
    burn_in = [normal_size] * (10 * normal_size)
    gens = list(range(0, len(event_sizes) + 1, record_every))
    record_gens = [len(burn_in) + t for t in gens]

    event = average_sample_stats(burn_in + event_sizes, mu, sample_size, num_trials, record_gens)
    steady = average_sample_stats(burn_in + [normal_size] * len(event_sizes), mu, sample_size,
                                  num_trials, record_gens)

    recovery = {}
    for stat in ["S", "pi", "theta_w", "D"]:
        tolerances = list(num_se * np.hypot(event[stat + "_se"], steady[stat + "_se"]))
        recovery[stat] = recovery_time(gens, event[stat], steady[stat], tolerances)
    return {"gens": gens, "event": event, "steady": steady, "recovery": recovery}


def main():
    wright_fisher.set_seed(120)

    normal_size, mu, sample_size = 200, 0.0125, 20
    gens_after = 2000

    print(f"N = {normal_size}, theta = 4N * mu = {4 * normal_size * mu:.0f}, n = {sample_size}, "
          f"200 runs each; crashes last 20 generations")
    print(f"{'crash size':>10} {'peak D':>7} {'lowest D':>9} {'D recovers':>11} {'pi recovers':>12}")
    for crash_size in [100, 50, 20, 10]:
        sizes = wright_fisher.make_crash_history(normal_size, crash_size, gens_before=0,
                                                 gens_crash=20, gens_after=gens_after)
        result = compare_to_steady(normal_size, sizes, mu, sample_size, num_trials=200,
                                   record_every=20)
        d = result["event"]["D"]
        print(f"{crash_size:>10} {max(d):>7.2f} {min(d):>9.2f} "
              f"{str(result['recovery']['D']):>11} {str(result['recovery']['pi']):>12}")


if __name__ == "__main__":
    main()
