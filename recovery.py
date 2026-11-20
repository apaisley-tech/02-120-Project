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

RecoveryTimes(gens, event, steady, numSE)
    for every statistic (S, pi, theta_W, D)
        tolerances ← numSE * sqrt(event standard error^2 + steady standard error^2)
        recovery[statistic] ← RecoveryTime(gens, event, steady, tolerances)
    return recovery

CompareHistories(normalSize, histories, mu, n, numTrials, recordEvery, numSE, controlSize)
    (Every history must have the same length; the steady control is simulated once.
    controlSize is normalSize unless given: after growth, the control is a population
    that has always been at the new, larger size.)
    burnIn ← 10 * normalSize copies of normalSize
    controlBurnIn ← 10 * controlSize copies of controlSize
    gens ← 0, recordEvery, 2 * recordEvery, ... up to the history length
    steady ← AverageSampleStats(controlBurnIn + controlSize copies, ...)
             at the end of controlBurnIn + gens
    for every label, eventSizes in histories
        event ← AverageSampleStats(burnIn + eventSizes, ...) at the same generations
        recovery ← RecoveryTimes(gens, event, steady, numSE)
        results[label] ← gens, event, steady, recovery
    return results

RecoveryTimesOverSeeds(normalSize, histories, mu, n, numTrials, recordEvery, seeds, controlSize)
    (Seeds share nothing, so they run at the same time on different CPU cores.)
    for every seed in seeds
        SetSeed(seed)
        results ← CompareHistories(normalSize, histories, ..., controlSize)
        for every statistic and every history
            times[statistic][seed][history] ← the recovery time in results
                                              (undefined if not recovered)
    return times

CompareToSteady(normalSize, eventSizes, mu, n, numTrials, recordEvery, numSE)
    return CompareHistories(normalSize, {one label: eventSizes}, ...) for that label

GrowthRecoveryTimes(startSize, endSizes, gensGrowth, totalGens, mu, n, numTrials,
                    recordEvery, seeds)
    for every endSize in endSizes
        history ← MakeGrowthHistory(startSize, endSize, 0, gensGrowth,
                                    totalGens[endSize] - gensGrowth)
        times for endSize ← RecoveryTimesOverSeeds(startSize, {endSize: history}, ...,
                                                   recordEvery[endSize], seeds,
                                                   controlSize = endSize)
    return the times for every endSize side by side
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import wright_fisher
from diversity import sample_stats_per_trial, summarize_trials


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


def recovery_times(gens: list[int], event: dict, steady: dict,
                   num_se: float = 4) -> dict[str, int | None]:
    """
    Finds the recovery time of every statistic, given averages for an event
    population and a steady control.
    Parameters:
    - gens (list[int]): The recorded generations, in increasing order.
    - event, steady (dict): Averages and standard errors from summarize_trials.
    - num_se (float): How many standard errors apart the event and control may be
      and still count as recovered.
    Returns:
    - dict[str, int | None]: For each of "S", "pi", "theta_w" and "D", the
      recovery time, or None if it has not recovered.
    """
    recovery = {}
    for stat in ["S", "pi", "theta_w", "D"]:
        tolerances = list(num_se * np.hypot(event[stat + "_se"], steady[stat + "_se"]))
        recovery[stat] = recovery_time(gens, event[stat], steady[stat], tolerances)
    return recovery


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
      "steady" (the full averages from average_sample_stats), "event_trials" and
      "steady_trials" (each simulation's values, from sample_stats_per_trial),
      and "recovery" (for each of "S", "pi", "theta_w" and "D", the recovery
      time, or None if it has not recovered).
    """
    results = compare_histories(normal_size, {"event": event_sizes}, mu, sample_size,
                                num_trials, record_every, num_se)
    return results["event"]


def compare_histories(normal_size: int, histories: dict[str, list[int]], mu: float,
                      sample_size: int, num_trials: int, record_every: int,
                      num_se: float = 4, control_size: int | None = None) -> dict[str, dict]:
    """
    Runs compare_to_steady for several events that share one steady control,
    so the control only needs to be simulated once.
    Parameters:
    - normal_size (int): The population size during the events' burn-in, and in
      the control unless control_size is given.
    - histories (dict): A label and the population size in each
      generation after the burn-in, for every event. All must have the same length.
    - mu, sample_size, num_trials, record_every, num_se: As in compare_to_steady.
    - control_size (int | None): The size of the steady control, which gets its own
      burn-in at that size. Defaults to normal_size. After growth, set it to the new
      size, so recovery means looking like a population that was always that large.
    Returns:
    - dict[str, dict]: The compare_to_steady result for each label.
    """
    lengths = {len(sizes) for sizes in histories.values()}
    if len(lengths) != 1:
        raise ValueError("Every history must have the same number of generations.")
    num_gens = lengths.pop()

    if control_size is None:
        control_size = normal_size

    burn_in = [normal_size] * (10 * normal_size)
    control_burn_in = [control_size] * (10 * control_size)
    gens = list(range(0, num_gens + 1, record_every))
    record_gens = [len(burn_in) + t for t in gens]
    control_record_gens = [len(control_burn_in) + t for t in gens]
    steady_trials = sample_stats_per_trial(control_burn_in + [control_size] * num_gens, mu,
                                           sample_size, num_trials, control_record_gens)
    steady = summarize_trials(steady_trials)

    results = {}
    for label, event_sizes in histories.items():
        event_trials = sample_stats_per_trial(burn_in + event_sizes, mu, sample_size,
                                              num_trials, record_gens)
        event = summarize_trials(event_trials)
        recovery = recovery_times(gens, event, steady, num_se)
        results[label] = {"gens": gens, "event": event, "steady": steady,
                          "event_trials": event_trials, "steady_trials": steady_trials,
                          "recovery": recovery}
    return results


def run_in_parallel(func, tasks: list, workers: int | None = None) -> list:
    """
    Runs func on every task, using several processes (one per CPU core) at once.
    Each task must set its own random seed, so the results are the same as
    running the tasks one after another.
    Parameters:
    - func: A function defined at the top level of a module (so other processes
      can find it), taking one task.
    - tasks (list): The inputs, one per call.
    - workers (int | None): How many processes to use. None uses every CPU core;
      1 runs the tasks one after another in this process.
    Returns:
    - list: func(task) for every task, in the order of tasks.
    """
    if workers is None:
        workers = os.cpu_count() or 1
    workers = min(workers, len(tasks))
    if workers <= 1:
        return [func(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, tasks))


def _recovery_for_seed(task: tuple) -> dict:
    """
    Runs one seed's experiment for recovery_times_over_seeds and growth_recovery_times.
    Parameters:
    - task (tuple): seed, normal_size, histories, mu, sample_size, num_trials,
      record_every and control_size, as in recovery_times_over_seeds.
    Returns:
    - dict: For each history label, the recovery time of each statistic (None if
      not recovered).
    """
    (seed, normal_size, histories, mu, sample_size, num_trials, record_every,
     control_size) = task
    wright_fisher.set_seed(seed)
    results = compare_histories(normal_size, histories, mu, sample_size, num_trials,
                                record_every, control_size=control_size)
    return {label: result["recovery"] for label, result in results.items()}


def _times_table(recoveries: list[dict], labels: list) -> dict[str, np.ndarray]:
    """
    Turns per-seed recovery times into one array per statistic.
    Parameters:
    - recoveries (list[dict]): One _recovery_for_seed result per seed.
    - labels (list): The history labels, one column each, in order.
    Returns:
    - dict[str, np.ndarray]: As in recovery_times_over_seeds.
    """
    times = {stat: np.full((len(recoveries), len(labels)), np.nan)
             for stat in ["S", "pi", "theta_w", "D"]}
    for i, recovery in enumerate(recoveries):
        for j, label in enumerate(labels):
            for stat, gen in recovery[label].items():
                if gen is not None:
                    times[stat][i, j] = gen
    return times


def recovery_times_over_seeds(normal_size: int, histories: dict, mu: float,
                              sample_size: int, num_trials: int, record_every: int,
                              seeds: list[int], control_size: int | None = None,
                              workers: int | None = None) -> dict[str, np.ndarray]:
    """
    Measures recovery times for several events, repeating the whole experiment
    with several random seeds to show how much the times vary.
    Parameters:
    - normal_size (int): The population size during the burn-in and in the control.
    - histories (dict): A label and the population size in each generation after
      the burn-in, for every event. All must have the same length.
    - mu, sample_size, num_trials, record_every: As in compare_to_steady.
    - seeds (list[int]): The random seeds, one per repeat of the experiment.
    - control_size (int | None): As in compare_histories.
    - workers (int | None): How many processes run seeds at once (see
      run_in_parallel). The results do not depend on it.
    Returns:
    - dict[str, np.ndarray]: For each of "S", "pi", "theta_w" and "D", an array with
      one row per seed and one column per history (in the order of histories),
      holding the recovery time in generations since the event began, or nan if
      it had not recovered by the end.
    """
    tasks = [(seed, normal_size, histories, mu, sample_size, num_trials, record_every,
              control_size) for seed in seeds]
    return _times_table(run_in_parallel(_recovery_for_seed, tasks, workers), list(histories))


def growth_recovery_times(start_size: int, end_sizes: list[int], gens_growth: int,
                          total_gens: dict[int, int], mu: float, sample_size: int,
                          num_trials: int, record_every: dict[int, int],
                          seeds: list[int], workers: int | None = None) -> dict[str, np.ndarray]:
    """
    Measures how long growth of different sizes leaves a mark, over several seeds.
    Each growing population is compared with a steady control that has always been
    at the new size, so it has recovered once it looks like a population that was
    always that large. Bigger populations take longer to settle, so each growth size
    can be followed for a different number of generations.
    Parameters:
    - start_size (int): The population size before growth.
    - end_sizes (list[int]): The population size after growth, one per experiment.
    - gens_growth (int): The number of generations over which the population grows.
    - total_gens (dict[int, int]): For each end size, how many generations to follow
      after growth begins.
    - mu, sample_size, num_trials: As in compare_to_steady.
    - record_every (dict[int, int]): For each end size, how many generations apart to sample.
    - seeds (list[int]): The random seeds, one per repeat of the experiment.
    - workers (int | None): As in recovery_times_over_seeds.
    Returns:
    - dict[str, np.ndarray]: As in recovery_times_over_seeds, with one column per end
      size: the recovery time in generations since growth began, or nan if it had
      not recovered by the end.
    """
    # Every (end size, seed) pair is its own task, so all CPU cores stay busy.
    tasks = []
    for end_size in end_sizes:
        history = wright_fisher.make_growth_history(start_size, end_size, gens_before=0,
                                                    gens_growth=gens_growth,
                                                    gens_after=total_gens[end_size] - gens_growth)
        tasks += [(seed, start_size, {end_size: history}, mu, sample_size, num_trials,
                   record_every[end_size], end_size) for seed in seeds]
    recoveries = run_in_parallel(_recovery_for_seed, tasks, workers)

    columns = []
    for j, end_size in enumerate(end_sizes):
        per_seed = recoveries[j * len(seeds):(j + 1) * len(seeds)]
        columns.append(_times_table(per_seed, [end_size]))
    return {stat: np.hstack([column[stat] for column in columns])
            for stat in ["S", "pi", "theta_w", "D"]}


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
