"""
Basic Wright-Fisher simulation of genetic drift at one locus with two alleles.

Pseudocode

WrightFisherGeneration(popSize, p)
    numCopies ← 2 * popSize
    count ← RandBinomial(numCopies, p)    (same as counting numCopies draws with RandFloat() < p)
    return count / numCopies

SimulateDrift(popSizes, p0)
    freqs ← [p0]
    p ← p0
    for every popSize in popSizes
        p ← WrightFisherGeneration(popSize, p)
        append p to freqs
    return freqs

MakeCrashHistory(normalSize, crashSize, gensBefore, gensCrash, gensAfter)
    return gensBefore copies of normalSize, then gensCrash copies of crashSize,
           then gensAfter copies of normalSize

MakeGrowthHistory(startSize, endSize, gensBefore, gensGrowth, gensAfter)
    sizes ← gensBefore copies of startSize
    for t ← 1 to gensGrowth
        append startSize * (endSize / startSize)^(t / gensGrowth), rounded
    append gensAfter copies of endSize
    return sizes

AverageHeterozygosity(popSizes, p0, numTrials)
    totals ← list of zeros, one per generation
    for numTrials total trials
        freqs ← SimulateDrift(popSizes, p0)
        for every generation t
            totals[t] ← totals[t] + 2 * freqs[t] * (1 - freqs[t])
    return totals / numTrials

RunUntilFixedOrLost(popSize, p0)
    p ← p0
    while 0 < p < 1
        p ← WrightFisherGeneration(popSize, p)
    return p = 1

EstimateFixationProbability(popSize, p0, numTrials)
    count ← 0
    for numTrials total trials
        if RunUntilFixedOrLost(popSize, p0)
            count ← count + 1
    return count / numTrials
"""

import numpy as np

rng = np.random.default_rng()


def set_seed(seed: int) -> None:
    """
    Resets the random number generator so that simulations are reproducible.
    Parameters:
    - seed (int): The seed for the new generator.
    """
    global rng
    rng = np.random.default_rng(seed)


def wright_fisher_generation(pop_size: int, p: float) -> float:
    """
    Produces the next generation's allele frequency under Wright-Fisher sampling.
    Parameters:
    - pop_size (int): The number of diploid individuals N, so there are 2N gene copies.
    - p (float): The current frequency of the allele, between 0 and 1.
    Returns:
    - float: The allele frequency in the next generation.
    """
    if pop_size <= 0:
        raise ValueError("pop_size must be a positive integer.")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1.")

    # Each of the 2N copies is independently the allele with probability p,
    # so the count is one binomial draw instead of a loop of 2N draws.
    num_copies = 2 * pop_size
    count = int(rng.binomial(num_copies, p))
    return count / num_copies


def simulate_drift(pop_sizes: list[int], p0: float) -> list[float]:
    """
    Simulates genetic drift in a population whose size can change every generation.
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation;
      pop_sizes[t] is the size of generation t + 1, so the list length is the
      number of generations simulated.
    - p0 (float): The starting allele frequency.
    Returns:
    - list[float]: The allele frequency in each generation, starting with p0,
      so its length is len(pop_sizes) + 1.
    """
    freqs = [p0]
    p = p0
    for pop_size in pop_sizes:
        p = wright_fisher_generation(pop_size, p)
        freqs.append(p)
    return freqs


def make_crash_history(normal_size: int, crash_size: int, gens_before: int,
                       gens_crash: int, gens_after: int) -> list[int]:
    """
    Builds a population size history with a sudden crash and a sudden recovery.
    Parameters:
    - normal_size (int): The population size before and after the crash.
    - crash_size (int): The population size during the crash.
    - gens_before (int): The number of generations before the crash.
    - gens_crash (int): The number of generations the crash lasts.
    - gens_after (int): The number of generations after the recovery.
    Returns:
    - list[int]: One population size per generation.
    """
    if normal_size <= 0 or crash_size <= 0:
        raise ValueError("Population sizes must be positive integers.")

    return [normal_size] * gens_before + [crash_size] * gens_crash + [normal_size] * gens_after


def make_growth_history(start_size: int, end_size: int, gens_before: int,
                        gens_growth: int, gens_after: int) -> list[int]:
    """
    Builds a population size history with exponential growth from start_size to end_size.
    Parameters:
    - start_size (int): The population size before growth.
    - end_size (int): The population size after growth.
    - gens_before (int): The number of generations before growth starts.
    - gens_growth (int): The number of generations over which the population grows.
    - gens_after (int): The number of generations after growth ends.
    Returns:
    - list[int]: One population size per generation.
    """
    if start_size <= 0 or end_size <= 0:
        raise ValueError("Population sizes must be positive integers.")
    if gens_growth <= 0:
        raise ValueError("gens_growth must be a positive integer.")

    sizes = [start_size] * gens_before
    for t in range(1, gens_growth + 1):
        sizes.append(round(start_size * (end_size / start_size) ** (t / gens_growth)))
    sizes += [end_size] * gens_after
    return sizes


def average_heterozygosity(pop_sizes: list[int], p0: float, num_trials: int) -> list[float]:
    """
    Averages the heterozygosity 2p(1 - p) over many simulations, generation by generation.
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation.
    - p0 (float): The starting allele frequency.
    - num_trials (int): The number of independent simulations to average over.
    Returns:
    - list[float]: The average heterozygosity in each generation, starting with generation 0.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")

    totals = [0.0] * (len(pop_sizes) + 1)
    for _ in range(num_trials):
        freqs = simulate_drift(pop_sizes, p0)
        for t, p in enumerate(freqs):
            totals[t] += 2 * p * (1 - p)
    return [total / num_trials for total in totals]


def run_until_fixed_or_lost(pop_size: int, p0: float) -> bool:
    """
    Simulates drift until the allele is either fixed or lost.
    Parameters:
    - pop_size (int): The number of diploid individuals N.
    - p0 (float): The starting allele frequency.
    Returns:
    - bool: True if the allele fixed (p = 1), False if it was lost (p = 0).
    """
    p = p0
    while 0 < p < 1:
        p = wright_fisher_generation(pop_size, p)
    return p == 1


def estimate_fixation_probability(pop_size: int, p0: float, num_trials: int) -> float:
    """
    Estimates the probability that a neutral allele fixes, by repeated simulation.
    Parameters:
    - pop_size (int): The number of diploid individuals N.
    - p0 (float): The starting allele frequency.
    - num_trials (int): The number of independent simulations to run.
    Returns:
    - float: The fraction of trials in which the allele fixed.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")

    count = 0
    for _ in range(num_trials):
        if run_until_fixed_or_lost(pop_size, p0):
            count += 1
    return count / num_trials


def main():
    set_seed(120)

    # One trajectory: watch the frequency wander.
    freqs = simulate_drift(pop_sizes=[50] * 20, p0=0.5)
    print("Frequency over 20 generations (N = 50):")
    print(" ".join(f"{p:.2f}" for p in freqs))

    # Steady population vs. one that crashes from 500 to 10 for 10 generations.
    steady = [500] * 60
    crash = make_crash_history(normal_size=500, crash_size=10,
                               gens_before=20, gens_crash=10, gens_after=30)
    steady_h = average_heterozygosity(steady, p0=0.5, num_trials=200)
    crash_h = average_heterozygosity(crash, p0=0.5, num_trials=200)
    print()
    print("Average heterozygosity 2p(1-p), 200 runs each (starts at 0.500):")
    print(f"{'gen':>4} {'N (crash run)':>14} {'steady':>8} {'crash':>8}")
    for t in [0, 10, 20, 25, 30, 40, 60]:
        size = crash[t - 1] if t > 0 else crash[0]
        print(f"{t:>4} {size:>14} {steady_h[t]:>8.3f} {crash_h[t]:>8.3f}")

    # Sanity check from the proposal: P(fix) should equal p0.
    print()
    for p0 in [0.1, 0.2, 0.5]:
        estimate = estimate_fixation_probability(pop_size=50, p0=p0, num_trials=2000)
        print(f"p0 = {p0:.1f}: fixed in {estimate:.3f} of trials (expected {p0:.3f})")


if __name__ == "__main__":
    main()
