"""
Basic Wright-Fisher simulation of genetic drift at one locus with two alleles.

Pseudocode

WrightFisherGeneration(popSize, p)
    numCopies ← 2 * popSize
    count ← 0
    for numCopies total draws
        if RandFloat() < p
            count ← count + 1
    return count / numCopies

SimulateDrift(popSize, p0, numGens)
    freqs ← [p0]
    p ← p0
    for numGens generations
        p ← WrightFisherGeneration(popSize, p)
        append p to freqs
    return freqs

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

import random


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

    num_copies = 2 * pop_size
    count = 0
    for _ in range(num_copies):
        if random.random() < p:
            count += 1
    return count / num_copies


def simulate_drift(pop_size: int, p0: float, num_gens: int) -> list[float]:
    """
    Simulates genetic drift for a fixed number of generations.
    Parameters:
    - pop_size (int): The number of diploid individuals N.
    - p0 (float): The starting allele frequency.
    - num_gens (int): The number of generations to simulate.
    Returns:
    - list[float]: The allele frequency in each generation, starting with p0.
    """
    freqs = [p0]
    p = p0
    for _ in range(num_gens):
        p = wright_fisher_generation(pop_size, p)
        freqs.append(p)
    return freqs


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
    random.seed(120)

    # One trajectory: watch the frequency wander.
    freqs = simulate_drift(pop_size=50, p0=0.5, num_gens=20)
    print("Frequency over 20 generations (N = 50):")
    print(" ".join(f"{p:.2f}" for p in freqs))

    # Sanity check from the proposal: P(fix) should equal p0.
    print()
    for p0 in [0.1, 0.2, 0.5]:
        estimate = estimate_fixation_probability(pop_size=50, p0=p0, num_trials=2000)
        print(f"p0 = {p0:.1f}: fixed in {estimate:.3f} of trials (expected {p0:.3f})")


if __name__ == "__main__":
    main()
