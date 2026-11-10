"""
Wright-Fisher simulation of many sites under the infinite-sites mutation model.

Every mutation hits a brand-new site, so each variable site holds exactly two
alleles: the original and the mutant. We store only the mutant frequency at each
site that currently varies in the population. Sites whose mutant is lost or fixed
no longer vary, so they are dropped.

Pseudocode

SitesGeneration(popSize, freqs, mu)
    numCopies ← 2 * popSize
    for every site i
        freqs[i] ← RandBinomial(numCopies, freqs[i]) / numCopies
    remove every site with freqs[i] = 0 or freqs[i] = 1
    numNew ← RandPoisson(numCopies * mu)
    append numNew sites with frequency 1 / numCopies to freqs
    return freqs

SimulateSites(popSizes, mu, freqs)
    for every popSize in popSizes
        freqs ← SitesGeneration(popSize, freqs, mu)
    return freqs

TotalHeterozygosity(freqs)
    return the sum of 2 * freqs[i] * (1 - freqs[i]) over all sites i

AverageTotalHeterozygosity(popSizes, mu, numTrials)
    totals ← list of zeros, one per generation
    for numTrials total trials
        freqs ← empty list
        for every generation t
            freqs ← SitesGeneration(popSizes[t], freqs, mu)
            totals[t] ← totals[t] + TotalHeterozygosity(freqs)
    return totals / numTrials
"""

import numpy as np

import wright_fisher


def sites_generation(pop_size: int, freqs: np.ndarray, mu: float) -> np.ndarray:
    """
    Produces the next generation's mutant frequencies at every variable site.
    Parameters:
    - pop_size (int): The number of diploid individuals N in the new generation.
    - freqs (np.ndarray): The mutant frequency at each variable site.
    - mu (float): The mutation rate per gene copy per generation, so on average
      2N * mu new sites appear each generation.
    Returns:
    - np.ndarray: The mutant frequency at each site that varies in the new generation.
    """
    if pop_size <= 0:
        raise ValueError("pop_size must be a positive integer.")
    if mu < 0:
        raise ValueError("mu must be non-negative.")

    rng = wright_fisher.rng
    num_copies = 2 * pop_size

    # Drift: one binomial draw per site, all sites at once.
    freqs = rng.binomial(num_copies, freqs) / num_copies
    freqs = freqs[(freqs > 0) & (freqs < 1)]

    # Mutation: each new mutation lands on a single gene copy at a new site.
    num_new = rng.poisson(num_copies * mu)
    new_freqs = np.full(num_new, 1 / num_copies)
    return np.concatenate([freqs, new_freqs])


def simulate_sites(pop_sizes: list[int], mu: float,
                   freqs: np.ndarray | None = None) -> np.ndarray:
    """
    Simulates drift and mutation at many sites through a population size history.
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation.
    - mu (float): The mutation rate per gene copy per generation.
    - freqs (np.ndarray | None): The starting mutant frequencies; None starts the
      population with no variation at all.
    Returns:
    - np.ndarray: The mutant frequency at each variable site after the last generation.
    """
    if freqs is None:
        freqs = np.empty(0)
    for pop_size in pop_sizes:
        freqs = sites_generation(pop_size, freqs, mu)
    return freqs


def total_heterozygosity(freqs: np.ndarray) -> float:
    """
    Sums the heterozygosity 2p(1 - p) over all sites. This is the expected number
    of sites at which two randomly chosen gene copies differ.
    Parameters:
    - freqs (np.ndarray): The mutant frequency at each variable site.
    Returns:
    - float: The total heterozygosity.
    """
    return float(np.sum(2 * freqs * (1 - freqs)))


def average_total_heterozygosity(pop_sizes: list[int], mu: float,
                                 num_trials: int) -> list[float]:
    """
    Averages the total heterozygosity over many simulations that start with no
    variation, generation by generation.
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation.
    - mu (float): The mutation rate per gene copy per generation.
    - num_trials (int): The number of independent simulations to average over.
    Returns:
    - list[float]: The average total heterozygosity in each generation, starting
      with generation 0 (which is always 0).
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")

    totals = [0.0] * (len(pop_sizes) + 1)
    for _ in range(num_trials):
        freqs = np.empty(0)
        for t, pop_size in enumerate(pop_sizes, start=1):
            freqs = sites_generation(pop_size, freqs, mu)
            totals[t] += total_heterozygosity(freqs)
    return [total / num_trials for total in totals]


def main():
    wright_fisher.set_seed(120)

    pop_size, mu = 500, 0.005
    freqs = simulate_sites([pop_size] * (10 * pop_size), mu)
    print(f"After {10 * pop_size} generations at N = {pop_size}, mu = {mu}:")
    print(f"  variable sites:       {len(freqs)}")
    print(f"  total heterozygosity: {total_heterozygosity(freqs):.2f} "
          f"(expected about 4N * mu = {4 * pop_size * mu:.2f})")


if __name__ == "__main__":
    main()
