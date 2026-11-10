"""
Samples DNA sequences from an infinite-sites population and measures genetic variation.

A sample of n sequences is summarized by its mutant counts: for each site that
varies within the sample, how many of the n sequences carry the mutant. From
these counts we compute
- S, the number of segregating (variable) sites,
- pi, the average number of differences between two sampled sequences,
- Watterson's theta, S / a1, where a1 = 1 + 1/2 + ... + 1/(n - 1),
- Tajima's D, the scaled difference pi - theta_W (Tajima 1989).
Under a steady population size, pi and theta_W both estimate 4N * mu, so D is
about 0. Too few rare variants (after a crash) makes D positive; too many
(after growth) makes it negative.

Pseudocode

SampleCounts(freqs, popSize, n)
    numCopies ← 2 * popSize
    for every site i
        copies ← freqs[i] * numCopies
        counts[i] ← number of mutants among n copies drawn without replacement
                    from copies mutants and numCopies - copies originals
    return the counts that are neither 0 nor n

NucleotideDiversity(counts, n)
    return the sum of counts[i] * (n - counts[i]) over all sites, divided by n(n - 1)/2

WattersonTheta(counts, n)
    a1 ← 1 + 1/2 + ... + 1/(n - 1)
    return len(counts) / a1

TajimasD(counts, n)
    S ← len(counts)
    if S = 0
        return undefined
    compute the constants a1, a2, b1, b2, c1, c2, e1, e2 from n (Tajima 1989, eq. 38)
    return (NucleotideDiversity(counts, n) - S / a1) / sqrt(e1 * S + e2 * S * (S - 1))

AverageSampleStats(popSizes, mu, n, numTrials, recordGens)
    for numTrials total trials
        freqs ← empty list
        for every generation t
            freqs ← SitesGeneration(popSizes[t], freqs, mu)
            if t is in recordGens
                counts ← SampleCounts(freqs, popSizes[t], n)
                add S, pi, theta_W and D of counts to the running totals for t
    return the averages for every t in recordGens, skipping undefined D values
"""

import math

import numpy as np

import wright_fisher
from infinite_sites import sites_generation


def sample_counts(freqs: np.ndarray, pop_size: int, sample_size: int) -> np.ndarray:
    """
    Draws a sample of gene copies from the population without replacement and
    counts the mutants at every site.
    Parameters:
    - freqs (np.ndarray): The mutant frequency at each variable site in the population.
    - pop_size (int): The number of diploid individuals N, so there are 2N gene copies.
    - sample_size (int): The number of sampled sequences n, at least 2 and at most 2N.
    Returns:
    - np.ndarray: The number of sampled sequences carrying the mutant, for each
      site that varies within the sample (so every count is between 1 and n - 1).
    """
    num_copies = 2 * pop_size
    if sample_size < 2 or sample_size > num_copies:
        raise ValueError("sample_size must be between 2 and 2 * pop_size.")

    mutant_copies = np.round(freqs * num_copies).astype(int)
    counts = wright_fisher.rng.hypergeometric(mutant_copies, num_copies - mutant_copies,
                                              sample_size)
    return counts[(counts > 0) & (counts < sample_size)]


def nucleotide_diversity(counts: np.ndarray, sample_size: int) -> float:
    """
    Computes pi, the average number of sites at which two sampled sequences differ.
    Parameters:
    - counts (np.ndarray): The mutant count at each segregating site in the sample.
    - sample_size (int): The number of sampled sequences n.
    Returns:
    - float: The average pairwise difference pi.
    """
    num_pairs = sample_size * (sample_size - 1) / 2
    return float(np.sum(counts * (sample_size - counts))) / num_pairs


def harmonic_sums(sample_size: int) -> tuple[float, float]:
    """
    Computes the sums a1 = 1 + 1/2 + ... + 1/(n - 1) and a2 = 1 + 1/4 + ... + 1/(n - 1)^2.
    Parameters:
    - sample_size (int): The number of sampled sequences n.
    Returns:
    - tuple[float, float]: The pair (a1, a2).
    """
    i = np.arange(1, sample_size)
    return float(np.sum(1 / i)), float(np.sum(1 / i ** 2))


def watterson_theta(counts: np.ndarray, sample_size: int) -> float:
    """
    Computes Watterson's estimate of theta = 4N * mu from the number of segregating sites.
    Parameters:
    - counts (np.ndarray): The mutant count at each segregating site in the sample.
    - sample_size (int): The number of sampled sequences n.
    Returns:
    - float: S / a1.
    """
    a1, _ = harmonic_sums(sample_size)
    return len(counts) / a1


def tajimas_d(counts: np.ndarray, sample_size: int) -> float:
    """
    Computes Tajima's D, following equation 38 of Tajima (1989).
    Parameters:
    - counts (np.ndarray): The mutant count at each segregating site in the sample.
    - sample_size (int): The number of sampled sequences n.
    Returns:
    - float: Tajima's D, or nan when the sample has no segregating sites
      (D is undefined then).
    """
    num_segregating = len(counts)
    if num_segregating == 0:
        return math.nan

    n = sample_size
    a1, a2 = harmonic_sums(n)
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)

    difference = nucleotide_diversity(counts, n) - num_segregating / a1
    variance = e1 * num_segregating + e2 * num_segregating * (num_segregating - 1)
    return difference / math.sqrt(variance)


def average_sample_stats(pop_sizes: list[int], mu: float, sample_size: int,
                         num_trials: int, record_gens: list[int]) -> dict[str, list[float]]:
    """
    Runs many simulations that start with no variation and, at chosen generations,
    samples sequences and averages S, pi, theta_W and Tajima's D across the runs.
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation,
      including any burn-in generations at the start.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - num_trials (int): The number of independent simulations to average over.
    - record_gens (list[int]): The generations (1 to len(pop_sizes)) to sample at.
    Returns:
    - dict[str, list[float]]: Keys "S", "pi", "theta_w" and "D", each with one
      average per entry of record_gens. Runs where D is undefined are left out
      of the D average.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")
    if any(t < 1 or t > len(pop_sizes) for t in record_gens):
        raise ValueError("record_gens must be between 1 and len(pop_sizes).")

    index = {t: i for i, t in enumerate(record_gens)}
    stats = {name: np.zeros((num_trials, len(record_gens))) for name in ["S", "pi", "theta_w", "D"]}
    for trial in range(num_trials):
        freqs = np.empty(0)
        for t, pop_size in enumerate(pop_sizes, start=1):
            freqs = sites_generation(pop_size, freqs, mu)
            if t in index:
                counts = sample_counts(freqs, pop_size, sample_size)
                i = index[t]
                stats["S"][trial, i] = len(counts)
                stats["pi"][trial, i] = nucleotide_diversity(counts, sample_size)
                stats["theta_w"][trial, i] = watterson_theta(counts, sample_size)
                stats["D"][trial, i] = tajimas_d(counts, sample_size)

    averages = {name: list(np.mean(values, axis=0)) for name, values in stats.items()}
    averages["D"] = list(np.nanmean(stats["D"], axis=0))
    return averages


def main():
    wright_fisher.set_seed(120)

    pop_size, mu, sample_size = 200, 0.0125, 20
    burn_in = [pop_size] * (10 * pop_size)
    histories = {
        "steady": burn_in + [pop_size] * 200,
        "crash": burn_in + wright_fisher.make_crash_history(
            normal_size=pop_size, crash_size=10, gens_before=0, gens_crash=20, gens_after=180),
        "growth": burn_in + wright_fisher.make_growth_history(
            start_size=pop_size, end_size=20 * pop_size, gens_before=0, gens_growth=20,
            gens_after=180),
    }
    after = [0, 10, 20, 30, 50, 100, 200]

    print(f"Average Tajima's D, 100 runs each (N = {pop_size}, theta = 4N * mu = "
          f"{4 * pop_size * mu:.0f}, n = {sample_size}):")
    print(f"{'gens after burn-in':>18} " + " ".join(f"{t:>6}" for t in after))
    for name, sizes in histories.items():
        stats = average_sample_stats(sizes, mu, sample_size, num_trials=100,
                                     record_gens=[len(burn_in) + t for t in after])
        print(f"{name:>18} " + " ".join(f"{d:>6.2f}" for d in stats["D"]))


if __name__ == "__main__":
    main()
