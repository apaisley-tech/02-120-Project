"""
Simulates samples of completely linked sites (no recombination), to compare with
the rest of the project, where every site drifts independently (unlinked sites).

With no recombination, all sites in a sequence share one family tree: the sampled
sequences' ancestry, traced back to their common ancestor. A mutation on a branch
of the tree is carried by every sampled sequence below that branch. So instead of
following every site forwards, we trace the n sampled sequences backwards
(Wright-Fisher genealogy): each generation, every ancestral lineage picks a parent
at random among the 2N gene copies of the generation before, and lineages that pick
the same parent merge. Each lineage spends one generation in each generation it
passes through; mutations hit it at rate mu per generation, and each one is carried
by all the sampled sequences descended from that lineage.

This is the same population model as the forward simulation (same size history, the
same mutation rate per gene copy, and no variation at generation 0), so averages
such as E[pi] and E[S] are the same. What changes is how much one sample varies:
linked sites share one tree, so a sample with a deep tree has many sites at once.
Tajima's test assumes exactly this case (Tajima 1989).

Pseudocode

LinkedSampleStats(popSizes, mu, n, numTrials, recordGens)
    (Every trial and every recorded generation gets its own, independent tree.
    All of them are traced at once, one generation at a time.)
    lineages ← none; branchLength[replicate][i] ← 0 for i = 1 to n - 1
    for g from the last recorded generation down to 1
        for every replicate sampled at generation g
            add n lineages, each with 1 sampled descendant
        for every lineage
            branchLength[its replicate][its descendants] += 1
        if g > 1
            every lineage picks a parent among 2 * popSizes[g - 1] gene copies
            lineages of the same replicate with the same parent merge, adding
            their descendants
            drop lineages that are the ancestor of all n sequences
    for every replicate and every i from 1 to n - 1
        sfs[i] ← Poisson(mu * branchLength[replicate][i])
    compute S, pi, theta_W, D and singletons from sfs

(plots.py linkage repeats the crash experiments and the inference test with
LinkedSampleStats in place of SampleStatsPerTrial.)
"""

import sys

import numpy as np

import wright_fisher
from diversity import harmonic_sums, sample_stats_per_trial, tajimas_d_from_summary


def linked_sample_stats(pop_sizes: list[int], mu: float, sample_size: int, num_trials: int,
                        record_gens: list[int]) -> dict[str, np.ndarray]:
    """
    Simulates samples of completely linked sites: each sample of n sequences gets one
    Wright-Fisher family tree, traced back through the population size history, and
    mutations fall on its branches. Same inputs and outputs as
    diversity.sample_stats_per_trial, so the two can be swapped.
    Unlike the forward simulation, samples taken at different generations of the same
    trial are independent of each other (each has its own tree).
    Parameters:
    - pop_sizes (list[int]): The number of diploid individuals in each generation,
      including any burn-in generations at the start (generation 0 has no variation).
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - num_trials (int): The number of independent samples per recorded generation.
    - record_gens (list[int]): The generations (1 to len(pop_sizes)) to sample at.
    Returns:
    - dict[str, np.ndarray]: Keys "S", "pi", "theta_w", "D" and "singletons", each with
      one row per trial and one column per entry of record_gens. D is nan where the
      sample had no segregating sites.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")
    if any(t < 1 or t > len(pop_sizes) for t in record_gens):
        raise ValueError("record_gens must be between 1 and len(pop_sizes).")
    if any(2 * pop_sizes[t - 1] < sample_size for t in record_gens):
        raise ValueError("Every sampled generation needs at least n gene copies.")

    rng = wright_fisher.rng
    n = sample_size
    num_reps = num_trials * len(record_gens)
    # Replicate i * num_trials + j is trial j sampled at record_gens[i].
    start_gen = np.repeat(np.asarray(record_gens), num_trials)
    order = np.argsort(-start_gen, kind="stable")
    next_new = 0
    # branch[r * n + i - 1]: generations spent by lineages of replicate r with i descendants.
    branch = np.zeros(num_reps * n)
    rep = np.empty(0, dtype=np.int64)
    count = np.empty(0, dtype=np.int64)

    for g in range(max(record_gens), 0, -1):
        # Start the replicates sampled at generation g: n lineages, one descendant each.
        first = next_new
        while next_new < num_reps and start_gen[order[next_new]] == g:
            next_new += 1
        if next_new > first:
            new = np.repeat(order[first:next_new], n)
            rep = np.concatenate([rep, new])
            count = np.concatenate([count, np.ones(len(new), dtype=np.int64)])
        if len(rep) == 0:
            continue

        # Every lineage alive in generation g can gain mutations there.
        branch += np.bincount(rep * n + count - 1, minlength=num_reps * n)
        if g == 1:
            break

        # Each lineage picks a parent in generation g - 1; same parent means they merge.
        copies = 2 * pop_sizes[g - 2]
        key = rep * copies + rng.integers(0, copies, size=len(rep))
        parents, which = np.unique(key, return_inverse=True)
        count = np.bincount(which, weights=count).astype(np.int64)
        rep = parents // copies
        # A lineage carrying all n sequences is their common ancestor: its mutations
        # are in every sampled sequence, so they do not vary within the sample.
        alive = count < n
        rep, count = rep[alive], count[alive]

    sfs = rng.poisson(mu * branch.reshape(num_reps, n)[:, :n - 1])
    i = np.arange(1, n)
    num_segregating = sfs.sum(axis=1)
    pi = (sfs * i * (n - i)).sum(axis=1) / (n * (n - 1) / 2)
    a1, _ = harmonic_sums(n)
    flat = {"S": num_segregating.astype(float), "pi": pi, "theta_w": num_segregating / a1,
            "D": tajimas_d_from_summary(pi, num_segregating, n),
            "singletons": sfs[:, 0].astype(float)}
    return {name: values.reshape(len(record_gens), num_trials).T for name, values in flat.items()}


def main():
    normal_size, mu, sample_size = 200, 0.0125, 20
    burn_in = [normal_size] * (10 * normal_size)
    sizes = burn_in + [normal_size] * 500
    record_gens = [len(burn_in) + t for t in range(0, 501, 10)]
    print(f"Steady population, N = {normal_size}, theta = 4N * mu = {4 * normal_size * mu:g}, "
          f"n = {sample_size} (200 populations x 51 times)")
    for model, simulate in [("unlinked", sample_stats_per_trial), ("linked", linked_sample_stats)]:
        wright_fisher.set_seed(1)
        stats = simulate(sizes, mu, sample_size, 200, record_gens)
        print(f"  {model:>8}: average pi {np.mean(stats['pi']):.2f}, theta_W "
              f"{np.mean(stats['theta_w']):.2f}, D {np.nanmean(stats['D']):+.2f}; "
              f"spread (SD) of D {np.nanstd(stats['D']):.2f}, of pi {np.std(stats['pi']):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("usage: python3 linkage.py")
    main()
