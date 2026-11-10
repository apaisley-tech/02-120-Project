"""
Plots how Tajima's D and pi change over time after population crashes of different sizes.

Run with: python3 plots.py
The figure is saved to figures/crash_recovery.png.

Pseudocode

RunCrashExperiment(normalSize, crashSizes, gensCrash, gensAfter, mu, n, numTrials, recordEvery)
    for every crashSize in crashSizes
        sizes ← MakeCrashHistory(normalSize, crashSize, 0, gensCrash, gensAfter)
        results[crashSize] ← CompareToSteady(normalSize, sizes, mu, n, numTrials, recordEvery)
    return results

PlotCrashRecovery(results)
    draw two panels sharing the time axis: Tajima's D on top, pi below
    in each panel, draw the steady control as a gray dashed line and one line
    per crash size, darker for more severe crashes
    shade the generations the crash lasted
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wright_fisher
from recovery import compare_to_steady

# Ordinal blue ramp, light (mild crash) to dark (severe crash).
CRASH_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STEADY_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def run_crash_experiment(normal_size: int, crash_sizes: list[int], gens_crash: int,
                         gens_after: int, mu: float, sample_size: int, num_trials: int,
                         record_every: int) -> dict[int, dict]:
    """
    Runs compare_to_steady for crashes of several sizes.
    Parameters:
    - normal_size (int): The population size before and after each crash.
    - crash_sizes (list[int]): The population sizes during the crash to compare.
    - gens_crash (int): The number of generations each crash lasts.
    - gens_after (int): The number of generations simulated after the crash ends.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - num_trials (int): The number of independent simulations per population.
    - record_every (int): How many generations apart to sample.
    Returns:
    - dict[int, dict]: The compare_to_steady result for each crash size.
    """
    results = {}
    for crash_size in crash_sizes:
        sizes = wright_fisher.make_crash_history(normal_size, crash_size, gens_before=0,
                                                 gens_crash=gens_crash, gens_after=gens_after)
        results[crash_size] = compare_to_steady(normal_size, sizes, mu, sample_size,
                                                num_trials, record_every)
    return results


def style_axes(ax: plt.Axes) -> None:
    """
    Gives a panel a light grid, recessive axes and no top or right border.
    Parameters:
    - ax (plt.Axes): The panel to style.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def plot_crash_recovery(results: dict[int, dict], normal_size: int, gens_crash: int,
                        theta: float, path: str) -> None:
    """
    Draws Tajima's D and pi over time for each crash size and saves the figure.
    Parameters:
    - results (dict[int, dict]): The output of run_crash_experiment.
    - normal_size (int): The population size before and after each crash.
    - gens_crash (int): The number of generations each crash lasted.
    - theta (float): The expected pi in a steady population, 4N * mu.
    - path (str): Where to save the figure.
    """
    fig, (ax_d, ax_pi) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                      facecolor=SURFACE)
    panels = [(ax_d, "D", "Tajima's D"),
              (ax_pi, "pi", f"π (steady value θ = 4Nμ = {theta:g})")]
    steady = next(iter(results.values()))

    for ax, stat, label in panels:
        style_axes(ax)
        ax.axvspan(0, gens_crash, color=GRID_COLOR, linewidth=0)
        ax.plot(steady["gens"], steady["steady"][stat], color=STEADY_COLOR,
                linewidth=1.5, linestyle="--", label=f"steady (N = {normal_size})")
        for color, (crash_size, result) in zip(CRASH_COLORS, results.items()):
            ax.plot(result["gens"], result["event"][stat], color=color, linewidth=2,
                    label=f"crash to N = {crash_size}")
        ax.set_ylabel(label, color=TEXT_PRIMARY, fontsize=10)

    ax_d.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax_pi.set_ylim(bottom=0)
    ax_d.text(gens_crash + 8, ax_d.get_ylim()[1], "crash", va="top", fontsize=8,
              color=TEXT_SECONDARY)

    ax_pi.set_xlabel("generations since the crash began", color=TEXT_PRIMARY, fontsize=10)
    ax_pi.set_xlim(0, steady["gens"][-1])
    ax_d.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="upper right")

    fig.suptitle(f"A {gens_crash}-generation crash from N = {normal_size}: "
                 "Tajima's D and π over time", x=0.07, ha="left", fontsize=12,
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    wright_fisher.set_seed(120)

    normal_size, mu, sample_size = 200, 0.0125, 20
    gens_crash = 20
    results = run_crash_experiment(normal_size, crash_sizes=[100, 50, 20, 10],
                                   gens_crash=gens_crash, gens_after=1000, mu=mu,
                                   sample_size=sample_size, num_trials=200, record_every=10)

    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", "crash_recovery.png")
    plot_crash_recovery(results, normal_size, gens_crash, 4 * normal_size * mu, path)
    print(f"Saved {path}")
    for crash_size, result in results.items():
        print(f"crash to N = {crash_size:>3}: D recovers at {result['recovery']['D']}, "
              f"pi recovers at {result['recovery']['pi']}")


if __name__ == "__main__":
    main()
