"""
Plots how Tajima's D and pi change over time after population crashes or growth
of different sizes.

Run with: python3 plots.py [crash | growth]
With no argument, both figures are made. They are saved to figures/crash_recovery.png
and figures/growth_recovery.png.

Pseudocode

RunExperiment(normalSize, histories, mu, n, numTrials, recordEvery)
    for every label, sizes in histories
        results[label] ← CompareToSteady(normalSize, sizes, mu, n, numTrials, recordEvery)
    return results

PlotRecovery(results, eventGens)
    draw two panels sharing the time axis: Tajima's D on top, pi below
    in each panel, draw the steady control as a gray dashed line and one line
    per event, darker for more extreme events
    shade the generations the event lasted
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wright_fisher
from recovery import compare_to_steady

# Ordinal blue ramp, light (mild event) to dark (extreme event).
EVENT_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STEADY_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def run_experiment(normal_size: int, histories: dict[str, list[int]], mu: float,
                   sample_size: int, num_trials: int, record_every: int) -> dict[str, dict]:
    """
    Runs compare_to_steady for several population size histories.
    Parameters:
    - normal_size (int): The population size during the burn-in and in the steady control.
    - histories (dict[str, list[int]]): A label and the population size in each
      generation after the burn-in, for every event to compare, from mildest to most extreme.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - num_trials (int): The number of independent simulations per population.
    - record_every (int): How many generations apart to sample.
    Returns:
    - dict[str, dict]: The compare_to_steady result for each label.
    """
    return {label: compare_to_steady(normal_size, sizes, mu, sample_size, num_trials,
                                     record_every)
            for label, sizes in histories.items()}


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


def plot_recovery(results: dict[str, dict], normal_size: int, event_name: str,
                  event_gens: int, theta: float, title: str, path: str) -> None:
    """
    Draws Tajima's D and pi over time for each event and saves the figure.
    Parameters:
    - results (dict[str, dict]): The output of run_experiment, mildest event first.
    - normal_size (int): The population size of the steady control.
    - event_name (str): A word for the event, such as "crash" or "growth".
    - event_gens (int): The number of generations the event lasted.
    - theta (float): The expected pi in the steady control, 4N * mu.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, (ax_d, ax_pi) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                      facecolor=SURFACE)
    panels = [(ax_d, "D", "Tajima's D"),
              (ax_pi, "pi", f"π (steady value θ = 4Nμ = {theta:g})")]
    steady = next(iter(results.values()))

    for ax, stat, label in panels:
        style_axes(ax)
        ax.axvspan(0, event_gens, color=GRID_COLOR, linewidth=0)
        ax.plot(steady["gens"], steady["steady"][stat], color=STEADY_COLOR,
                linewidth=1.5, linestyle="--", label=f"steady (N = {normal_size})")
        for color, (event_label, result) in zip(EVENT_COLORS, results.items()):
            ax.plot(result["gens"], result["event"][stat], color=color, linewidth=2,
                    label=event_label)
        ax.set_ylabel(label, color=TEXT_PRIMARY, fontsize=10)

    ax_d.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax_pi.set_ylim(bottom=0)
    ax_d.text(event_gens + 8, ax_d.get_ylim()[1], event_name, va="top", fontsize=8,
              color=TEXT_SECONDARY)

    ax_pi.set_xlabel(f"generations since the {event_name} began", color=TEXT_PRIMARY,
                     fontsize=10)
    ax_pi.set_xlim(0, steady["gens"][-1])
    ax_d.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="best")

    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    normal_size, mu, sample_size = 200, 0.0125, 20
    event_gens, gens_after = 20, 1000
    theta = 4 * normal_size * mu

    crash_histories = {
        f"crash to N = {crash_size}": wright_fisher.make_crash_history(
            normal_size, crash_size, gens_before=0, gens_crash=event_gens, gens_after=gens_after)
        for crash_size in [100, 50, 20, 10]
    }
    growth_histories = {
        f"growth to N = {end_size}": wright_fisher.make_growth_history(
            normal_size, end_size, gens_before=0, gens_growth=event_gens, gens_after=gens_after)
        for end_size in [400, 1000, 2000, 4000]
    }
    figures = {
        "crash": (crash_histories,
                  f"A {event_gens}-generation crash from N = {normal_size}: "
                  "Tajima's D and π over time"),
        "growth": (growth_histories,
                   f"{event_gens} generations of rapid growth from N = {normal_size}: "
                   "Tajima's D and π over time"),
    }

    chosen = sys.argv[1:] or list(figures)
    if any(name not in figures for name in chosen):
        sys.exit("usage: python3 plots.py [crash | growth]")

    os.makedirs("figures", exist_ok=True)
    for name in chosen:
        wright_fisher.set_seed(120)
        histories, title = figures[name]
        results = run_experiment(normal_size, histories, mu, sample_size, num_trials=200,
                                 record_every=10)
        path = os.path.join("figures", f"{name}_recovery.png")
        plot_recovery(results, normal_size, name, event_gens, theta, title, path)
        print(f"Saved {path}")
        for label, result in results.items():
            recovery = {stat: result["recovery"][stat] for stat in ["D", "pi"]}
            print("  " + label + ": " + ", ".join(
                f"{stat} recovers at {gen}" if gen is not None
                else f"{stat} not recovered within {gens_after + event_gens} generations"
                for stat, gen in recovery.items()))


if __name__ == "__main__":
    main()
