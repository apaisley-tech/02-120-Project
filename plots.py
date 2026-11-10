"""
Plots how Tajima's D and pi change over time after population crashes or growth
of different sizes, and how long crashes of different sizes take to recover from.

Run with: python3 plots.py [crash | growth | severity]
With no argument, all three figures are made. They are saved to
figures/crash_recovery.png, figures/growth_recovery.png and
figures/recovery_vs_crash_size.png; the severity option also saves the
recovery times to figures/recovery_vs_crash_size.csv.

Pseudocode

PlotRecovery(results, eventGens)
    draw two panels sharing the time axis: Tajima's D on top, pi below
    in each panel, draw the steady control as a gray dashed line and one line
    per event, darker for more extreme events
    shade the generations the event lasted

PlotRecoveryVsCrashSize(times, crashSizes)
    for D and for pi
        plot each seed's recovery time against crash size as a small dot
        plot the average over seeds as a line
    put crash size on a log scale, reversed so more severe crashes are to the right
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wright_fisher
from recovery import compare_histories, recovery_times_over_seeds

# Ordinal blue ramp, light (mild event) to dark (extreme event).
EVENT_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Categorical slots 1 and 2, one per statistic.
STAT_COLORS = {"D": "#2a78d6", "pi": "#eb6834"}
STAT_LABELS = {"D": "Tajima's D", "pi": "π"}
STEADY_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


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
    - results (dict[str, dict]): The output of compare_histories, mildest event first.
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


def plot_recovery_vs_crash_size(times: dict[str, np.ndarray], crash_sizes: list[int],
                                normal_size: int, gens_crash: int, title: str,
                                path: str) -> None:
    """
    Plots recovery time against crash size for Tajima's D and pi and saves the figure.
    Parameters:
    - times (dict[str, np.ndarray]): The output of recovery_times_over_seeds.
    - crash_sizes (list[int]): The population sizes during the crash, one per column of times.
    - normal_size (int): The population size before and after each crash.
    - gens_crash (int): The number of generations each crash lasted.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    style_axes(ax)

    for stat in ["D", "pi"]:
        color = STAT_COLORS[stat]
        for row in times[stat]:
            ax.scatter(crash_sizes, row, s=14, color=color, alpha=0.35, linewidths=0)
        means = np.nanmean(times[stat], axis=0)
        ax.plot(crash_sizes, means, color=color, linewidth=2, marker="o", markersize=6,
                label=f"{STAT_LABELS[stat]} (average of {len(times[stat])} seeds)")
        ax.annotate(STAT_LABELS[stat], (crash_sizes[-1], means[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_xscale("log")
    ax.set_xticks(crash_sizes)
    ax.set_xticklabels([str(size) for size in crash_sizes])
    ax.minorticks_off()
    ax.set_xlim(crash_sizes[0] * 1.15, crash_sizes[-1] / 1.15)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"population size during the crash (normal size N = {normal_size}, "
                  "log scale; more severe →)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("recovery time (generations since the crash began)", color=TEXT_PRIMARY,
                  fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="upper left")

    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_recovery_csv(times: dict[str, np.ndarray], crash_sizes: list[int],
                      seeds: list[int], path: str) -> None:
    """
    Saves recovery times to a CSV file with one row per seed and crash size.
    Parameters:
    - times (dict[str, np.ndarray]): The output of recovery_times_over_seeds.
    - crash_sizes (list[int]): The population sizes during the crash.
    - seeds (list[int]): The random seeds used, one per row of times.
    - path (str): Where to save the file. Times that did not recover are left blank.
    """
    stats = list(times)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "crash_size"] + [f"{stat}_recovery" for stat in stats])
        for i, seed in enumerate(seeds):
            for j, crash_size in enumerate(crash_sizes):
                row = [seed, crash_size]
                for stat in stats:
                    value = times[stat][i, j]
                    row.append("" if np.isnan(value) else int(value))
                writer.writerow(row)


def make_severity_figure(normal_size: int, mu: float, sample_size: int,
                         gens_crash: int) -> None:
    """
    Runs the crash-size experiment over several seeds, then saves the figure and CSV.
    Parameters:
    - normal_size (int): The population size before and after each crash.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - gens_crash (int): The number of generations each crash lasts.
    """
    crash_sizes = [150, 100, 70, 50, 30, 20, 15, 10]
    seeds = [120, 121, 122, 123, 124]
    gens_after = 1500
    times = recovery_times_over_seeds(normal_size, crash_sizes, gens_crash, gens_after, mu,
                                      sample_size, num_trials=200, record_every=10,
                                      seeds=seeds)

    title = (f"How long a {gens_crash}-generation crash affects Tajima's D and π "
             "(200 populations per seed)")
    path = os.path.join("figures", "recovery_vs_crash_size.png")
    plot_recovery_vs_crash_size(times, crash_sizes, normal_size, gens_crash, title, path)
    csv_path = os.path.join("figures", "recovery_vs_crash_size.csv")
    save_recovery_csv(times, crash_sizes, seeds, csv_path)
    print(f"Saved {path} and {csv_path}")

    print(f"{'crash size':>10} {'D recovers':>16} {'pi recovers':>16}   (mean ± SD over seeds)")
    for j, crash_size in enumerate(crash_sizes):
        cells = []
        for stat in ["D", "pi"]:
            column = times[stat][:, j]
            missing = int(np.sum(np.isnan(column)))
            cell = f"{np.nanmean(column):.0f} ± {np.nanstd(column):.0f}"
            if missing:
                cell += f" ({missing} n.r.)"
            cells.append(cell)
        print(f"{crash_size:>10} {cells[0]:>16} {cells[1]:>16}")
    print(f"n.r. = not recovered within {gens_crash + gens_after} generations")


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

    chosen = sys.argv[1:] or list(figures) + ["severity"]
    if any(name not in list(figures) + ["severity"] for name in chosen):
        sys.exit("usage: python3 plots.py [crash | growth | severity]")

    os.makedirs("figures", exist_ok=True)
    if "severity" in chosen:
        make_severity_figure(normal_size, mu, sample_size, event_gens)
    for name in [name for name in chosen if name in figures]:
        wright_fisher.set_seed(120)
        histories, title = figures[name]
        results = compare_histories(normal_size, histories, mu, sample_size, num_trials=200,
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
