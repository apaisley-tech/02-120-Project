"""
Plots how Tajima's D and pi change over time after population crashes or growth
of different sizes, and how long crashes and growth of different sizes take to
recover from.

Run with: python3 plots.py [crash | growth | severity | length | growthsize]
With no argument, all five figures are made. They are saved to
figures/crash_recovery.png, figures/growth_recovery.png,
figures/recovery_vs_crash_size.png, figures/recovery_vs_crash_strength.png and
figures/recovery_vs_growth_size.png; the severity, length and growthsize options
also save their recovery times to a CSV file with the same name.

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

PlotRecoveryVsStrength(times, depths, lengths)
    in one panel for D and one for pi
        for every crash depth, plot recovery time against the crash strength
        length / (2 * depth), as a line through the averages over seeds
    if strength alone sets recovery time, the lines for different depths overlap

PlotRecoveryVsGrowthSize(times, endSizes)
    in the left panel, plot recovery time in generations against the size after growth
    in the right panel, plot recovery time divided by the size after growth
    in both, draw each seed as a small dot and the average over seeds as a line,
    for D and for pi
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wright_fisher
from recovery import compare_histories, growth_recovery_times, recovery_times_over_seeds

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


def save_recovery_csv(times: dict[str, np.ndarray], scenarios: list[tuple],
                      scenario_headers: list[str], seeds: list[int], path: str) -> None:
    """
    Saves recovery times to a CSV file with one row per seed and scenario.
    Parameters:
    - times (dict[str, np.ndarray]): The output of recovery_times_over_seeds.
    - scenarios (list[tuple]): The settings of each scenario, one per column of times,
      such as (crash size,) or (crash size, crash length).
    - scenario_headers (list[str]): The column names for the settings in scenarios.
    - seeds (list[int]): The random seeds used, one per row of times.
    - path (str): Where to save the file. Times that did not recover are left blank.
    """
    stats = list(times)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed"] + scenario_headers + [f"{stat}_recovery" for stat in stats])
        for i, seed in enumerate(seeds):
            for j, scenario in enumerate(scenarios):
                row = [seed, *scenario]
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
    histories = {crash_size: wright_fisher.make_crash_history(
                     normal_size, crash_size, gens_before=0, gens_crash=gens_crash,
                     gens_after=gens_after)
                 for crash_size in crash_sizes}
    times = recovery_times_over_seeds(normal_size, histories, mu, sample_size, num_trials=200,
                                      record_every=10, seeds=seeds)

    title = (f"How long a {gens_crash}-generation crash affects Tajima's D and π "
             "(200 populations per seed)")
    path = os.path.join("figures", "recovery_vs_crash_size.png")
    plot_recovery_vs_crash_size(times, crash_sizes, normal_size, gens_crash, title, path)
    csv_path = os.path.join("figures", "recovery_vs_crash_size.csv")
    save_recovery_csv(times, [(size,) for size in crash_sizes], ["crash_size"], seeds,
                      csv_path)
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


def plot_recovery_vs_strength(times: dict[str, np.ndarray], scenarios: list[tuple[int, int]],
                              depths: list[int], normal_size: int, title: str,
                              path: str) -> None:
    """
    Plots recovery time against crash strength, length / (2 * depth), with one line per
    crash depth, in one panel for Tajima's D and one for pi, and saves the figure.
    Parameters:
    - times (dict[str, np.ndarray]): The output of recovery_times_over_seeds.
    - scenarios (list[tuple[int, int]]): The (depth, length) of each crash, one per
      column of times.
    - depths (list[int]): The crash depths, mildest (largest) first.
    - normal_size (int): The population size before and after each crash.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, facecolor=SURFACE)
    strengths = sorted({length / (2 * depth) for depth, length in scenarios})
    # Skip the lightest ramp step so three lines stay distinct.
    colors = dict(zip(depths, EVENT_COLORS[-len(depths):]))

    for ax, stat in zip(axes, ["D", "pi"]):
        style_axes(ax)
        for depth in depths:
            columns = [j for j, (d, _) in enumerate(scenarios) if d == depth]
            x = [scenarios[j][1] / (2 * depth) for j in columns]
            for j, strength in zip(columns, x):
                ax.scatter([strength] * len(times[stat]), times[stat][:, j], s=14,
                           color=colors[depth], alpha=0.35, linewidths=0)
            means = np.nanmean(times[stat][:, columns], axis=0)
            ax.plot(x, means, color=colors[depth], linewidth=2, marker="o", markersize=6,
                    label=f"crash to N = {depth}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(strengths)
        ax.set_xticklabels([f"{strength:g}" for strength in strengths])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(STAT_LABELS[stat], loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.set_xlabel("crash strength: length / (2 × crash size)", color=TEXT_PRIMARY,
                      fontsize=10)

    axes[0].set_ylabel("recovery time (generations since the crash began)",
                       color=TEXT_PRIMARY, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="upper left")

    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_length_figure(normal_size: int, mu: float, sample_size: int) -> None:
    """
    Runs crashes of every combination of depth and length over several seeds,
    then saves the figure and CSV.
    Parameters:
    - normal_size (int): The population size before and after each crash.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    """
    depths = [40, 20, 10]
    lengths = [5, 10, 20, 40]
    seeds = [120, 121, 122, 123, 124]
    total_gens = 1600
    scenarios = [(depth, length) for depth in depths for length in lengths]
    histories = {(depth, length): wright_fisher.make_crash_history(
                     normal_size, depth, gens_before=0, gens_crash=length,
                     gens_after=total_gens - length)
                 for depth, length in scenarios}
    times = recovery_times_over_seeds(normal_size, histories, mu, sample_size, num_trials=200,
                                      record_every=10, seeds=seeds)

    title = (f"Crashes from N = {normal_size} of different depths and lengths: "
             "recovery depends mainly on strength")
    path = os.path.join("figures", "recovery_vs_crash_strength.png")
    plot_recovery_vs_strength(times, scenarios, depths, normal_size, title, path)
    csv_path = os.path.join("figures", "recovery_vs_crash_strength.csv")
    save_recovery_csv(times, scenarios, ["crash_size", "crash_length"], seeds, csv_path)
    print(f"Saved {path} and {csv_path}")

    print(f"{'crash size':>10} {'length':>6} {'strength':>8} {'D recovers':>12} "
          f"{'pi recovers':>12}   (mean ± SD over seeds)")
    for j, (depth, length) in enumerate(scenarios):
        cells = [f"{np.nanmean(times[stat][:, j]):.0f} ± {np.nanstd(times[stat][:, j]):.0f}"
                 for stat in ["D", "pi"]]
        print(f"{depth:>10} {length:>6} {length / (2 * depth):>8g} {cells[0]:>12} {cells[1]:>12}")


def plot_recovery_vs_growth_size(times: dict[str, np.ndarray], end_sizes: list[int],
                                 start_size: int, gens_growth: int, title: str,
                                 path: str) -> None:
    """
    Plots recovery time against the population size after growth, in generations
    (left) and in units of the new size (right), for Tajima's D and pi, and saves
    the figure.
    Parameters:
    - times (dict[str, np.ndarray]): The output of growth_recovery_times.
    - end_sizes (list[int]): The population sizes after growth, one per column of times.
    - start_size (int): The population size before growth.
    - gens_growth (int): The number of generations the growth lasted.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, (ax_gens, ax_scaled) = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)
    sizes = np.array(end_sizes)

    for ax, scale, ylabel in [
            (ax_gens, np.ones(len(sizes)), "recovery time (generations since growth began)"),
            (ax_scaled, sizes, "recovery time ÷ population size after growth")]:
        style_axes(ax)
        for stat in ["D", "pi"]:
            color = STAT_COLORS[stat]
            scaled = times[stat] / scale
            for row in scaled:
                ax.scatter(sizes, row, s=14, color=color, alpha=0.35, linewidths=0)
            means = np.nanmean(scaled, axis=0)
            ax.plot(sizes, means, color=color, linewidth=2, marker="o", markersize=6,
                    label=f"{STAT_LABELS[stat]} (average of {len(scaled)} seeds)")
        ax.set_xscale("log")
        ax.set_xticks(end_sizes)
        ax.set_xticklabels([f"{size}\n({size / start_size:g}×)" for size in end_sizes])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"population size after growth (from N = {start_size}, log scale)",
                      color=TEXT_PRIMARY, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=10)

    ax_gens.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="upper left")
    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_growth_size_figure(start_size: int, mu: float, sample_size: int,
                            gens_growth: int) -> None:
    """
    Runs the growth-size experiment over several seeds, then saves the figure and CSV.
    Each growth size is followed for 10 times its new size in generations, and
    compared with a steady control at the new size.
    Parameters:
    - start_size (int): The population size before growth.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - gens_growth (int): The number of generations the growth lasts.
    """
    end_sizes = [300, 400, 600, 1000]
    seeds = [120, 121, 122, 123, 124]
    total_gens = {end_size: 10 * end_size for end_size in end_sizes}
    record_every = {end_size: end_size // 20 for end_size in end_sizes}
    times = growth_recovery_times(start_size, end_sizes, gens_growth, total_gens, mu,
                                  sample_size, num_trials=200, record_every=record_every,
                                  seeds=seeds)

    title = (f"How long {gens_growth} generations of growth affect Tajima's D and π "
             "(200 populations per seed)")
    path = os.path.join("figures", "recovery_vs_growth_size.png")
    plot_recovery_vs_growth_size(times, end_sizes, start_size, gens_growth, title, path)
    csv_path = os.path.join("figures", "recovery_vs_growth_size.csv")
    save_recovery_csv(times, [(size,) for size in end_sizes], ["end_size"], seeds, csv_path)
    print(f"Saved {path} and {csv_path}")

    print(f"{'end size':>8} {'D recovers':>16} {'pi recovers':>16}   (mean ± SD over seeds)")
    for j, end_size in enumerate(end_sizes):
        cells = []
        for stat in ["D", "pi"]:
            column = times[stat][:, j]
            missing = int(np.sum(np.isnan(column)))
            cell = f"{np.nanmean(column):.0f} ± {np.nanstd(column):.0f}"
            if missing:
                cell += f" ({missing} n.r.)"
            cells.append(cell)
        print(f"{end_size:>8} {cells[0]:>16} {cells[1]:>16}")
    print("n.r. = not recovered within 10 × the new size generations")


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

    options = list(figures) + ["severity", "length", "growthsize"]
    chosen = sys.argv[1:] or options
    if any(name not in options for name in chosen):
        sys.exit("usage: python3 plots.py [crash | growth | severity | length | growthsize]")

    os.makedirs("figures", exist_ok=True)
    if "severity" in chosen:
        make_severity_figure(normal_size, mu, sample_size, event_gens)
    if "length" in chosen:
        make_length_figure(normal_size, mu, sample_size)
    if "growthsize" in chosen:
        make_growth_size_figure(normal_size, mu, sample_size, event_gens)
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
