"""
Plots how Tajima's D and pi change over time after population crashes or growth
of different sizes, and how long crashes and growth of different sizes take to
recover from.

Run with: python3 plots.py [crash | growth | severity | length | growthsize | sfs |
                           inference | linkage | wrongsize]
With no argument, all nine figures are made. They are saved to
figures/crash_recovery.png, figures/growth_recovery.png,
figures/recovery_vs_crash_size.png, figures/recovery_vs_crash_strength.png,
figures/recovery_vs_growth_size.png, figures/sfs.png, figures/inference.png,
figures/linkage.png and figures/inference_wrong_size.png; the
severity, length and growthsize options also save their recovery times to a CSV file with the same name.

Pseudocode

PlotRecovery(results, eventGens)
    draw two panels sharing the time axis: Tajima's D on top, pi below
    in each panel, draw the steady control as a gray dashed line and one line
    per event, darker for more extreme events
    shade the generations the event lasted
    on the pi panel, draw ExpectedSamplePi for every history as a dotted line

PlotSFS(ratios)
    in one panel for a crash and one for growth
        draw a dashed line at 1 (the steady shape)
        for every time, plot SFSRatio of the average spectrum against i = 1 to n - 1

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

PlotLinkage(spread, times)
    in the left panel, for unlinked and linked sites, draw the average D after one
    crash as a line and the middle 90% of single samples as a shaded band
    in the right panel, plot recovery time against crash size for D and pi,
    one color for unlinked and one for linked sites

PlotWrongSize(trueSizes, rates)
    for each real population size, test the inference made with a table at the
    assumed size, and plot how often it names the right kind of history and how
    often its 90% range for the crash time holds the truth
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wright_fisher
import inference
from diversity import sample_stats_and_sfs, sample_stats_per_trial, sfs_ratio
from infinite_sites import expected_sample_pi
from linkage import linked_sample_stats
from recovery import compare_histories, growth_recovery_times, recovery_times_over_seeds

# Ordinal blue ramp, light (mild event) to dark (extreme event).
EVENT_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Categorical slots 1 and 2, one per statistic.
STAT_COLORS = {"D": "#2a78d6", "pi": "#eb6834"}
STAT_LABELS = {"D": "Tajima's D", "pi": "π"}
# The same two slots, for unlinked against linked sites.
MODEL_COLORS = {"unlinked": "#2a78d6", "linked": "#eb6834"}
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
                  event_gens: int, theta: float, title: str, path: str,
                  theory: dict[str, np.ndarray] | None = None) -> None:
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
    - theory (dict[str, np.ndarray] | None): For each label in results, and for
      "steady", the theoretical average pi at every recorded generation (see
      expected_sample_pi). Drawn as dotted lines on the pi panel when given.
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

    if theory is not None:
        gens = steady["gens"]
        ax_pi.plot(gens, theory["steady"], color=TEXT_PRIMARY, linewidth=1, linestyle=":",
                   label="theory")
        for event_label in results:
            ax_pi.plot(gens, theory[event_label], color=TEXT_PRIMARY, linewidth=1,
                       linestyle=":")
        # One legend entry covers every dotted line.
        ax_pi.legend(handles=[ax_pi.get_lines()[-len(results) - 1]], frameon=False,
                     fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="lower right")

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


def plot_sfs(ratios: dict[str, dict[int, np.ndarray]], titles: dict[str, str],
             title: str, path: str) -> None:
    """
    Plots the shape of the site frequency spectrum at several times for each
    event, relative to a steady population, and saves the figure.
    Parameters:
    - ratios (dict): For each panel name, the sfs_ratio at each generation since
      the event began, earliest first.
    - titles (dict[str, str]): The heading of each panel.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, axes = plt.subplots(1, len(ratios), figsize=(11, 4.6), sharey=True,
                             facecolor=SURFACE)
    for ax, (name, by_gen) in zip(axes, ratios.items()):
        style_axes(ax)
        ax.axhline(1, color=STEADY_COLOR, linewidth=1.5, linestyle="--",
                   label="steady shape")
        # The first time (before the event) in gray, later times darker blue.
        colors = [STEADY_COLOR] + EVENT_COLORS
        for color, (gen, ratio) in zip(colors, by_gen.items()):
            classes = np.arange(1, len(ratio) + 1)
            ax.plot(classes, ratio, color=color, linewidth=2, marker="o", markersize=3.5,
                    label=f"generation {gen}" if gen else "before the event")
        ax.set_yscale("log")
        ax.set_xlim(0.5, len(ratio) + 0.5)
        ax.set_xticks([1, 5, 10, 15, len(ratio)])
        ax.set_title(titles[name], loc="left", fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("sequences carrying the mutant, i (of n = "
                      f"{len(ratio) + 1})", color=TEXT_PRIMARY, fontsize=10)
    axes[0].set_ylabel("share of variable sites ÷ steady share", color=TEXT_PRIMARY,
                       fontsize=10)
    axes[0].set_yticks([0.25, 0.5, 1, 2, 4])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # Both panels use the same times, so one legend under the title serves both.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.06, 0.93), ncol=5, frameon=False, fontsize=8.5,
               labelcolor=TEXT_PRIMARY)
    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_sfs_figure(normal_size: int, mu: float, sample_size: int, event_gens: int) -> None:
    """
    Simulates one crash and one growth history and plots their site frequency
    spectra at several times (figures/sfs.png).
    Parameters:
    - normal_size (int): The population size before the event.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - event_gens (int): The number of generations the crash or growth lasts.
    """
    crash_size, end_size, gens_after = 10, 10 * normal_size, 500
    histories = {
        "crash": wright_fisher.make_crash_history(normal_size, crash_size, 0, event_gens,
                                                  gens_after),
        "growth": wright_fisher.make_growth_history(normal_size, end_size, 0, event_gens,
                                                    gens_after),
    }
    titles = {"crash": f"Crash to N = {crash_size} for {event_gens} generations",
              "growth": f"Growth to N = {end_size} over {event_gens} generations"}
    gens = [0, event_gens, 100, 500]
    burn_in = [normal_size] * (10 * normal_size)

    ratios = {}
    for name, sizes in histories.items():
        wright_fisher.set_seed(120)
        _, sfs = sample_stats_and_sfs(burn_in + sizes, mu, sample_size, num_trials=500,
                                      record_gens=[len(burn_in) + t for t in gens])
        ratios[name] = dict(zip(gens, sfs_ratio(sfs)))

    path = os.path.join("figures", "sfs.png")
    plot_sfs(ratios, titles, f"Site frequency spectrum after a crash and after growth "
             f"(N = {normal_size} before)", path)
    print(f"Saved {path}")
    for name, by_gen in ratios.items():
        for gen, ratio in by_gen.items():
            print(f"  {name}, generation {gen}: singletons {ratio[0]:.2f} × steady, "
                  f"middle class (i = {sample_size // 2}) {ratio[sample_size // 2 - 1]:.2f} × steady")


def plot_inference(results: dict[str, np.ndarray], kept: dict[str, np.ndarray],
                   truth: tuple[float, float], example: str, title: str, path: str) -> None:
    """
    Plots how often one sample reveals the right kind of history, against time
    since the event, and the inferred crash strength and time for one example.
    Parameters:
    - results (dict): The output of inference.validate.
    - kept (dict): The output of inference.infer for the example sample.
    - truth (tuple[float, float]): The example's true crash strength and time.
    - example (str): A description of the example's true history.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, (ax_rate, ax_post) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE,
                                           gridspec_kw={"width_ratios": [1.15, 1]})
    style_axes(ax_rate)
    # 200-generation bins keep at least a few dozen test samples in every point.
    edges = np.arange(0, 1001, 200)
    centers = (edges[:-1] + edges[1:]) / 2
    groups = [(f"{name} crash", (results["kind"] == 1) & (results["strength"] >= low)
               & (results["strength"] < high), 1, color)
              for (low, high, name), color in zip(inference.STRENGTH_CLASSES, EVENT_COLORS)]
    groups.append(("growth", results["kind"] == 2, 2, STAT_COLORS["pi"]))
    for label, rows, kind, color in groups:
        rates = [np.mean(results["guess"][rows & (results["time"] >= a) & (results["time"] < b)]
                         == kind) * 100
                 if np.any(rows & (results["time"] >= a) & (results["time"] < b)) else np.nan
                 for a, b in zip(edges[:-1], edges[1:])]
        ax_rate.plot(centers, rates, color=color, linewidth=2, marker="o", markersize=3.5,
                     label=label)
    steady_rate = np.mean(results["guess"][results["kind"] == 0] == 0) * 100
    ax_rate.axhline(steady_rate, color=STEADY_COLOR, linewidth=1.5, linestyle="--",
                    label=f"steady samples called steady ({steady_rate:.0f}%)")
    ax_rate.set_ylim(0, 105)
    ax_rate.set_xlim(0, 1000)
    ax_rate.set_xlabel("generations since the event began", color=TEXT_PRIMARY, fontsize=10)
    ax_rate.set_ylabel("% of single samples given the right kind", color=TEXT_PRIMARY,
                       fontsize=10)
    ax_rate.set_title("How long can one sample reveal the past?", loc="left", fontsize=10,
                      color=TEXT_PRIMARY)
    ax_rate.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="lower left")

    # The example: weight of the kept crash rows over strength and time.
    crash = kept["kind"] == 1
    strength = kept["crash_length"][crash] / (2 * kept["crash_size"][crash])
    levels = np.unique(np.concatenate([strength, [truth[0]]]))
    time_edges = np.arange(0, 1001, 50)
    grid = np.zeros((len(levels), len(time_edges) - 1))
    rows = np.searchsorted(levels, strength)
    cols = np.clip(np.searchsorted(time_edges, kept["time"][crash], side="right") - 1,
                   0, len(time_edges) - 2)
    np.add.at(grid, (rows, cols), kept["weight"][crash])
    ax_post.set_facecolor(SURFACE)
    ax_post.pcolormesh(time_edges, np.arange(len(levels) + 1), grid, cmap="Blues",
                       shading="flat")
    ax_post.scatter([truth[1]], [np.searchsorted(levels, truth[0]) + 0.5], marker="*", s=180,
                    color=STAT_COLORS["pi"], edgecolor="white", linewidth=0.8, zorder=3,
                    label="truth")
    ax_post.set_yticks(np.arange(len(levels)) + 0.5, [f"{v:g}" for v in levels])
    ax_post.set_xlabel("generations since the crash began", color=TEXT_PRIMARY, fontsize=10)
    ax_post.set_ylabel("crash strength, length / (2 × size)", color=TEXT_PRIMARY, fontsize=10)
    ax_post.set_title(f"One sample: {example}\nP(crash) = {kept['weight'][crash].sum():.2f}; "
                      "darker = more likely; ★ = truth", loc="left", fontsize=10,
                      color=TEXT_PRIMARY)
    ax_post.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ax_post.spines.values():
        side.set_visible(False)

    fig.suptitle(title, x=0.06, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_inference_figure(normal_size: int, mu: float, sample_size: int) -> None:
    """
    Builds (or loads) the reference table, validates the inference and plots it
    with one worked example (figures/inference.png).
    Parameters:
    - normal_size (int): The population size before any event.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    """
    table = inference.cached_reference_table(normal_size=normal_size, mu=mu,
                                             sample_size=sample_size)
    results = inference.validate(table, num_tests=9000)
    print(inference.describe_validation(results))

    crash_size, length, ago = 20, 20, 200
    sizes = wright_fisher.make_crash_history(normal_size, crash_size, 0, length, ago - length)
    observed = inference.observe(sizes, normal_size, mu, sample_size, seed=10_001)
    kept = inference.infer(observed, table)
    print(inference.describe(inference.summarize(kept)))

    path = os.path.join("figures", "inference.png")
    plot_inference(results, kept, (length / (2 * crash_size), ago),
                   f"crash to N = {crash_size} for {length} generations, {ago} ago",
                   f"Working backwards from one sample of n = {sample_size} sequences "
                   f"(N = {normal_size} before)", path)
    print(f"Saved {path}")


def plot_linkage(spread: dict[str, dict], times: dict[str, dict[str, np.ndarray]],
                 crash_sizes: list[int], example: str, gens_crash: int, title: str,
                 path: str) -> None:
    """
    Compares unlinked and linked sites: Tajima's D over time after one crash (average
    and spread of single samples), and recovery time against crash size.
    Parameters:
    - spread (dict): For "unlinked" and "linked", "gens" and the "mean", "low" and
      "high" (5% and 95% points) of D across populations at each generation.
    - times (dict): For "unlinked" and "linked", the output of recovery_times_over_seeds.
    - crash_sizes (list[int]): The crash sizes, one per column of times.
    - example (str): A description of the crash in the left panel.
    - gens_crash (int): The number of generations each crash lasted.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, (ax_d, ax_rec) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    for ax in (ax_d, ax_rec):
        style_axes(ax)

    ax_d.axvspan(0, gens_crash, color=GRID_COLOR, linewidth=0)
    ax_d.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    for model, color in MODEL_COLORS.items():
        s = spread[model]
        ax_d.fill_between(s["gens"], s["low"], s["high"], color=color, alpha=0.15, linewidth=0)
        ax_d.plot(s["gens"], s["mean"], color=color, linewidth=2,
                  label=f"{model} sites: average")
        ax_d.plot(s["gens"], s["low"], color=color, linewidth=0.8, linestyle=":")
        ax_d.plot(s["gens"], s["high"], color=color, linewidth=0.8, linestyle=":",
                  label=f"{model} sites: middle 90% of single samples")
    ax_d.set_xlim(0, spread["unlinked"]["gens"][-1])
    ax_d.set_xlabel("generations since the crash began", color=TEXT_PRIMARY, fontsize=10)
    ax_d.set_ylabel("Tajima's D", color=TEXT_PRIMARY, fontsize=10)
    ax_d.set_title(f"{example}: same average, wider spread", loc="left", fontsize=10,
                   color=TEXT_PRIMARY)
    ax_d.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="lower right")

    for model, color in MODEL_COLORS.items():
        for stat, style in [("D", "-"), ("pi", "--")]:
            ax_rec.plot(crash_sizes, np.nanmean(times[model][stat], axis=0), color=color,
                        linewidth=2, linestyle=style, marker="o", markersize=5,
                        label=f"{STAT_LABELS[stat]}, {model} sites")
    ax_rec.set_xscale("log")
    ax_rec.set_xticks(crash_sizes)
    ax_rec.set_xticklabels([str(size) for size in crash_sizes])
    ax_rec.minorticks_off()
    ax_rec.set_xlim(crash_sizes[0] * 1.15, crash_sizes[-1] / 1.15)
    ax_rec.set_ylim(bottom=0)
    ax_rec.set_xlabel("population size during the crash (log scale; more severe →)",
                      color=TEXT_PRIMARY, fontsize=10)
    ax_rec.set_ylabel("recovery time (generations since the crash began)",
                      color=TEXT_PRIMARY, fontsize=10)
    ax_rec.set_title(f"Recovery time, 200 populations (average of "
                     f"{len(times['linked']['D'])} seeds)", loc="left", fontsize=10,
                     color=TEXT_PRIMARY)
    ax_rec.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="upper left")

    fig.suptitle(title, x=0.05, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_linkage_figure(normal_size: int, mu: float, sample_size: int, gens_crash: int) -> None:
    """
    Repeats the crash experiments and the inference test with completely linked
    sites, and compares them with unlinked sites (figures/linkage.png).
    Parameters:
    - normal_size (int): The population size before and after each crash.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    - gens_crash (int): The number of generations each crash lasts.
    """
    simulators = {"unlinked": sample_stats_per_trial, "linked": linked_sample_stats}
    burn_in = [normal_size] * (10 * normal_size)

    # Left panel: D after one crash, averaged and spread across populations.
    example_size, gens_after = 10, 1000
    sizes = wright_fisher.make_crash_history(normal_size, example_size, 0, gens_crash,
                                             gens_after)
    gens = list(range(0, gens_crash + gens_after + 1, 10))
    spread = {}
    for model, simulate in simulators.items():
        wright_fisher.set_seed(120)
        d = simulate(burn_in + sizes, mu, sample_size, 1000,
                     [len(burn_in) + t for t in gens])["D"]
        spread[model] = {"gens": gens, "mean": np.nanmean(d, axis=0),
                         "low": np.nanpercentile(d, 5, axis=0),
                         "high": np.nanpercentile(d, 95, axis=0)}
        print(f"  {model}: middle 90% of D before the crash "
              f"{spread[model]['low'][0]:.2f} to {spread[model]['high'][0]:.2f}; "
              f"average D at generation {gens_crash}: {spread[model]['mean'][gens_crash // 10]:.2f}, "
              f"at 200: {spread[model]['mean'][20]:.2f}")

    # Right panel: recovery time against crash size, as in the severity figure.
    crash_sizes = [100, 50, 30, 20, 15, 10]
    seeds = [120, 121, 122, 123, 124]
    histories = {size: wright_fisher.make_crash_history(normal_size, size, 0, gens_crash, 1500)
                 for size in crash_sizes}
    times = {model: recovery_times_over_seeds(normal_size, histories, mu, sample_size,
                                              num_trials=200, record_every=10, seeds=seeds,
                                              simulate=simulate)
             for model, simulate in simulators.items()}
    print(f"{'crash size':>10} " + " ".join(f"{model + ' ' + stat:>14}"
                                           for model in times for stat in ["D", "pi"]))
    for j, size in enumerate(crash_sizes):
        print(f"{size:>10} " + " ".join(f"{np.nanmean(times[model][stat][:, j]):>14.0f}"
                                        for model in times for stat in ["D", "pi"]))

    path = os.path.join("figures", "linkage.png")
    plot_linkage(spread, times, crash_sizes,
                 f"Crash to N = {example_size} for {gens_crash} generations", gens_crash,
                 f"Unlinked sites (the rest of this project) against completely linked sites "
                 f"(N = {normal_size} before)", path)
    print(f"Saved {path}")

    # Working backwards: linked samples, judged with an unlinked or a linked table.
    tables = {model: inference.cached_reference_table(normal_size=normal_size, mu=mu,
                                                      sample_size=sample_size,
                                                      linked=model == "linked")
              for model in simulators}
    print("\nWorking backwards (3000 test samples each; % right kind, % of 90% ranges "
          "holding the truth):")
    for label, reference, tests in [("unlinked samples, unlinked table", "unlinked", None),
                                    ("linked samples, unlinked table", "unlinked", "linked"),
                                    ("linked samples, linked table", "linked", None)]:
        results = inference.validate(tables[reference], num_tests=3000,
                                     tests=None if tests is None else tables[tests])
        r = inference.validation_rates(results)
        print(f"  {label:>33}: steady {r['steady']:.0%}, severe crash in first 300 "
              f"{r['severe_early']:.0%}, growth {r['growth']:.0%}; ranges: strength "
              f"{r['strength_covered']:.0%}, crash time {r['crash_time_covered']:.0%}, "
              f"growth time {r['growth_time_covered']:.0%}")


def plot_wrong_size(true_sizes: list[int], rates: list[dict], assumed_size: int, title: str,
                    path: str) -> None:
    """
    Plots how often the inference gets the right kind of history, and how often its
    ranges hold the truth, when the population's real size differs from the table's.
    Parameters:
    - true_sizes (list[int]): The real population sizes before any event.
    - rates (list[dict]): inference.validation_rates for each true size.
    - assumed_size (int): The population size the reference table assumes.
    - title (str): The figure title.
    - path (str): Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    style_axes(ax)
    lines = [("steady", "steady samples called steady", STEADY_COLOR, "-"),
             ("severe_early", "severe crashes (first 300 generations) called crashes",
              EVENT_COLORS[3], "-"),
             ("growth", "growth called growth", STAT_COLORS["pi"], "-"),
             ("crash_time_covered", "crash time: 90% range holds the truth", EVENT_COLORS[1],
              "--")]
    for key, label, color, style in lines:
        ax.plot(true_sizes, [100 * r[key] for r in rates], color=color, linewidth=2,
                linestyle=style, marker="o", markersize=5, label=label)
    ax.axvline(assumed_size, color=TEXT_SECONDARY, linewidth=0.8)
    ax.text(assumed_size, 103, f" table assumes N = {assumed_size}", fontsize=8.5,
            color=TEXT_SECONDARY, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(true_sizes)
    ax.set_xticklabels([str(size) for size in true_sizes])
    ax.minorticks_off()
    ax.set_ylim(0, 110)
    ax.set_xlabel("real population size before any event (log scale)", color=TEXT_PRIMARY,
                  fontsize=10)
    ax.set_ylabel("% of test samples", color=TEXT_PRIMARY, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, loc="lower left")
    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_wrong_size_figure(normal_size: int, mu: float, sample_size: int) -> None:
    """
    Tests the inference on populations whose real size before any event differs
    from the size the reference table assumes (figures/inference_wrong_size.png).
    Parameters:
    - normal_size (int): The population size the reference table assumes.
    - mu (float): The mutation rate per gene copy per generation.
    - sample_size (int): The number of sampled sequences n.
    """
    table = inference.cached_reference_table(normal_size=normal_size, mu=mu,
                                             sample_size=sample_size)
    true_sizes = [100, 130, 160, 200, 250, 300, 400]
    rates = []
    for true_size in true_sizes:
        # The same grid of histories, starting from the real size (30 populations each).
        tests = inference.build_reference_table(normal_size=true_size, mu=mu,
                                                sample_size=sample_size, num_trials=30,
                                                seed=500)
        rates.append(inference.validation_rates(
            inference.validate(table, num_tests=3000, tests=tests)))
        r = rates[-1]
        print(f"  real N = {true_size}: steady {r['steady']:.0%}, severe crash in first 300 "
              f"{r['severe_early']:.0%}, growth {r['growth']:.0%}; crash time range holds "
              f"the truth {r['crash_time_covered']:.0%}")
    path = os.path.join("figures", "inference_wrong_size.png")
    plot_wrong_size(true_sizes, rates, normal_size,
                    "Working backwards when the population's real size is not the "
                    f"assumed N = {normal_size}", path)
    print(f"Saved {path}")


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

    options = list(figures) + ["severity", "length", "growthsize", "sfs", "inference",
                               "linkage", "wrongsize"]
    chosen = sys.argv[1:] or options
    if any(name not in options for name in chosen):
        sys.exit("usage: python3 plots.py [crash | growth | severity | length | "
                 "growthsize | sfs | inference | linkage | wrongsize]")

    os.makedirs("figures", exist_ok=True)
    if "severity" in chosen:
        make_severity_figure(normal_size, mu, sample_size, event_gens)
    if "length" in chosen:
        make_length_figure(normal_size, mu, sample_size)
    if "growthsize" in chosen:
        make_growth_size_figure(normal_size, mu, sample_size, event_gens)
    if "sfs" in chosen:
        make_sfs_figure(normal_size, mu, sample_size, event_gens)
    if "inference" in chosen:
        make_inference_figure(normal_size, mu, sample_size)
    if "linkage" in chosen:
        make_linkage_figure(normal_size, mu, sample_size, event_gens)
    if "wrongsize" in chosen:
        make_wrong_size_figure(normal_size, mu, sample_size)
    for name in [name for name in chosen if name in figures]:
        wright_fisher.set_seed(120)
        histories, title = figures[name]
        results = compare_histories(normal_size, histories, mu, sample_size, num_trials=200,
                                    record_every=10)
        # Theory for pi, read off at the same generations as the simulation.
        burn_in = [normal_size] * (10 * normal_size)
        record_gens = [len(burn_in) + t for t in next(iter(results.values()))["gens"]]
        theory = {label: expected_sample_pi(burn_in + sizes, mu)[record_gens]
                  for label, sizes in histories.items()}
        theory["steady"] = expected_sample_pi(burn_in + [normal_size] * (event_gens + gens_after),
                                              mu)[record_gens]
        path = os.path.join("figures", f"{name}_recovery.png")
        plot_recovery(results, normal_size, name, event_gens, theta, title, path, theory)
        print(f"Saved {path}")
        for label, result in results.items():
            recovery = {stat: result["recovery"][stat] for stat in ["D", "pi"]}
            print("  " + label + ": " + ", ".join(
                f"{stat} recovers at {gen}" if gen is not None
                else f"{stat} not recovered within {gens_after + event_gens} generations"
                for stat, gen in recovery.items()))


if __name__ == "__main__":
    main()
