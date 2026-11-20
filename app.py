"""
An interactive app for exploring how a population crash or rapid growth changes
genetic variation, and for how long.

The user picks a population size history (steady, crash or growth), how severe
and how long the event is, the mutation rate, the sample size and how many
populations to simulate. The app simulates the event and a steady control
population, samples sequences from every population, and shows
- Tajima's D, pi, Watterson's theta and S averaged over the populations, with
  the generation at which the event can no longer be told apart from the control,
- pi predicted by theory alone, next to the simulated pi,
- the site frequency spectrum at any generation, compared with a steady population's,
- how often a single sample (one study, one population) has a D outside the
  range seen in the steady control, and the spread of D across populations at
  any generation.

Run with: streamlit run app.py

Pseudocode

MakeEventHistory(kind, normalSize, eventSize, eventGens, totalGens)
    if kind is crash
        return MakeCrashHistory(normalSize, eventSize, 0, eventGens, totalGens - eventGens)
    if kind is growth
        return MakeGrowthHistory(normalSize, eventSize, 0, eventGens, totalGens - eventGens)
    return totalGens copies of normalSize

RunScenario(settings)
    burnIn ← 10 * normalSize copies of normalSize
    gens ← 0, recordEvery, 2 * recordEvery, ... up to totalGens
    steadyTrials, steadySFS ← SampleStatsAndSFS(burnIn + steady history, ...) with seed      (stored)
    eventTrials, eventSFS ← SampleStatsAndSFS(burnIn + event history, ...) with seed + 1    (stored)
    steady, event ← the averages of steadyTrials and eventTrials
    recovery ← RecoveryTimes(gens, event, steady)
    theoryPi ← ExpectedSamplePi of each history at the sampled generations
    return gens, steadyTrials, eventTrials, steady, event, recovery, the spectra and theoryPi

FractionSignificant(dValues, lower, upper)
    for every recorded generation
        fraction ← share of populations whose D is below lower or above upper
                   (populations with undefined D count as not significant)
    return the fractions
    (lower and upper come from ControlLimits on the steady control's D values.)
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import inference
import wright_fisher
from animation import animated_histogram_figure, animated_sfs_figure
from diversity import (control_limits, harmonic_sums, sample_stats_and_sfs, sfs_ratio,
                       summarize_trials, tajima_critical_values)
from infinite_sites import expected_sample_pi
from recovery import recovery_times

EVENT_COLOR = "#2a78d6"
STEADY_COLOR = "#8a8984"
STAT_NAMES = {"D": "Tajima's D", "pi": "π", "theta_w": "Watterson's θ", "S": "S"}
STAT_HELP = {
    "D": "Positive when rare variants are missing (after a crash), negative when they are "
         "in excess (after growth), about 0 in a steady population.",
    "pi": "The average number of differences between two sampled sequences. In a steady "
          "population it averages θ = 4Nμ.",
    "theta_w": "S divided by 1 + 1/2 + ... + 1/(n − 1). In a steady population it also "
               "averages θ = 4Nμ.",
    "S": "The number of sites that vary within the sample.",
}


def make_event_history(kind: str, normal_size: int, event_size: int, event_gens: int,
                       total_gens: int) -> list[int]:
    """
    Builds the population size in every generation after the burn-in.
    Parameters:
    - kind (str): "crash", "growth" or "steady".
    - normal_size (int): The population size before the event (and after a crash).
    - event_size (int): The size during a crash, or the size reached by growth.
      Ignored for "steady".
    - event_gens (int): How many generations the crash or the growth lasts.
    - total_gens (int): The number of generations to follow from the start of the event.
    Returns:
    - list[int]: total_gens population sizes.
    """
    if not 0 < event_gens <= total_gens:
        raise ValueError("event_gens must be positive and at most total_gens.")
    if kind == "crash":
        return wright_fisher.make_crash_history(normal_size, event_size, 0, event_gens,
                                                total_gens - event_gens)
    if kind == "growth":
        return wright_fisher.make_growth_history(normal_size, event_size, 0, event_gens,
                                                 total_gens - event_gens)
    if kind == "steady":
        return [normal_size] * total_gens
    raise ValueError(f"Unknown kind of history: {kind}")


def choose_record_every(total_gens: int) -> int:
    """
    Picks how often to sample so that about 100 generations are recorded.
    The recovery cutoff of 4 standard errors was chosen for about 100 points.
    Parameters:
    - total_gens (int): The number of generations followed after the event begins.
    Returns:
    - int: A multiple of 10, at least 10.
    """
    return 10 * max(1, round(total_gens / 1000))


def fraction_significant(d_values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Finds, at each recorded generation, the share of populations whose sample
    has a D outside the range of a steady population.
    Parameters:
    - d_values (np.ndarray): Tajima's D with one row per population and one column
      per recorded generation; nan where D is undefined.
    - lower, upper (float): The limits from control_limits.
    Returns:
    - np.ndarray: One fraction between 0 and 1 per recorded generation. Undefined
      D values count as not significant.
    """
    with np.errstate(invalid="ignore"):
        significant = (d_values < lower) | (d_values > upper)
    return significant.mean(axis=0)


@st.cache_resource(show_spinner=False)
def simulation_store() -> dict:
    """
    Keeps finished simulations between reruns of the app (and between users). A plain
    dictionary is used instead of st.cache_data because a simulation moves a progress
    bar, and st.cache_data would try to replay that bar on later runs, after it is gone.
    Returns:
    - dict: The same dictionary on every call: settings -> simulation result.
    """
    return {}


def simulate(pop_sizes: tuple[int, ...], mu: float, sample_size: int, num_trials: int,
             record_gens: tuple[int, ...], seed: int,
             progress=None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Runs sample_stats_and_sfs with a fixed seed, or returns the stored result if these
    settings were already simulated, so changing only the event's settings does not
    re-simulate the steady control. The 20 most recent results are kept.
    Parameters:
    - pop_sizes, mu, sample_size, num_trials, record_gens: As in sample_stats_and_sfs.
    - seed (int): The random seed to set first.
    - progress: As in sample_stats_and_sfs; not called for a stored result.
    Returns:
    - tuple[dict[str, np.ndarray], np.ndarray]: The output of sample_stats_and_sfs.
    """
    store = simulation_store()
    key = (pop_sizes, mu, sample_size, num_trials, record_gens, seed)
    if key not in store:
        wright_fisher.set_seed(seed)
        store[key] = sample_stats_and_sfs(list(pop_sizes), mu, sample_size, num_trials,
                                          list(record_gens), progress=progress)
        while len(store) > 20:
            del store[next(iter(store))]  # The oldest result (dicts keep insertion order).
    return store[key]


def run_scenario(settings: dict, progress=None) -> dict:
    """
    Simulates the chosen event and a steady control, both after a burn-in of
    10N generations, and compares them.
    Parameters:
    - settings (dict): "kind", "normal_size", "event_size", "event_gens",
      "total_gens", "mu", "sample_size", "num_trials" and "seed".
    - progress: An optional function called with the share done (0 to 1): the
      steady control is the first half, the event the second.
    Returns:
    - dict: "gens" (generations since the event began that were sampled),
      "steady_trials" and "event_trials" (each population's values), "steady" and
      "event" (averages from summarize_trials), "recovery" (from recovery_times),
      "steady_sfs" and "event_sfs" (average site frequency spectra, one row per
      sampled generation) and "steady_theory_pi" and "event_theory_pi" (pi
      predicted by expected_sample_pi at every sampled generation), and
      "event_sizes" (the event's population size in each generation).
    """
    normal_size, total_gens = settings["normal_size"], settings["total_gens"]
    burn_in = [normal_size] * (10 * normal_size)
    gens = list(range(0, total_gens + 1, choose_record_every(total_gens)))
    record_gens = tuple(len(burn_in) + t for t in gens)
    event_sizes = make_event_history(settings["kind"], normal_size, settings["event_size"],
                                     settings["event_gens"], total_gens)
    common = (settings["mu"], settings["sample_size"], settings["num_trials"], record_gens)

    def halves(offset: float):
        # Maps one simulation's progress onto its half of the bar.
        return None if progress is None else (lambda share: progress(offset + share / 2))

    # Different seeds keep the event and control populations independent.
    steady_history = burn_in + [normal_size] * total_gens
    event_history = burn_in + event_sizes
    steady_trials, steady_sfs = simulate(tuple(steady_history), *common, settings["seed"],
                                         progress=halves(0))
    if progress is not None:
        progress(0.5)  # In case the control was already stored and did not run.
    event_trials, event_sfs = simulate(tuple(event_history), *common, settings["seed"] + 1,
                                       progress=halves(0.5))
    steady, event = summarize_trials(steady_trials), summarize_trials(event_trials)
    mu, record = settings["mu"], list(record_gens)
    return {"gens": gens, "steady_trials": steady_trials, "event_trials": event_trials,
            "steady": steady, "event": event, "recovery": recovery_times(gens, event, steady),
            "steady_sfs": steady_sfs, "event_sfs": event_sfs,
            "steady_theory_pi": expected_sample_pi(steady_history, mu)[record],
            "event_theory_pi": expected_sample_pi(event_history, mu)[record],
            "event_sizes": event_sizes}


def color_scale(event_label: str) -> alt.Scale:
    """
    Gives the event and the steady control the same colors in every chart.
    Parameters:
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Scale: A color scale for the "population" field.
    """
    return alt.Scale(domain=[event_label, "steady control"], range=[EVENT_COLOR, STEADY_COLOR])


def dash_scale(event_label: str) -> alt.Scale:
    """
    Draws the steady control dashed, so it is not told apart by color alone.
    Parameters:
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Scale: A stroke dash scale for the "population" field.
    """
    return alt.Scale(domain=[event_label, "steady control"], range=[[1, 0], [5, 4]])


def event_band(settings: dict) -> alt.Chart:
    """
    Shades the generations the event lasted.
    Parameters:
    - settings (dict): As in run_scenario.
    Returns:
    - alt.Chart: A light gray rectangle over the event (empty if there is no event).
    """
    end = 0 if settings["kind"] == "steady" else settings["event_gens"]
    return alt.Chart(pd.DataFrame({"start": [0], "end": [end]})).mark_rect(
        color="#8a8984", opacity=0.15).encode(x="start:Q", x2="end:Q")


def history_chart(settings: dict, event_label: str) -> alt.Chart:
    """
    Draws the population size over time for the event and the steady control.
    Parameters:
    - settings (dict): As in run_scenario.
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Chart: A step chart of population size, on a log scale.
    """
    total_gens = settings["total_gens"]
    event = make_event_history(settings["kind"], settings["normal_size"],
                               settings["event_size"], settings["event_gens"], total_gens)
    lead = max(10, total_gens // 20)
    gens = list(range(-lead, total_gens))
    sizes = [settings["normal_size"]] * lead + event
    frame = pd.concat([
        pd.DataFrame({"generation": gens, "size": sizes, "population": event_label}),
        pd.DataFrame({"generation": gens, "size": settings["normal_size"],
                      "population": "steady control"}),
    ])
    lines = alt.Chart(frame).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("generation:Q", title="generations since the event began",
                scale=alt.Scale(domain=[-lead, total_gens], nice=False)),
        y=alt.Y("size:Q", title="population size N", scale=alt.Scale(type="log")),
        color=alt.Color("population:N", scale=color_scale(event_label), title=None,
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("population:N", scale=dash_scale(event_label), legend=None),
        tooltip=["population", "generation", "size"],
    )
    return (event_band(settings) + lines).properties(height=170)


def stat_chart(result: dict, stat: str, settings: dict, event_label: str) -> alt.Chart:
    """
    Draws the average of one statistic over time for the event and the control,
    with a band of 2 standard errors and a line at the recovery time.
    Parameters:
    - result (dict): The output of run_scenario.
    - stat (str): "D", "pi", "theta_w" or "S".
    - settings (dict): As in run_scenario.
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Chart: The layered chart, with a hover tooltip on every recorded generation.
    """
    gens = result["gens"]
    frames = []
    for label, key in [(event_label, "event"), ("steady control", "steady")]:
        mean = np.array(result[key][stat])
        se = np.array(result[key][stat + "_se"])
        frames.append(pd.DataFrame({"generation": gens, "value": mean, "low": mean - 2 * se,
                                    "high": mean + 2 * se, "population": label}))
    frame = pd.concat(frames)
    name = STAT_NAMES[stat]
    x = alt.X("generation:Q", title="generations since the event began",
              scale=alt.Scale(domain=[0, gens[-1]], nice=False))
    color = alt.Color("population:N", scale=color_scale(event_label), title=None,
                      legend=alt.Legend(orient="top"))
    base = alt.Chart(frame)

    bands = base.mark_area(opacity=0.18).encode(x=x, y="low:Q", y2="high:Q", color=color)
    lines = base.mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("value:Q", title=f"average {name}", scale=alt.Scale(zero=stat != "D")),
        color=color,
        strokeDash=alt.StrokeDash("population:N", scale=dash_scale(event_label), legend=None))
    layers = [event_band(settings), bands, lines]

    if stat == "pi":
        # Theory for both populations, dotted and near-black so it shows on either line.
        theory = pd.concat([
            pd.DataFrame({"generation": gens, "theory": result[f"{key}_theory_pi"],
                          "population": label})
            for label, key in [(event_label, "event"), ("steady control", "steady")]])
        layers.append(alt.Chart(theory).mark_line(
            color="#0b0b0b", strokeWidth=1.2, strokeDash=[1, 3]).encode(
            x=x, y="theory:Q", detail="population:N"))
    if stat == "D":
        layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="#52514e", strokeWidth=0.8).encode(y="y:Q"))
    recovered = result["recovery"][stat]
    if recovered is not None and settings["kind"] != "steady":
        marker = pd.DataFrame({"generation": [recovered],
                               "label": [f"back to normal at {recovered}"]})
        layers.append(alt.Chart(marker).mark_rule(color="#52514e", strokeDash=[2, 3]).encode(
            x="generation:Q"))
        layers.append(alt.Chart(marker).mark_text(align="left", dx=4, dy=8, baseline="top",
                                                  color="#52514e", fontSize=11).encode(
            x="generation:Q", y=alt.value(0), text="label:N"))

    # Hover: a crosshair on the nearest recorded generation and a tooltip with both values.
    wide = pd.DataFrame({"generation": gens,
                         event_label: np.round(result["event"][stat], 3),
                         "steady control": np.round(result["steady"][stat], 3)})
    hover = alt.selection_point(fields=["generation"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    layers.append(alt.Chart(wide).mark_rule(color="#8a8984").encode(
        x="generation:Q",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("generation:Q"), alt.Tooltip(f"{event_label}:Q"),
                 alt.Tooltip("steady control:Q")],
    ).add_params(hover))
    return alt.layer(*layers).properties(height=340)


def significance_chart(result: dict, lower: float, upper: float, settings: dict,
                       event_label: str) -> alt.Chart:
    """
    Draws, over time, the share of single samples whose D is outside the range
    of the steady control.
    Parameters:
    - result (dict): The output of run_scenario.
    - lower, upper (float): The limits of D from control_limits.
    - settings (dict): As in run_scenario.
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Chart: A line chart of percentages, with a reference line at 5%.
    """
    frame = pd.concat([
        pd.DataFrame({"generation": result["gens"], "population": label,
                      "percent": 100 * fraction_significant(result[key]["D"], lower, upper)})
        for label, key in [(event_label, "event_trials"), ("steady control", "steady_trials")]
    ])
    lines = alt.Chart(frame).mark_line(strokeWidth=2).encode(
        x=alt.X("generation:Q", title="generations since the event began",
                scale=alt.Scale(domain=[0, result["gens"][-1]], nice=False)),
        y=alt.Y("percent:Q", title="% of samples with unusual D",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("population:N", scale=color_scale(event_label), title=None,
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("population:N", scale=dash_scale(event_label), legend=None),
        tooltip=["population", "generation", alt.Tooltip("percent:Q", format=".1f")],
    )
    five = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(
        color="#52514e", strokeDash=[2, 3]).encode(y="y:Q")
    return (event_band(settings) + five + lines).properties(height=260)


def event_length(settings: dict) -> int:
    """
    Gives how many generations the event lasted.
    Parameters:
    - settings (dict): As in run_scenario.
    Returns:
    - int: The event's length, or 0 for a steady history.
    """
    return 0 if settings["kind"] == "steady" else settings["event_gens"]


def describe_recovery(gen: int | None, total_gens: int) -> str:
    """
    Turns a recovery time into text for the summary row.
    Parameters:
    - gen (int | None): The recovery time, or None if not recovered.
    - total_gens (int): How long the populations were followed.
    Returns:
    - str: For example "after 640 gens" or "not within 1500 gens".
    """
    return f"after {gen} gens" if gen is not None else f"not within {total_gens} gens"


def sidebar_settings() -> dict:
    """
    Draws the controls in the sidebar and returns what the user picked.
    Returns:
    - dict: The settings, as described in run_scenario.
    """
    sb = st.sidebar
    sb.header("Population history")
    kind = sb.radio("Event", ["crash", "growth", "steady"], horizontal=True,
                    format_func={"crash": "Crash", "growth": "Growth",
                                 "steady": "Steady (no event)"}.get)
    normal_size = sb.select_slider("Normal population size N", [50, 100, 200, 400], value=200,
                                   help="The size during the burn-in and in the steady control.")
    event_size = normal_size
    if kind == "crash":
        event_size = sb.slider("Crash to N =", 2, normal_size - 1,
                               value=max(2, normal_size // 20), key=f"crash_{normal_size}")
    elif kind == "growth":
        options = [m * normal_size for m in [2, 5, 10, 20] if m * normal_size <= 4000]
        event_size = sb.select_slider("Grow to N =", options, value=options[-2],
                                      key=f"growth_{normal_size}",
                                      help="Larger populations take longer to simulate.")
    event_gens = normal_size // 10
    if kind != "steady":
        event_gens = sb.slider(f"{kind.capitalize()} lasts (generations)", 1, 100, value=20)

    sb.header("Genetics and sampling")
    mu = sb.select_slider("Mutation rate μ (per gene copy per generation)",
                          [0.0025, 0.005, 0.0125, 0.025, 0.05], value=0.0125)
    sb.caption(f"θ = 4Nμ = {4 * normal_size * mu:g} for the steady population")
    # A sample cannot hold more sequences than the population has gene copies (2N),
    # and the populations are also sampled during the crash.
    max_sample = 50 if kind != "crash" else min(50, 2 * event_size)
    if max_sample > 4:
        sample_size = sb.slider("Sequences sampled per population (n)", 4, max_sample,
                                value=min(20, max_sample))
    else:
        # Only one choice is left (a slider needs at least two), so no slider is drawn.
        sample_size = max_sample
    if max_sample < 50:
        sb.caption(f"At most {max_sample} sequences: during the crash the population has "
                   f"only 2N = {2 * event_size} gene copies to sample.")

    sb.header("Simulation")
    total_gens = sb.slider("Generations to follow", 200, 3000, value=1500, step=100)
    num_trials = sb.select_slider("Populations to simulate", [50, 100, 200, 400], value=200,
                                  help="More populations detect smaller differences "
                                       "but take longer.")
    seed = int(sb.number_input("Random seed", min_value=0, value=120, step=1))
    return {"kind": kind, "normal_size": normal_size, "event_size": event_size,
            "event_gens": event_gens, "total_gens": total_gens, "mu": mu,
            "sample_size": sample_size, "num_trials": num_trials, "seed": seed}


def show_results(result: dict, settings: dict, event_label: str) -> None:
    """
    Draws the summary row, the averages over time and the single-sample view.
    Parameters:
    - result (dict): The output of run_scenario.
    - settings (dict): The settings result was run with.
    - event_label (str): The legend name of the event population.
    """
    total_gens, n = settings["total_gens"], settings["sample_size"]
    recovery = result["recovery"]
    d = np.array(result["event"]["D"])
    extreme = d[np.nanargmax(np.abs(d - np.array(result["steady"]["D"])))]

    cols = st.columns(4)
    if settings["kind"] == "steady":
        cols[0].metric("Average D", f"{np.nanmean(d):.3f}", help="Should be about 0.")
    else:
        cols[0].metric("D back to normal", describe_recovery(recovery["D"], total_gens))
    cols[1].metric("π back to normal", describe_recovery(recovery["pi"], total_gens))
    cols[2].metric("θ_W back to normal", describe_recovery(recovery["theta_w"], total_gens))
    cols[3].metric("Most extreme average D", f"{extreme:+.2f}")
    if settings["kind"] == "steady":
        st.caption("Both populations are steady, so they should match: every statistic "
                   "should count as back to normal at generation 0.")

    lower, upper = control_limits(result["steady_trials"]["D"])
    tajima_lower, tajima_upper = tajima_critical_values(n)
    # The animations start at the end of the event (the start, for a steady history).
    start = result["gens"].index(min(result["gens"],
                                     key=lambda g: abs(g - event_length(settings))))
    tab_avg, tab_sfs, tab_single, tab_data = st.tabs(
        ["Averages over time", "Site frequency spectrum", "Single samples", "Data"])

    with tab_avg:
        stat = st.radio("Statistic", list(STAT_NAMES), format_func=STAT_NAMES.get,
                        horizontal=True, label_visibility="collapsed")
        theory_note = (" Dotted lines show π predicted by theory alone: each generation "
                       "of size N keeps a share 1 − 1/2N of the variation, and mutation "
                       "adds 2μ." if stat == "pi" else "")
        st.caption(STAT_HELP[stat] + " Shaded bands are ±2 standard errors of the average. "
                   "The event counts as back to normal from the first generation after which "
                   "it stays within 4 standard errors of the control." + theory_note)
        st.altair_chart(stat_chart(result, stat, settings, event_label), width="stretch")

    with tab_sfs:
        st.markdown(
            f"The site frequency spectrum counts how many variable sites are carried by "
            f"1, 2, ..., {n - 1} of the {n} sampled sequences. In a steady population, a "
            "site carried by i sequences is expected in proportion to 1/i, so here each "
            "share is divided by that steady share: **1 means steady**. Too few rare "
            "variants (low i below 1) makes D positive; too many makes D negative.")
        st.plotly_chart(animated_sfs_figure(
            result["gens"], result["event_sizes"], settings["normal_size"], result["event_sfs"],
            result["steady_sfs"], event_label, event_length(settings), start), width="stretch")
        st.caption("Press **▶ Play** to watch the spectrum change, or drag the slider. The "
                   "orange line on the size chart marks the generation shown. Averaged over "
                   "all simulated populations; the steady control wobbles around 1 by chance, "
                   "more for big i, where sites are few.")

    with tab_single:
        if np.isnan(lower):
            st.warning("D is undefined in every steady sample (no segregating sites), so "
                       "there is nothing to compare against. Try a higher mutation rate.")
            return
        st.markdown(
            f"A real study has **one** sample from **one** population. In 95% of the "
            f"steady control's samples (n = {n}), D lies between {lower:.2f} and "
            f"{upper:.2f}. A sample outside that range looks unusual for a steady "
            "population. This is how often that happens in each simulated population.")
        st.altair_chart(significance_chart(result, lower, upper, settings, event_label),
                        width="stretch")
        st.caption(
            "In the steady control this stays near 5% (the dotted line) by design. "
            f"Tajima's standard 5% limits for n = {n} are {tajima_lower:.2f} and "
            f"{tajima_upper:.2f}. They are wider because they assume the sites are linked "
            "on one stretch of DNA, while this simulation lets every site drift "
            "independently, so D varies less from population to population.")
        st.markdown("**The spread of D across populations over time.** Press **▶ Play**: "
                    "the dotted lines are the steady control's 95% range, and samples "
                    "beyond them look unusual.")
        st.plotly_chart(animated_histogram_figure(
            result["gens"], result["event_sizes"], settings["normal_size"],
            result["event_trials"]["D"], result["steady_trials"]["D"], lower, upper,
            event_label, event_length(settings), start), width="stretch")

    with tab_data:
        table = pd.DataFrame({"generation": result["gens"]})
        for key, prefix in [("event", "event"), ("steady", "steady")]:
            for stat in STAT_NAMES:
                table[f"{prefix}_{stat}"] = result[key][stat]
                table[f"{prefix}_{stat}_se"] = result[key][stat + "_se"]
            table[f"{prefix}_pi_theory"] = result[f"{key}_theory_pi"]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download CSV", table.to_csv(index=False), "simulation.csv",
                           "text/csv")


# The reference table used to work backwards (see inference.py).
TABLE_SETTINGS = {"normal_size": 200, "mu": 0.0125, "sample_size": 20}


@st.cache_resource(show_spinner=False)
def reference_table() -> dict[str, np.ndarray]:
    """
    Loads the reference table, building it the first time (about a minute).
    Returns:
    - dict[str, np.ndarray]: The output of inference.cached_reference_table.
    """
    return inference.cached_reference_table(**TABLE_SETTINGS)


def new_mystery(seed: int) -> dict:
    """
    Picks a random history (steady, crash or growth with equal chance) and takes
    one sample from a population with that history.
    Parameters:
    - seed (int): Chooses the history and the simulated population.
    Returns:
    - dict: "label" (the true history in words), "kind", "strength" and "time"
      (nan where they do not apply) and "observed" (the sample's statistics).
    """
    rng = np.random.default_rng(seed)
    normal_size = TABLE_SETTINGS["normal_size"]
    kind = str(rng.choice(inference.KINDS))
    strength, ago = np.nan, np.nan
    if kind == "steady":
        sizes, label = [normal_size] * 100, "steady: no crash or growth"
    elif kind == "crash":
        size, length = int(rng.integers(10, 101)), int(rng.integers(5, 41))
        ago = int(rng.integers(length, 1001))
        sizes = wright_fisher.make_crash_history(normal_size, size, 0, length, ago - length)
        strength = length / (2 * size)
        label = (f"a crash to N = {size} for {length} generations (strength "
                 f"{strength:.2f}), which began {ago} generations ago")
    else:
        end_size, ago = int(rng.integers(400, 4001)), int(rng.integers(20, 1001))
        sizes = wright_fisher.make_growth_history(normal_size, end_size, 0, 20, ago - 20)
        label = f"growth to N = {end_size} over 20 generations, which began {ago} generations ago"
    observed = inference.observe(sizes, normal_size, TABLE_SETTINGS["mu"],
                                 TABLE_SETTINGS["sample_size"], seed=100_000 + seed)
    return {"label": label, "kind": kind, "strength": strength, "time": ago,
            "observed": observed}


def posterior_chart(kept: dict[str, np.ndarray], kind: str) -> alt.Chart:
    """
    Draws how the kept rows of one kind spread over severity and time.
    Parameters:
    - kept (dict): The output of inference.infer.
    - kind (str): "crash" or "growth".
    Returns:
    - alt.Chart: A heat map: crash strength (or size after growth) against
      generations since the event began; darker means more likely.
    """
    rows = kept["kind"] == inference.KINDS.index(kind)
    if kind == "crash":
        level = kept["crash_length"][rows] / (2 * kept["crash_size"][rows])
        level_title = "crash strength, length / (2 × size)"
    else:
        level = kept["end_size"][rows]
        level_title = "size after growth"
    frame = pd.DataFrame({"level": level, "time": kept["time"][rows] // 50 * 50,
                          "weight": kept["weight"][rows]})
    frame = frame.groupby(["level", "time"], as_index=False)["weight"].sum()
    frame["percent"] = 100 * frame["weight"] / frame["weight"].sum()
    frame["level"] = frame["level"].map(lambda v: f"{v:g}")
    order = sorted(frame["level"].unique(), key=float, reverse=True)
    return alt.Chart(frame).mark_rect().encode(
        x=alt.X("time:Q", title="generations since the event began", bin=alt.Bin(step=50),
                scale=alt.Scale(domain=[0, 1000])),
        y=alt.Y("level:O", title=level_title, sort=order),
        color=alt.Color("percent:Q", title="% of this kind", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("level:N", title=level_title),
                 alt.Tooltip("time:Q", title="from generation"),
                 alt.Tooltip("percent:Q", format=".1f", title="% of this kind")],
    ).properties(height=260)


def work_backwards_page() -> None:
    """
    Infers the history of a population from one sample, using the reference table.
    """
    n = TABLE_SETTINGS["sample_size"]
    st.write(
        "Work backwards: from **one sample** of DNA sequences, which history most likely "
        "produced it? The app compares the sample with hundreds of thousands of simulated "
        "samples whose histories are known (steady, crashes to N = 10–100 lasting 5–40 "
        "generations, and growth to N = 400–4000), keeps the 500 most similar, and reports "
        "which histories they came from. Steady, crash and growth start with equal odds. "
        f"The simulated samples use N = {TABLE_SETTINGS['normal_size']} before any event, "
        f"μ = {TABLE_SETTINGS['mu']} and n = {n}, so a sample should come from the same "
        "settings.")

    with st.spinner("Loading the reference table (about a minute the first time)..."):
        table = reference_table()

    source = st.radio("Sample", ["Mystery population", "Enter my own numbers"],
                      horizontal=True, label_visibility="collapsed")
    if source == "Mystery population":
        if st.button("New mystery population") or "mystery" not in st.session_state:
            st.session_state["mystery_seed"] = st.session_state.get("mystery_seed", -1) + 1
            st.session_state["mystery"] = new_mystery(st.session_state["mystery_seed"])
        mystery = st.session_state["mystery"]
        observed = mystery["observed"]
        cols = st.columns(3)
        cols[0].metric("π", f"{observed['pi']:.2f}")
        cols[1].metric("S (variable sites)", f"{observed['S']:.0f}")
        cols[2].metric("Singletons", f"{observed['singletons']:.0f}")
    else:
        cols = st.columns(3)
        observed = {
            "pi": cols[0].number_input("π", min_value=0.0, value=6.0, step=0.1),
            "S": cols[1].number_input("S (variable sites)", min_value=0, value=25, step=1),
            "singletons": cols[2].number_input("Singletons", min_value=0, value=10, step=1),
        }
        if observed["singletons"] > observed["S"]:
            st.error("There cannot be more singletons than variable sites.")
            return
        st.caption(f"For reference, a steady sample averages π ≈ S / "
                   f"{harmonic_sums(n)[0]:.2f} ≈ {4 * TABLE_SETTINGS['normal_size'] * TABLE_SETTINGS['mu']:g}.")

    kept = inference.infer(observed, table)
    summary = inference.summarize(kept)
    probability = summary["probability"]
    cols = st.columns(3)
    for col, kind in zip(cols, inference.KINDS):
        col.metric(f"P({kind})", f"{probability[kind]:.2f}")
    st.text(inference.describe(summary))

    for kind in ["crash", "growth"]:
        if probability[kind] > 0.05:
            st.markdown(f"**If it was {'a crash' if kind == 'crash' else 'growth'}:** "
                        "darker cells are more likely.")
            st.altair_chart(posterior_chart(kept, kind), width="stretch")

    if source == "Mystery population":
        with st.expander("Reveal the truth"):
            st.write(f"This population had {mystery['label']}.")
    st.caption(
        "In tests on simulated samples, severe crashes (strength above 1) are recognized "
        "about 98% of the time in the first 300 generations and weak ones (below 0.1) only "
        "about a third of the time, because a weak crash leaves a sample that looks steady. "
        "The 90% ranges contain the truth about 90–97% of the time.")


def main():
    st.set_page_config(page_title="Crashes, growth and genetic variation", layout="wide")
    st.title("Population crashes, growth and genetic variation")
    st.write("Does a rapid population crash or rapid growth change the genetic variation of "
             "later generations, and for how long? Pick a history in the sidebar, then "
             "simulate it alongside a steady control population (Wright-Fisher model with "
             "infinite-sites mutation, after a burn-in of 10N generations).")

    mode = st.sidebar.radio("Mode", ["Explore an event", "Work backwards"], horizontal=True)
    if mode == "Work backwards":
        work_backwards_page()
        return

    settings = sidebar_settings()
    event_label = {"crash": f"crash to N = {settings['event_size']}",
                   "growth": f"growth to N = {settings['event_size']}",
                   "steady": "second steady population"}[settings["kind"]]
    st.altair_chart(history_chart(settings, event_label), width="stretch")

    if st.button("Run simulation", type="primary"):
        bar = st.progress(0.0, text="Simulating... 0%")
        st.session_state["result"] = run_scenario(
            settings, progress=lambda share: bar.progress(share, text=f"Simulating... {share:.0%}"))
        bar.empty()
        st.session_state["settings"] = dict(settings)
        st.session_state["event_label"] = event_label

    if "result" not in st.session_state:
        st.info("Press **Run simulation** to see the results.")
        return
    if st.session_state["settings"] != settings:
        st.warning("The settings have changed since this run. Press **Run simulation** "
                   "to update the results below.")
    show_results(st.session_state["result"], st.session_state["settings"],
                 st.session_state["event_label"])


if __name__ == "__main__":
    main()
