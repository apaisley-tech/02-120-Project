"""
An interactive app for exploring how a population crash or rapid growth changes
genetic variation, and for how long.

The user picks a population size history (steady, crash or growth), how severe
and how long the event is, the mutation rate, the sample size and how many
populations to simulate. The app simulates the event and a steady control
population, samples sequences from every population, and shows
- Tajima's D, pi, Watterson's theta and S averaged over the populations, with
  the generation at which the event can no longer be told apart from the control,
- how often a single sample (one study, one population) would detect the event
  with Tajima's test, and the spread of D across populations at any generation.

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
    steadyTrials ← SampleStatsPerTrial(burnIn + steady history, ...) with seed      (cached)
    eventTrials ← SampleStatsPerTrial(burnIn + event history, ...) with seed + 1    (cached)
    steady, event ← the averages of steadyTrials and eventTrials
    recovery ← RecoveryTimes(gens, event, steady)
    return gens, steadyTrials, eventTrials, steady, event, recovery

FractionSignificant(dValues, lower, upper)
    for every recorded generation
        fraction ← share of populations whose D is below lower or above upper
                   (populations with undefined D count as not significant)
    return the fractions
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import wright_fisher
from diversity import sample_stats_per_trial, summarize_trials, tajima_critical_values
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
    would reject a steady population under Tajima's test.
    Parameters:
    - d_values (np.ndarray): Tajima's D with one row per population and one column
      per recorded generation; nan where D is undefined.
    - lower, upper (float): The critical values from tajima_critical_values.
    Returns:
    - np.ndarray: One fraction between 0 and 1 per recorded generation. Undefined
      D values count as not significant.
    """
    with np.errstate(invalid="ignore"):
        significant = (d_values < lower) | (d_values > upper)
    return significant.mean(axis=0)


@st.cache_data(show_spinner=False, max_entries=20)
def simulate(pop_sizes: tuple[int, ...], mu: float, sample_size: int, num_trials: int,
             record_gens: tuple[int, ...], seed: int) -> dict[str, np.ndarray]:
    """
    Runs sample_stats_per_trial with a fixed seed. Streamlit keeps the result, so
    changing only the event's settings does not re-simulate the steady control.
    Parameters:
    - pop_sizes, mu, sample_size, num_trials, record_gens: As in sample_stats_per_trial.
    - seed (int): The random seed to set first.
    Returns:
    - dict[str, np.ndarray]: The output of sample_stats_per_trial.
    """
    wright_fisher.set_seed(seed)
    return sample_stats_per_trial(list(pop_sizes), mu, sample_size, num_trials,
                                  list(record_gens))


def run_scenario(settings: dict) -> dict:
    """
    Simulates the chosen event and a steady control, both after a burn-in of
    10N generations, and compares them.
    Parameters:
    - settings (dict): "kind", "normal_size", "event_size", "event_gens",
      "total_gens", "mu", "sample_size", "num_trials" and "seed".
    Returns:
    - dict: "gens" (generations since the event began that were sampled),
      "steady_trials" and "event_trials" (each population's values), "steady" and
      "event" (averages from summarize_trials) and "recovery" (from recovery_times).
    """
    normal_size, total_gens = settings["normal_size"], settings["total_gens"]
    burn_in = [normal_size] * (10 * normal_size)
    gens = list(range(0, total_gens + 1, choose_record_every(total_gens)))
    record_gens = tuple(len(burn_in) + t for t in gens)
    event_history = make_event_history(settings["kind"], normal_size, settings["event_size"],
                                       settings["event_gens"], total_gens)
    common = (settings["mu"], settings["sample_size"], settings["num_trials"], record_gens)

    # Different seeds keep the event and control populations independent.
    steady_trials = simulate(tuple(burn_in + [normal_size] * total_gens), *common,
                             settings["seed"])
    event_trials = simulate(tuple(burn_in + event_history), *common, settings["seed"] + 1)
    steady, event = summarize_trials(steady_trials), summarize_trials(event_trials)
    return {"gens": gens, "steady_trials": steady_trials, "event_trials": event_trials,
            "steady": steady, "event": event, "recovery": recovery_times(gens, event, steady)}


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
    Draws, over time, the share of single samples in which Tajima's test detects
    a departure from a steady population.
    Parameters:
    - result (dict): The output of run_scenario.
    - lower, upper (float): The critical values of D.
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
        y=alt.Y("percent:Q", title="% of samples where D is significant",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("population:N", scale=color_scale(event_label), title=None,
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("population:N", scale=dash_scale(event_label), legend=None),
        tooltip=["population", "generation", alt.Tooltip("percent:Q", format=".1f")],
    )
    five = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(
        color="#52514e", strokeDash=[2, 3]).encode(y="y:Q")
    return (event_band(settings) + five + lines).properties(height=260)


def histogram_chart(result: dict, index: int, lower: float, upper: float,
                    event_label: str) -> alt.Chart:
    """
    Draws the spread of Tajima's D across populations at one recorded generation,
    one panel for the event and one for the control, with the critical values marked.
    Parameters:
    - result (dict): The output of run_scenario.
    - index (int): Which recorded generation to show.
    - lower, upper (float): The critical values of D.
    - event_label (str): The legend name of the event population.
    Returns:
    - alt.Chart: Two histograms sharing the D axis.
    """
    values = {label: result[key]["D"][:, index]
              for label, key in [(event_label, "event_trials"), ("steady control", "steady_trials")]}
    finite = np.concatenate([v[~np.isnan(v)] for v in values.values()])
    low = min(lower, finite.min()) if finite.size else lower
    high = max(upper, finite.max()) if finite.size else upper
    edges = np.linspace(np.floor(low * 4) / 4, np.ceil(high * 4) / 4,
                        int(round((np.ceil(high * 4) - np.floor(low * 4)))) + 1)

    frames = []
    for label, v in values.items():
        counts, _ = np.histogram(v[~np.isnan(v)], bins=edges)
        frames.append(pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts,
                                    "population": label}))
    frame = pd.concat(frames)
    x = alt.X("start:Q", title="Tajima's D of one sample",
              scale=alt.Scale(domain=[edges[0], edges[-1]], nice=False))
    bars = alt.Chart().mark_bar(stroke="white", strokeWidth=1, cornerRadiusTopLeft=2,
                                cornerRadiusTopRight=2).encode(
        x=x, x2="end:Q", y=alt.Y("count:Q", title="populations"),
        color=alt.Color("population:N", scale=color_scale(event_label), legend=None),
        tooltip=["population", alt.Tooltip("start:Q", title="D from"),
                 alt.Tooltip("end:Q", title="D to"), "count"],
    )
    limits = alt.Chart(pd.DataFrame({"x": [lower, upper]})).mark_rule(
        color="#52514e", strokeDash=[2, 3]).encode(x="x:Q")
    return alt.layer(bars, limits, data=frame).properties(height=130).facet(
        row=alt.Row("population:N", title=None, sort=[event_label, "steady control"],
                    header=alt.Header(labelAngle=0, labelAlign="left", labelOrient="top")))


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
    sample_size = sb.slider("Sequences sampled per population (n)", 4, 50, value=20)

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

    lower, upper = tajima_critical_values(n)
    tab_avg, tab_single, tab_data = st.tabs(["Averages over time", "Single samples", "Data"])

    with tab_avg:
        stat = st.radio("Statistic", list(STAT_NAMES), format_func=STAT_NAMES.get,
                        horizontal=True, label_visibility="collapsed")
        st.caption(STAT_HELP[stat] + " Shaded bands are ±2 standard errors of the average. "
                   "The event counts as back to normal from the first generation after which "
                   "it stays within 4 standard errors of the control.")
        st.altair_chart(stat_chart(result, stat, settings, event_label), width="stretch")

    with tab_single:
        st.markdown(
            f"A real study has **one** sample from **one** population. With n = {n}, "
            f"Tajima's test flags a sample as not steady when D < {lower:.2f} or "
            f"D > {upper:.2f} (5% level, Tajima 1989). This is how often that happens "
            "in each simulated population.")
        st.altair_chart(significance_chart(result, lower, upper, settings, event_label),
                        width="stretch")
        st.caption("In the steady control this should stay at or below about 5% "
                   "(the dotted line).")
        gen = st.select_slider("Show the spread of D at generation", result["gens"],
                               value=min(result["gens"], key=lambda g: abs(g - settings["event_gens"])))
        st.altair_chart(histogram_chart(result, result["gens"].index(gen), lower, upper,
                                        event_label), width="stretch")

    with tab_data:
        table = pd.DataFrame({"generation": result["gens"]})
        for key, prefix in [("event", "event"), ("steady", "steady")]:
            for stat in STAT_NAMES:
                table[f"{prefix}_{stat}"] = result[key][stat]
                table[f"{prefix}_{stat}_se"] = result[key][stat + "_se"]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download CSV", table.to_csv(index=False), "simulation.csv",
                           "text/csv")


def main():
    st.set_page_config(page_title="Crashes, growth and genetic variation", layout="wide")
    st.title("Population crashes, growth and genetic variation")
    st.write("Does a rapid population crash or rapid growth change the genetic variation of "
             "later generations, and for how long? Pick a history in the sidebar, then "
             "simulate it alongside a steady control population (Wright-Fisher model with "
             "infinite-sites mutation, after a burn-in of 10N generations).")

    settings = sidebar_settings()
    event_label = {"crash": f"crash to N = {settings['event_size']}",
                   "growth": f"growth to N = {settings['event_size']}",
                   "steady": "second steady population"}[settings["kind"]]
    st.altair_chart(history_chart(settings, event_label), width="stretch")

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating... (about 5–30 seconds)"):
            st.session_state["result"] = run_scenario(settings)
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
