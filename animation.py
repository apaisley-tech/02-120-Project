"""
Builds the app's animated charts with Plotly. Each chart plays through the
recorded generations, one frame per generation, with play and pause buttons and
a slider to scrub through time. A small chart of population size above the main
one has a marker that moves to the current generation, so it is always clear
whether a frame is before, during or after the event.

The animation runs in the browser, so it plays smoothly without re-running the app.

Pseudocode

AnimatedFigure(gens, sizes, frames, start)
    top panel: population size over time for the event and the control, with a
               vertical marker at the current generation
    bottom panel: the traces of the starting frame
    for every recorded generation g
        add a frame: the marker at g and the bottom panel's traces for g
    add play and pause buttons and a slider with one step per generation

SFSFrames(eventSFS, steadySFS)
    for every recorded generation: SFSRatio of the event and of the control against i

HistogramFrames(eventD, steadyD, lower, upper)
    edges ← bins of width 0.25 covering every D value at every generation
    for every recorded generation: the number of populations in each bin, for both
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diversity import sfs_ratio

EVENT_COLOR = "#2a78d6"
STEADY_COLOR = "#8a8984"
MARKER_COLOR = "#eb6834"
FRAME_MS = 250
TRANSITION_MS = 150


def _base_figure(gens: list[int], event_sizes: list[int], normal_size: int,
                 event_label: str, event_gens: int, main_traces: list,
                 start: int) -> go.Figure:
    """
    Makes the two-panel figure: population size over time on top, the main chart below.
    Parameters:
    - gens (list[int]): The recorded generations since the event began.
    - event_sizes (list[int]): The event population's size in each generation.
    - normal_size (int): The steady control's size.
    - event_label (str): The legend name of the event population.
    - event_gens (int): How long the event lasted (0 for no event), to shade it.
    - main_traces (list): The bottom panel's traces at the starting frame.
    - start (int): The index of the starting generation.
    Returns:
    - go.Figure: The figure, with the time marker as trace 2 and the main traces after it.
    """
    fig = make_subplots(rows=2, cols=1, row_heights=[0.26, 0.74], vertical_spacing=0.16)
    # The last recorded generation can be one past the end of the history; extend it.
    event_sizes = list(event_sizes) + [event_sizes[-1]] * max(0, gens[-1] + 1 - len(event_sizes))
    generations = list(range(len(event_sizes)))
    fig.add_trace(go.Scatter(x=generations, y=event_sizes, mode="lines", name=event_label,
                             line={"color": EVENT_COLOR, "width": 2, "shape": "hv"},
                             legendgroup="event", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, generations[-1]], y=[normal_size, normal_size],
                             mode="lines", name="steady control", legendgroup="steady",
                             line={"color": STEADY_COLOR, "width": 1.5, "dash": "dash"},
                             hoverinfo="skip"), row=1, col=1)
    low, high = min(event_sizes + [normal_size]), max(event_sizes + [normal_size])
    fig.add_trace(_marker(gens[start], low, high), row=1, col=1)
    for trace in main_traces:
        fig.add_trace(trace, row=2, col=1)

    if event_gens > 0:
        fig.add_vrect(x0=0, x1=event_gens, fillcolor=STEADY_COLOR, opacity=0.15, line_width=0,
                      row=1, col=1)
    fig.update_yaxes(type="log", title_text="size N", row=1, col=1,
                     range=[np.log10(low) - 0.15, np.log10(high) + 0.15])
    fig.update_xaxes(title_text="generations since the event began", row=1, col=1,
                     range=[0, generations[-1]])
    return fig


def _marker(gen: int, low: float, high: float) -> go.Scatter:
    """
    Makes the vertical line that marks the current generation on the size chart.
    Parameters:
    - gen (int): The current generation.
    - low, high (float): The smallest and largest population size, so the line spans both.
    Returns:
    - go.Scatter: The marker line, labeled with the generation.
    """
    return go.Scatter(x=[gen, gen], y=[low, high], mode="lines+text",
                      text=["", f"  generation {gen}"], textposition="middle right",
                      textfont={"color": MARKER_COLOR, "size": 12},
                      line={"color": MARKER_COLOR, "width": 3}, showlegend=False,
                      hoverinfo="skip")


def _add_animation(fig: go.Figure, gens: list[int], frames: list[go.Frame],
                   start: int) -> go.Figure:
    """
    Adds the frames, play and pause buttons, and a generation slider.
    Parameters:
    - fig (go.Figure): The figure from _base_figure.
    - gens (list[int]): The recorded generations, one per frame.
    - frames (list[go.Frame]): One frame per generation, named by the generation.
    - start (int): The index of the starting frame, where the slider starts.
    Returns:
    - go.Figure: The same figure, animated.
    """
    fig.frames = frames
    play = {"frame": {"duration": FRAME_MS, "redraw": True}, "fromcurrent": True,
            "transition": {"duration": TRANSITION_MS, "easing": "cubic-in-out"}, "mode": "immediate"}
    pause = {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
             "transition": {"duration": 0}}
    fig.update_layout(
        updatemenus=[{"type": "buttons", "direction": "left", "showactive": False,
                      "x": 0, "y": -0.13, "xanchor": "left", "yanchor": "top",
                      "pad": {"t": 0, "r": 10},
                      "buttons": [{"label": "▶ Play", "method": "animate", "args": [None, play]},
                                  {"label": "❚❚ Pause", "method": "animate",
                                   "args": [[None], pause]}]}],
        sliders=[{"active": start, "x": 0.17, "len": 0.83, "y": -0.08, "yanchor": "top",
                  "currentvalue": {"prefix": "generation ", "font": {"size": 12}},
                  "pad": {"t": 0},
                  "steps": [{"label": str(gen), "method": "animate",
                             "args": [[str(gen)], {**play, "frame": {"duration": 0,
                                                                     "redraw": True}}]}
                            for gen in gens]}],
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 60, "r": 20, "t": 40, "b": 110},
        height=620,
    )
    return fig


def animated_sfs_figure(gens: list[int], event_sizes: list[int], normal_size: int,
                        event_sfs: np.ndarray, steady_sfs: np.ndarray, event_label: str,
                        event_gens: int, start: int) -> go.Figure:
    """
    Animates the shape of the site frequency spectrum over the recorded generations.
    Parameters:
    - gens (list[int]): The recorded generations since the event began.
    - event_sizes (list[int]): The event population's size in each generation.
    - normal_size (int): The steady control's size.
    - event_sfs, steady_sfs (np.ndarray): Average spectra, one row per recorded generation.
    - event_label (str): The legend name of the event population.
    - event_gens (int): How long the event lasted (0 for no event).
    - start (int): The index of the generation shown first.
    Returns:
    - go.Figure: The animated figure.
    """
    event_ratio, steady_ratio = sfs_ratio(event_sfs), sfs_ratio(steady_sfs)
    classes = np.arange(1, event_sfs.shape[1] + 1)

    def traces(index: int) -> list[go.Scatter]:
        return [go.Scatter(x=classes, y=event_ratio[index], mode="lines+markers",
                           name=event_label, legendgroup="event", showlegend=False,
                           line={"color": EVENT_COLOR, "width": 2.5},
                           hovertemplate="i = %{x}: %{y:.2f} × steady share<extra></extra>"),
                go.Scatter(x=classes, y=steady_ratio[index], mode="lines+markers",
                           name="steady control", legendgroup="steady", showlegend=False,
                           line={"color": STEADY_COLOR, "width": 1.5, "dash": "dash"},
                           hovertemplate="i = %{x}: %{y:.2f} × steady share<extra></extra>")]

    fig = _base_figure(gens, event_sizes, normal_size, event_label, event_gens, traces(start),
                       start)
    # Fixed axes for every frame, so changes in the curves are not hidden by rescaling.
    values = np.concatenate([event_ratio.ravel(), steady_ratio.ravel()])
    values = values[np.isfinite(values) & (values > 0)]
    low = min(0.25, values.min()) if values.size else 0.25
    high = max(4.0, values.max()) if values.size else 4.0
    fig.update_yaxes(type="log", range=[np.log10(low) - 0.05, np.log10(high) + 0.05],
                     title_text="share of variable sites ÷ steady share", row=2, col=1)
    fig.update_xaxes(title_text="sequences carrying the mutant, i", row=2, col=1,
                     range=[0.5, classes[-1] + 0.5], dtick=1 if len(classes) <= 20 else 5)
    fig.add_hline(y=1, line={"color": "#52514e", "dash": "dot", "width": 1}, row=2, col=1)

    low_size, high_size = min(event_sizes + [normal_size]), max(event_sizes + [normal_size])
    frames = [go.Frame(name=str(gen), traces=[2, 3, 4],
                       data=[_marker(gen, low_size, high_size)] + traces(i))
              for i, gen in enumerate(gens)]
    return _add_animation(fig, gens, frames, start)


def histogram_edges(values: list[np.ndarray], lower: float, upper: float,
                    width: float = 0.25) -> np.ndarray:
    """
    Picks bin edges of a fixed width that cover every value and both limits.
    Parameters:
    - values (list[np.ndarray]): Arrays of D values, nan allowed.
    - lower, upper (float): Limits that must lie inside the bins.
    - width (float): The bin width.
    Returns:
    - np.ndarray: The bin edges, at multiples of width.
    """
    finite = np.concatenate([np.ravel(v) for v in values])
    finite = finite[np.isfinite(finite)]
    low = min([lower] + ([finite.min()] if finite.size else []))
    high = max([upper] + ([finite.max()] if finite.size else []))
    start, end = np.floor(low / width) * width, np.ceil(high / width) * width
    if end <= start:
        end = start + width
    return np.arange(start, end + width / 2, width)


def animated_histogram_figure(gens: list[int], event_sizes: list[int], normal_size: int,
                              event_d: np.ndarray, steady_d: np.ndarray, lower: float,
                              upper: float, event_label: str, event_gens: int,
                              start: int) -> go.Figure:
    """
    Animates the spread of Tajima's D across populations over the recorded generations.
    Parameters:
    - gens (list[int]): The recorded generations since the event began.
    - event_sizes (list[int]): The event population's size in each generation.
    - normal_size (int): The steady control's size.
    - event_d, steady_d (np.ndarray): D of every population (rows) at every recorded
      generation (columns); nan where D is undefined.
    - lower, upper (float): The limits of D from the steady control, drawn as lines.
    - event_label (str): The legend name of the event population.
    - event_gens (int): How long the event lasted (0 for no event).
    - start (int): The index of the generation shown first.
    Returns:
    - go.Figure: The animated figure.
    """
    edges = histogram_edges([event_d, steady_d], lower, upper)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]

    def counts(d: np.ndarray, index: int) -> np.ndarray:
        column = d[:, index]
        return np.histogram(column[np.isfinite(column)], bins=edges)[0]

    def traces(index: int) -> list[go.Bar]:
        return [go.Bar(x=centers, y=counts(steady_d, index), width=width,
                       name="steady control", legendgroup="steady", showlegend=False,
                       marker={"color": STEADY_COLOR}, opacity=0.55,
                       hovertemplate="D ≈ %{x:.2f}: %{y} populations<extra>steady</extra>"),
                go.Bar(x=centers, y=counts(event_d, index), width=width,
                       name=event_label, legendgroup="event", showlegend=False,
                       marker={"color": EVENT_COLOR}, opacity=0.65,
                       hovertemplate="D ≈ %{x:.2f}: %{y} populations<extra>event</extra>")]

    fig = _base_figure(gens, event_sizes, normal_size, event_label, event_gens, traces(start),
                       start)
    tallest = max(max(counts(d, i).max() for i in range(len(gens)))
                  for d in [event_d, steady_d])
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(title_text="populations", range=[0, tallest * 1.08], row=2, col=1)
    fig.update_xaxes(title_text="Tajima's D of one sample", range=[edges[0], edges[-1]],
                     row=2, col=1)
    for limit in [lower, upper]:
        fig.add_vline(x=limit, line={"color": "#52514e", "dash": "dot", "width": 1.2},
                      row=2, col=1)

    low_size, high_size = min(event_sizes + [normal_size]), max(event_sizes + [normal_size])
    frames = [go.Frame(name=str(gen), traces=[2, 3, 4],
                       data=[_marker(gen, low_size, high_size)] + traces(i))
              for i, gen in enumerate(gens)]
    return _add_animation(fig, gens, frames, start)
