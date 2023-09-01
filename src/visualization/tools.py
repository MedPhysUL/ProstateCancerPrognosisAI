"""
    @file:              tools.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file contains tools for visualization, mostly for survival analysis. Some of the code
                        was taken from the lifelines library.
"""

from typing import List

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def survival_table_from_events(
    death_times,
    event_observed,
    birth_times=None,
    columns=("removed", "observed", "censored", "entrance", "at_risk")
) -> pd.DataFrame:
    """
    Create a survival table from right-censored dataset.

    Parameters
    ----------
    death_times: (n,) array
      represent the event times
    event_observed: (n,) array
      1 if observed event, 0 is censored event.
    birth_times: a (n,) array, optional
      representing when the subject was first observed. A subject's death event is then at [birth times + duration observed].
      If None (default), birth_times are set to be the first observation or 0, which ever is smaller.
    columns: iterable, optional
      a 3-length array to call the, in order, removed individuals, observed deaths
      and censorships.

    Returns
    -------
    DataFrame
      Pandas DataFrame with index as the unique times or intervals in event_times. The columns named
      'removed' refers to the number of individuals who were removed from the population
      by the end of the period. The column 'observed' refers to the number of removed
      individuals who were observed to have died (i.e. not censored.) The column
      'censored' is defined as 'removed' - 'observed' (the number of individuals who
      left the population due to event_observed)
    """
    removed, observed, censored, entrance, at_risk = columns
    death_times = np.asarray(death_times)
    if birth_times is None:
        birth_times = min(0, death_times.min()) * np.ones(death_times.shape[0])
    else:
        birth_times = np.asarray(birth_times)
        if np.any(birth_times > death_times):
            raise ValueError("birth time must be less than time of death.")

    weights = 1
    df = pd.DataFrame(death_times, columns=["event_at"])
    df[removed] = np.asarray(weights)
    df[observed] = np.asarray(weights) * (np.asarray(event_observed).astype(bool))
    death_table = df.groupby("event_at").sum()
    death_table[censored] = (death_table[removed] - death_table[observed]).astype(int)

    births = pd.DataFrame(birth_times, columns=["event_at"])
    births[entrance] = np.asarray(weights)
    births_table = births.groupby("event_at").sum()
    event_table = death_table.join(births_table, how="outer", sort=True).fillna(0)
    event_table[at_risk] = event_table[entrance].cumsum() - event_table[removed].cumsum().shift(1).fillna(0)

    if (np.asarray(weights).astype(int) != weights).any():
        return event_table.astype(float)

    return event_table.astype(int)


def _create_axis_for_number_at_risk(
        axes: plt.Axes,
        figure: plt.Figure,
        y_position: float
) -> plt.Axes:
    """
    Create a second x-axis to show the number of individuals at risk at each time point.

    Parameters
    ----------
    axes : plt.Axes
        Axes.
    figure : plt.Figure
        Figure.
    y_position : float
        Y position of the axis.

    Returns
    -------
    axes: plt.Axes
        Axes.
    """
    new_axes = plt.twiny(ax=axes)
    ax_height = (axes.get_position().y1 - axes.get_position().y0) * figure.get_figheight()
    new_axes_y_position = y_position / ax_height
    new_axes.spines["bottom"].set_position(("axes", new_axes_y_position - 0.08))

    for side in ["top", "right", "bottom", "left"]:
        new_axes.spines[side].set_visible(False)

    new_axes.xaxis.tick_bottom()

    min_time, max_time = axes.get_xlim()
    new_axes.set_xlim(min_time, max_time)

    x_ticks = [x_tick for x_tick in axes.get_xticks() if min_time <= x_tick <= max_time]

    new_axes.set_xticks(x_ticks)
    new_axes.xaxis.set_ticks_position("none")
    new_axes.yaxis.set_ticks_position("none")

    return new_axes


def add_at_risk_counts(
        survival_tables: List[pd.DataFrame],
        colors: List[str],
        axes: plt.Axes,
        figure: plt.Figure
) -> plt.Axes:
    """
    Add counts showing how many individuals were at risk, censored, and observed, at each time point in
    survival/hazard plots.

    Parameters
    ----------
    survival_tables : List[pd.DataFrame]
        Survival tables. One for each group.
    colors : List[str]
        Colors. One for each group.
    axes : plt.Axes
        Axes.
    figure : plt.Figure
        Figure.

    Returns
    -------
    axes: plt.Axes
        Axes.
    """
    axes_list = [_create_axis_for_number_at_risk(axes, figure, -0.4*(i + 1)) for i in range(len(survival_tables) + 1)]

    rows_to_compute = ["at_risk", "censored", "observed"]
    offsets = {1: 0.055, 2: 0.059, 3: 0.065}
    for n, ax in enumerate(axes_list):
        if n == 0:
            tick_labels = ["" for _ in ax.get_xticks()]
            tick_labels[0] = "Number at risk (Number censored)"
            ax.set_xticklabels(tick_labels, ha="left", weight='bold', fontsize=14)
        else:
            tick_labels = []
            for idx, tick in enumerate(ax.get_xticks()):
                counts = []
                for surv_table in survival_tables:
                    event_table_slice = surv_table.assign(at_risk=lambda x: x.at_risk - x.removed)
                    if not event_table_slice.loc[:tick].empty:
                        event_table_slice = (
                            event_table_slice.loc[:tick, rows_to_compute].agg(
                                {
                                    "at_risk": lambda x: x.tail(1).values,
                                    "censored": "sum",
                                    "observed": "sum",
                                }
                            ).fillna(0)
                        )
                        counts.append([int(c) for c in event_table_slice.loc[rows_to_compute]])

                tick_labels.append(f"{int(counts[n - 1][0])} ({int(counts[n - 1][1])})")

            ax.set_xticklabels(tick_labels, ha="center", fontsize=14)

            if len(survival_tables) > 1:
                ax.get_xticklabels()[0].set_ha("left")
                ax.add_patch(
                    Rectangle(
                        (ax.get_xticks()[0]-0.065, ax.spines["bottom"].get_position()[1]-offsets[len(survival_tables)]),
                        width=0.05, height=0.03, color=colors[n - 1], transform=ax.transAxes, clip_on=False
                    )
                )

    return axes
