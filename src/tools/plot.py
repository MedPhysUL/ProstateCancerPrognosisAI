"""
    @file:              plot.py
    @Author:            Félix Desroches

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file is used to define methods related to general matplotlib plots.
"""
from typing import Optional

import matplotlib.pyplot as plt


def terminate_figure(
        fig: Optional[plt.Figure] = None,
        show: bool = True,
        path_to_save: Optional[str] = None,
        **kwargs
) -> None:
    """
    Terminates current figure.

    Parameters
    ----------

    fig : plt.Figure
        Current figure. If no figure is given, will close the opened figure.
    show : bool
        Whether to show figure. Defaults to True
    path_to_save : Optional[str]
        Path to save the figure.
    """
    if fig is not None:
        fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    if show:
        plt.show()

    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def add_details_to_kaplan_meier_curve(
        axes: plt.Axes,
        legend: bool = True
) -> None:
    """
    Adds details to a Kaplan-Meier curve.

    Parameters
    ----------
    axes : plt.Axes
        Axes.
    legend : bool
        Whether to add a legend.
    """
    axes.minorticks_on()
    axes.tick_params(axis="both", direction='in', color="k", which="major", labelsize=16, length=6)
    axes.tick_params(axis="both", direction='in', color="k", which="minor", labelsize=16, length=3)
    axes.set_ylabel(f"Survival probability", fontsize=18)
    axes.set_xlabel("Time $($months$)$", fontsize=18)
    axes.set_xlim(0, None)
    axes.set_ylim(-0.02, 1.02)
    axes.grid(False)
    if legend:
        legend = axes.legend(loc="upper right", edgecolor="k", fontsize=16, handlelength=1.5)

        for line in legend.get_lines():
            line.set_linewidth(8)
