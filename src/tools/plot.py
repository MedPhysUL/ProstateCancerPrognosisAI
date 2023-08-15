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
