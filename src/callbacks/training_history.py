"""
    @file:              training_history.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 12/2022

    @Description:       This file is used to define the 'TrainingHistory' class which is used to store losses and score
                        metrics values obtained during the training process.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.callbacks.callback import Callback, Priority


class MeasureCategory(Enum):
    LOSSES = "losses"
    METRICS = "metrics"


class TrainingHistory(Callback):
    """
    This class is used to store losses and score metrics values obtained during the training process.
    """

    TRAINING_SET_KEY = "training"
    VALIDATION_SET_KEY = "validation"

    def __init__(
            self,
            container: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
            **kwargs
    ):
        """
        Initialize the histories container.

        Parameters
        ----------
        container : Optional[Dict[str, Dict[str, Dict[str, List[float]]]]]
            Histories container.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        super().__init__(**kwargs)

        if container:
            self._container = container
        else:
            self._container = {
                self.TRAINING_SET_KEY: {
                    MeasureCategory.LOSSES.value: {},
                    MeasureCategory.METRICS.value: {}
                },
                self.VALIDATION_SET_KEY: {
                    MeasureCategory.LOSSES.value: {},
                    MeasureCategory.METRICS.value: {}
                }
            }

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.HIGH_PRIORITY.value

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'CallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
        """
        return False

    @property
    def training_set_history(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Training set losses and score metrics history.

        Returns
        -------
        history : Dict[str, Dict[str, List[float]]]
            Training set history.
        """
        return self._container[self.TRAINING_SET_KEY]

    @property
    def validation_set_history(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Validation set losses and score metrics history.

        Returns
        -------
        history : Dict[str, Dict[str, List[float]]]
            Validation set history.
        """
        return self._container[self.VALIDATION_SET_KEY]

    def _create_plot(
            self,
            measure_category: Union[MeasureCategory, str],
            **kwargs
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]:
        """
        Create a plot of the given measure category ('losses' or 'metrics') using the training and validation set
        histories.

        Parameters
        ----------
        measure_category : MeasureCategory
            Measure category, i.e 'losses' or 'metrics'.
        kwargs : dict
            Keywords arguments controlling figure aesthetics.

        Returns
        -------
        fig, axes, lines : Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]
            The figure, axes and lines of the plot.
        """
        measure_category = MeasureCategory(measure_category).value

        training_set_history_dict = self.training_set_history[measure_category]
        validation_set_history_dict = self.validation_set_history[measure_category]

        training_set_names = list(training_set_history_dict.keys())
        validation_set_names = list(validation_set_history_dict.keys())
        all_names = list(set(training_set_names + validation_set_names))

        axes_dict, lines = {}, {k: [] for k in all_names}
        fig, axes = plt.subplots(nrows=len(all_names), ncols=1, figsize=kwargs.get("figsize", (16, 12)), sharex='all')
        axes = np.ravel(axes)
        for i, (ax, key) in enumerate(zip(axes, all_names)):
            if key in training_set_names:
                lines[key].append(
                    ax.plot(training_set_history_dict[key], label="Train", linewidth=kwargs.get("lw", 3))[0]
                )
            if key in validation_set_names:
                lines[key].append(
                    ax.plot(training_set_history_dict[key], label="Valid", linewidth=kwargs.get("lw", 3))[0]
                )

            ax.set_ylabel(key, fontsize=kwargs.get("fontsize", 16))
            ax.set_xlabel("Epochs [-]", fontsize=kwargs.get("fontsize", 16))
            ax.tick_params(axis='both', which='major', labelsize=kwargs.get("labelsize", 16))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=kwargs.get("fontsize", 16))

            axes_dict[key] = ax

        return fig, axes_dict, lines

    def plot(
            self,
            path_to_save: Optional[str] = None,
            show: bool = False,
            **kwargs
    ) -> None:
        """
        Plot the losses and score metrics contained in the training history.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save the figure.
        show : bool
            Show figure.
        kwargs : dict
            Keywords arguments controlling figure aesthetics.
        """
        plt.close('all')

        for measure_category in [c.value for c in MeasureCategory]:
            fig, axes, lines = self._create_plot(measure_category, **kwargs)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            if path_to_save is not None:
                fig.savefig(f"{path_to_save}_{measure_category}.pdf", dpi=kwargs.get("dpi", 300))
            if show:
                plt.show(block=kwargs.get("block", True))
            if kwargs.get("close", True):
                plt.close(fig)

    def on_train_end(self, trainer, **kwargs):
        """
        Append the current training set losses and metric scores to the history.
        """
        for loss_name, loss_value in trainer.state.train_losses.items():
            self.training_set_history[MeasureCategory.LOSSES.value][loss_name].append(loss_value)

        for metric_name, metric_value in trainer.state.train_metrics.items():
            self.training_set_history[MeasureCategory.METRICS.value][metric_name].append(metric_value)

    def on_validation_end(self, trainer, **kwargs):
        """
        Append the current validation set losses and metric scores to the history.
        """
        for loss_name, loss_value in trainer.state.valid_losses.items():
            self.validation_set_history[MeasureCategory.LOSSES.value][loss_name].append(loss_value)

        for metric_name, metric_value in trainer.state.valid_metrics.items():
            self.validation_set_history[MeasureCategory.METRICS.value][metric_name].append(metric_value)
