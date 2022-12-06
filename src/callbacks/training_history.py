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
from src.data.processing.tools import MaskType


class MeasureCategory(Enum):
    LOSSES = "losses"
    METRICS = "metrics"


class TrainingHistory(Callback):
    """
    This class is used to store losses and score metrics values obtained during the training process.
    """

    def __init__(
            self,
            container: Optional[Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
            **kwargs
    ):
        """
        Initialize the history container.

        Parameters
        ----------
        container : Optional[Dict[str, Dict[str, Dict[str, List[float]]]]]
            History container.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        super().__init__(**kwargs)

        if container:
            self._container = container
        else:
            self._container = {
                MaskType.TRAIN: {
                    MeasureCategory.LOSSES.value: {},
                    MeasureCategory.METRICS.value: {}
                },
                MaskType.VALID: {
                    MeasureCategory.LOSSES.value: {},
                    MeasureCategory.METRICS.value: {}
                }
            }

    def __getitem__(self, item: Union[str, int, slice]) -> dict:
        if isinstance(item, str):
            return self._container[item]
        elif isinstance(item, (int, slice)):
            return self._get_state(self._container, item)

    def __contains__(self, item):
        return item in self._container

    def __iter__(self):
        return iter(self._container)

    def __len__(self):
        return len(self._container)

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
        return self._container[MaskType.TRAIN]

    @property
    def validation_set_history(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Validation set losses and score metrics history.

        Returns
        -------
        history : Dict[str, Dict[str, List[float]]]
            Validation set history.
        """
        return self._container[MaskType.VALID]

    def _get_state(self, container: dict, idx: Union[int, slice]) -> dict:
        """
        Get a specific epoch state dictionary, i.e. the state of the losses and metrics values at the specified epoch
        index.

        Parameters
        ----------
        container : dict
            History container.
        idx : Union[int, slice]
            Epoch index.

        Returns
        -------
        epoch_state : dict
            Epoch dict.
        """
        epoch_state = {}
        for k, v in container.items():
            if isinstance(v, dict):
                epoch_state[k] = self._get_state(v, idx)
            elif isinstance(v, list):
                epoch_state[k] = v[idx]
            else:
                raise TypeError(f"'container' dictionary must contain values of type 'dict' or 'list'. Found "
                                f"{type(v)}.")

        return epoch_state

    def _append_state(self, container: dict, state: dict):
        """
        Append an epoch state dictionary to the history container.

        Parameters
        ----------
        state : dict
            Epoch state.
        """
        for k, v in state.items():
            if isinstance(v, dict):
                self._append_state(container[k], v)
            elif isinstance(v, list):
                values_to_add = state[k]
                if k in container:
                    if isinstance(values_to_add, list):
                        for value in values_to_add:
                            container[k].append(value)
                    else:
                        container[k].append(state[k])
                else:
                    if isinstance(values_to_add, list):
                        container[k] = values_to_add
                    else:
                        container[k] = [state[k]]
            else:
                raise TypeError(f"'container' dictionary must contain values of type 'dict' or 'list'. Found "
                                f"{type(v)}.")

        return container

    def _create_plot(
            self,
            measure_category: Union[MeasureCategory, str],
            **kwargs
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]:
        """
        Create a plot of the given measure category ('losses' or 'metrics') using the training and validation set
        history.

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
                    ax.plot(validation_set_history_dict[key], label="Valid", linewidth=kwargs.get("lw", 3))[0]
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

    def on_epoch_end(self, trainer, **kwargs):
        """
        Append the current train and valid losses and metric scores to the history.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keywords arguments.
        """
        self._append_state(self._container, trainer.state.epoch_losses_and_metrics)


if __name__ == "__main__":
    history = TrainingHistory(
        container={
            "train": {
                "losses": {
                    "Entropy": [1.0, 3],
                    "PN": [1.0, -1]
                },
                "metrics": {
                    "Acc": [1.0, 2.0],
                    "F1": [1.0, 0]
                }
            },
            "valid": {
                "losses": {
                    "BCR": [1.0, 20],
                    "PN": [1.0, 5]
                },
                "metrics": {
                    "Acc": [1.0, 7],
                    "F1": [1.0, 1]
                }
            }
        }
    )

    history.plot(show=True)
