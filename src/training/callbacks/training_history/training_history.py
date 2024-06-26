"""
    @file:              training_history.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the 'TrainingHistory' class which is used to store losses and score
                        metrics values obtained during the training process.
"""

from dataclasses import asdict
from itertools import count
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from dacite import from_dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..base import Priority, TrainingCallback
from .containers import HistoryContainer, MeasurementsContainer


class TrainingHistory(TrainingCallback):
    """
    This class is used to store losses and score metrics values obtained during the training process.
    """

    instance_counter = count()

    def __init__(
            self,
            container: Optional[Union[Dict[str, MeasurementsContainer], HistoryContainer]] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the history container.

        Parameters
        ----------
        container : Optional[Union[Dict[str, MeasurementsContainer], HistoryContainer]]
            History container.
        name : Optional[str]
            The name of the callback.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, **kwargs)

        if container:
            if isinstance(container, HistoryContainer):
                self.container = container
            elif isinstance(container, dict):
                self.container = from_dict(data_class=HistoryContainer, data=container)
            else:
                raise TypeError(f"'container' must be of type 'HistoryContainer' or 'dict'. Found {type(container)}.")
        else:
            self.container = HistoryContainer()

    def __contains__(self, item: Any) -> bool:
        """
        Whether container contains given item.

        Parameters
        ----------
        item : Any
            A given item.

        Returns
        -------
        contains : bool
            Whether container contains given item.
        """
        return item in self.container

    def __getitem__(
            self,
            item: Union[str, int, slice]
    ) -> Union[HistoryContainer, MeasurementsContainer]:
        """
        Gets an item from the history container.

        Parameters
        ----------
        item : Union[str, int, slice]
            An item. If the item is a 'str', returns the 'MeasurementsContainer' associated to that string. If the item
            is an 'int' or a 'slice', returns the entire 'HistoryContainer' at the given epoch (int) or epochs (slice).

        Returns
        -------
        reduced_container : Union[HistoryContainer, MeasurementsContainer]
            Reduced container.
        """
        if isinstance(item, str):
            return from_dict(data_class=MeasurementsContainer, data=asdict(self.container)[item])
        elif isinstance(item, (int, slice)):
            return from_dict(data_class=HistoryContainer, data=self._get_state(asdict(self.container), item))

    @property
    def is_empty(self) -> bool:
        """
        Whether training history is empty.

        Returns
        -------
        is_empty : bool
            Whether training history is empty.
        """
        return self.container == HistoryContainer()

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.HIGH_PRIORITY

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'TrainingCallbackList'.

        Returns
        -------
        allow : bool
            Whether to allow duplicates.
        """
        return False

    @property
    def training_set_measurements(self) -> MeasurementsContainer:
        """
        Training set measurements.

        Returns
        -------
        measurements : MeasurementsContainer
            Training set measurements.
        """
        return self.container.train

    @property
    def validation_set_measurements(self) -> MeasurementsContainer:
        """
        Validation set measurements.

        Returns
        -------
        measurements : MeasurementsContainer
            Validation set measurements.
        """
        return self.container.valid

    def _get_state(self, container: dict, idx: Union[int, slice]) -> Dict[str, MeasurementsContainer]:
        """
        Gets a specific epoch state dictionary, i.e. the state of the losses and metrics values at the specified epoch
        index.

        Parameters
        ----------
        container : dict
            History container.
        idx : Union[int, slice]
            Epoch index.

        Returns
        -------
        epoch_state : Dict[str, MeasurementsContainer]
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

    def _init(self, container: dict, state: dict):
        """
        Initializes container using the first epoch state.

        Parameters
        ----------
        container: dict
            History container.
        state : dict
            Epoch state.
        """
        for k, v in state.items():
            if k in container:
                container[k] = {}
                for measurement_type_name, tasks in v.items():
                    container[k][measurement_type_name] = {}
                    for task_name, measurements in tasks.items():
                        container[k][measurement_type_name][task_name] = {}
                        for measurement_name, value in measurements.items():
                            container[k][measurement_type_name][task_name][measurement_name] = [value]

        self.container = from_dict(data_class=HistoryContainer, data=container)

    def _append_state(self, container: dict, state: dict):
        """
        Appends an epoch state dictionary to the history container.

        Parameters
        ----------
        container: dict
            History container.
        state : dict
            Epoch state.
        """
        for k, v in state.items():
            if isinstance(v, dict):
                self._append_state(container[k], v)
            elif isinstance(v, list):
                if k in container:
                    for value in v:
                        container[k].append(value)
                else:
                    container[k] = v
            elif isinstance(v, (int, float)):
                if k in container:
                    container[k].append(state[k])
                else:
                    container[k] = [state[k]]
            else:
                raise TypeError(f"'container' dictionary must contain values of type 'dict', 'list', 'int' or 'float'. "
                                f"Found {type(v)}.")

        self.container = from_dict(data_class=HistoryContainer, data=container)

    def _create_plot(
            self,
            measurement_category: str,
            task_key: str,
            **kwargs
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]:
        """
        Creates a plot of the given measurement category using the training and validation set measurements history.

        Parameters
        ----------
        measurement_category : str
            Measurement category, i.e 'multi_task_losses', 'single_task_losses' or 'metrics'.
        task_key : str
            Specific task key in the measurement category history.
        kwargs : dict
            Keywords arguments controlling figure aesthetics.

        Returns
        -------
        fig, axes, lines : Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]
            The figure, axes and lines of the plot.
        """
        train_measurement_history = asdict(self.training_set_measurements)[measurement_category]
        valid_measurement_history = asdict(self.validation_set_measurements)[measurement_category]

        if task_key in train_measurement_history:
            train_task_history = train_measurement_history[task_key]
            train_tasks = list(train_task_history.keys())
        else:
            train_task_history, train_tasks = [], []

        if task_key in valid_measurement_history:
            valid_task_history = valid_measurement_history[task_key]
            valid_tasks = list(valid_task_history.keys())
        else:
            valid_task_history, valid_tasks = [], []

        all_tasks = list(set(train_tasks + valid_tasks))

        axes_dict, lines = {}, {k: [] for k in all_tasks}
        fig, axes = plt.subplots(nrows=len(all_tasks), ncols=1, figsize=kwargs.get("figsize", (16, 12)), sharex='all')
        fig.suptitle(f"{task_key}", fontsize=kwargs.get("title_fontsize", 20))
        axes = np.ravel(axes)
        for i, (ax, key) in enumerate(zip(axes, all_tasks)):
            if key in train_tasks:
                lines[key].append(
                    ax.plot(
                        train_task_history[key],
                        label=kwargs.get("train_label", "train"),
                        linewidth=kwargs.get("lw", 3)
                    )[0]
                )
            if key in valid_tasks:
                lines[key].append(
                    ax.plot(
                        valid_task_history[key],
                        label=kwargs.get("valid_label", "valid"),
                        linewidth=kwargs.get("lw", 3)
                    )[0]
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
            Path to save the figures. Should be a directory.
        show : bool
            Show figure.
        kwargs : dict
            Keywords arguments controlling figure aesthetics.
        """
        plt.close('all')

        for measurement_category in list(MeasurementsContainer.__annotations__.keys()):
            train_history = asdict(self.training_set_measurements)[measurement_category]
            valid_history = asdict(self.validation_set_measurements)[measurement_category]

            training_set_task_keys = list(train_history.keys())
            validation_set_task_keys = list(valid_history.keys())
            all_task_keys = list(set(training_set_task_keys + validation_set_task_keys))

            if path_to_save is not None:
                path_to_measurement_category_directory = os.path.join(path_to_save, measurement_category)
                os.mkdir(path_to_measurement_category_directory)

            for task_key in all_task_keys:
                if set(list(train_history[task_key].keys()) + list(valid_history[task_key].keys())):
                    fig, axes, lines = self._create_plot(measurement_category, task_key, **kwargs)
                    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
                    if path_to_save is not None:
                        path_to_fig = os.path.join(os.path.join(path_to_save, measurement_category), f"{task_key}.pdf")
                        fig.savefig(path_to_fig, dpi=kwargs.get("dpi", 300))
                    if show:
                        plt.show(block=kwargs.get("block", True))
                    if kwargs.get("close", True):
                        plt.close(fig)

    def on_epoch_end(self, trainer, **kwargs):
        """
        Appends the current epoch state to the history.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keywords arguments.
        """
        if trainer.epoch_state.idx == 0:
            self._init(asdict(self.container), asdict(trainer.epoch_state))
        else:
            self._append_state(asdict(self.container), asdict(trainer.epoch_state))


if __name__ == "__main__":
    history = TrainingHistory(
        container=HistoryContainer(
            train=MeasurementsContainer(
                single_task_metrics={
                    "PN_classification": {
                        "Accuracy": [0.6, 0.5, 0.9],
                        "AUC": [0.6, 0.7, 0.75]
                    },
                    "Prostate_segmentation": {
                        "DICEMetric": [0.7, 0.76, 0.85]
                    },
                },
                multi_task_losses={
                    "LearningAlgorithm_0": {
                        "mean_loss_without_regularization": [0.9, 0.7, 0.6],
                        "mean_loss_with_regularization": [1.1, 0.8, 0.7]
                    },
                    "LearningAlgorithm_1": {
                        "median_loss_without_regularization": [1.1, 0.8, 0.7],
                        "median_loss_with_regularization": [2, 1.5, 1.1]
                    },
                },
                single_task_losses={
                    "PN_classification": {
                        "BinaryBalancedAccuracy": [0.6, 0.5, 0.9]
                    },
                    "Prostate_segmentation": {
                        "DICELoss": [0.7, 0.76, 0.85]
                    },
                }
            ),
            valid=MeasurementsContainer(
                single_task_metrics={
                    "PN_classification": {
                        "Accuracy": [0.55, 0.45, 0.85],
                        "AUC": [0.55, 0.65, 0.7]
                    },
                    "Prostate_segmentation": {
                        "DICEMetric": [0.65, 0.71, 0.8]
                    },
                },
                multi_task_losses={
                    "LearningAlgorithm_0": {
                        "mean_loss_without_regularization": [0.95, 0.75, 0.65],
                        "mean_loss_with_regularization": [1.15, 0.85, 0.75]
                    },
                    "LearningAlgorithm_1": {
                        "median_loss_without_regularization": [1.15, 0.85, 0.75],
                        "median_loss_with_regularization": [2.05, 1.55, 1.15]
                    },
                },
                single_task_losses={
                    "PN_classification": {
                        "BinaryBalancedAccuracy": [0.65, 0.55, 0.95]
                    },
                    "Prostate_segmentation": {
                        "DICELoss": [0.75, 0.81, 0.9]
                    },
                }
            )
        )
    )

    history.plot(show=True)
