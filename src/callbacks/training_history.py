"""
    @file:              training_history.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       This file is used to define the 'TrainingHistory' class which is used to store losses and score
                        metrics values obtained during the training process.
"""

from itertools import count
import os
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.callbacks.callback import Callback, Priority


MeasurementHistoryType = Dict[str, Dict[str, List[float]]]


class MeasurementsHistoryDict(TypedDict):
    """
    A TypeDict defining the data structure that contains the various measurements performed during a training process.

    Elements
    --------
    multi_task_losses: MeasurementHistoryType
        A dictionary containing multi-task losses. The keys are the names of the learning algorithm used, the values
        are dictionaries whose keys are the names of the losses while its values are a list of the loss measured each
        epoch.

        Example (2 learning algorithms, 3 epochs) :
            multi_task_losses = {
                "LearningAlgorithm(0)": {
                    "MeanLoss('regularization'=False)": [0.9, 0.7, 0.6],
                    "MeanLoss('regularization'=True)": [1.1, 0.8, 0.7]
                },
                "LearningAlgorithm(1)": {
                    "MedianLoss('regularization'=False)": [1.1, 0.8, 0.7],
                    "MedianLoss('regularization'=True)": [2, 1.5, 1.1]
                },
            }

    single_task_losses: MeasurementHistoryType
        A dictionary containing single task losses. The keys are the names of the tasks, the values are dictionaries
        whose keys are the names of the losses while its values are a list of the loss measured each epoch.

        Example (2 tasks, 3 epochs) :
            single_task_losses = {
                "ClassificationTask('target_column'='PN')": {
                    "BinaryBalancedAccuracy('reduction'='mean')": [0.6, 0.5, 0.9]
                },
                "SegmentationTask('modality'='CT', 'organ'='Prostate')": {
                    "DICELoss('reduction'='mean')": [0.7, 0.76, 0.85]
                },
            }

    single_task_metrics : MeasurementHistoryType
        A dictionary containing metric values. The keys are the names of the tasks, the values are dictionaries whose
        keys are the names of the metrics while its values are a list of the metric measured each epoch.

        Example (2 tasks, 3 epochs) :

            metrics = {
                "ClassificationTask('target_column'='PN')": {
                    "BinaryBalancedAccuracy('reduction'='mean')": [0.6, 0.5, 0.9],
                    "AUC('reduction'='none')": [0.6, 0.7, 0.75]
                },
                "SegmentationTask('modality'='CT', 'organ'='Prostate')": {
                    "DICEMetric('reduction'='mean')": [0.7, 0.76, 0.85]
                },
            }
    """
    multi_task_losses: MeasurementHistoryType
    single_task_losses: MeasurementHistoryType
    single_task_metrics: MeasurementHistoryType


class HistoryDict(TypedDict):
    """
    A TypeDict defining the data structure that contains the different sets measurements.

    Elements
    --------
    train : _Measures
        Training set measures.
    valid : _Measures
        Validation set measures.
    """
    train: MeasurementsHistoryDict
    valid: MeasurementsHistoryDict


class TrainingHistory(Callback):
    """
    This class is used to store losses and score metrics values obtained during the training process.
    """

    instance_counter = count()

    TRAIN: Literal["train"] = "train"
    VALID: Literal["valid"] = "valid"

    MULTI_TASK_LOSSES: Literal["multi_task_losses"] = "multi_task_losses"
    SINGLE_TASK_LOSSES: Literal["multi_task_losses"] = "single_task_losses"
    SINGLE_TASK_METRICS: Literal["multi_task_losses"] = "single_task_metrics"

    def __init__(
            self,
            container: Optional[HistoryDict] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the history container.

        Parameters
        ----------
        container : Optional[History]
            History container.
        name : Optional[str]
            The name of the callback.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name is not None else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, **kwargs)

        self.container = container if container else self._get_empty_container()

    def __getitem__(self, item: Union[str, int, slice]) -> dict:
        if isinstance(item, str):
            return self.container[item]  # type: ignore
        elif isinstance(item, (int, slice)):
            return self._get_state(self.container, item)

    def __contains__(self, item):
        return item in self.container

    def __iter__(self):
        return iter(self.container)

    def __len__(self):
        return len(self.container)

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
    def training_set_history(self) -> MeasurementsHistoryDict:
        """
        Training set losses and score metrics history.

        Returns
        -------
        history : MeasuresHistoryTypedDict
            Training set history.
        """
        return self.container[self.TRAIN]

    @property
    def validation_set_history(self) -> MeasurementsHistoryDict:
        """
        Validation set losses and score metrics history.

        Returns
        -------
        history : MeasuresHistoryTypedDict
            Validation set history.
        """
        return self.container[self.VALID]

    @staticmethod
    def _get_empty_container() -> HistoryDict:
        """
        Gets empty history container.

        Returns
        -------
        container : HistoryDict
            History container
        """
        empty_measure = MeasurementsHistoryDict(multi_task_losses={}, single_task_metrics={}, single_task_losses={})
        return HistoryDict(train=empty_measure, valid=empty_measure)

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
            measure_category: Literal["single_task_metrics", "multi_task_losses", "single_task_losses"],
            task_key: str,
            **kwargs
    ) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]:
        """
        Create a plot of the given measure category ('losses' or 'metrics') using the training and validation set
        history.

        Parameters
        ----------
        measure_category : Literal["single_task_metrics", "multi_task_losses", "single_task_losses"]
            Measure category, i.e 'multi_task_losses', 'single_task_losses' or 'metrics'.
        task_key : str
            Specific task key in the measure category history.
        kwargs : dict
            Keywords arguments controlling figure aesthetics.

        Returns
        -------
        fig, axes, lines : Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, List[plt.Line2D]]]
            The figure, axes and lines of the plot.
        """
        train_measure_history = self.training_set_history[measure_category]
        valid_measure_history = self.validation_set_history[measure_category]

        if task_key in train_measure_history:
            train_task_history = train_measure_history[task_key]
            train_tasks = list(train_task_history.keys())
        else:
            train_task_history, train_tasks = [], []

        if task_key in valid_measure_history:
            valid_task_history = valid_measure_history[task_key]
            valid_tasks = list(valid_task_history.keys())
        else:
            valid_task_history, valid_tasks = [], []

        all_tasks = list(set(train_tasks + valid_tasks))

        axes_dict, lines = {}, {k: [] for k in all_tasks}
        fig, axes = plt.subplots(nrows=len(all_tasks), ncols=1, figsize=kwargs.get("figsize", (16, 12)), sharex='all')
        axes = np.ravel(axes)
        for i, (ax, key) in enumerate(zip(axes, all_tasks)):
            if key in train_tasks:
                lines[key].append(ax.plot(train_task_history[key], label=self.TRAIN, linewidth=kwargs.get("lw", 3))[0])
            if key in valid_tasks:
                lines[key].append(ax.plot(valid_task_history[key], label=self.VALID, linewidth=kwargs.get("lw", 3))[0])

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

        for measure_category in MeasurementsHistoryDict.__annotations__.keys():
            training_set_task_keys = list(self.training_set_history[measure_category].keys())  # type: ignore
            validation_set_task_keys = list(self.validation_set_history[measure_category].keys())  # type: ignore
            all_task_keys = list(set(training_set_task_keys + validation_set_task_keys))

            if path_to_save is not None:
                path_to_measure_category_directory = os.path.join(path_to_save, measure_category)
                os.mkdir(path_to_measure_category_directory)

            for task_key in all_task_keys:
                fig, axes, lines = self._create_plot(measure_category, task_key, **kwargs)  # type: ignore
                plt.tight_layout(rect=(0, 0.03, 1, 0.95))
                if path_to_save is not None:
                    path_to_fig = os.path.join(os.path.join(path_to_save, measure_category), f"{task_key}.pdf")
                    fig.savefig(path_to_fig, dpi=kwargs.get("dpi", 300))
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
        self._append_state(self.container, trainer.epoch_state.as_dict())


if __name__ == "__main__":
    history = TrainingHistory(
        container={
            "train": {
                "single_task_metrics": {
                    "PN_classification": {
                        "Accuracy": [0.6, 0.5, 0.9],
                        "AUC": [0.6, 0.7, 0.75]
                    },
                    "Prostate_segmentation": {
                        "DICEMetric": [0.7, 0.76, 0.85]
                    },
                },
                "multi_task_losses": {
                    "LearningAlgorithm_0": {
                        "mean_loss_without_regularization": [0.9, 0.7, 0.6],
                        "mean_loss_with_regularization": [1.1, 0.8, 0.7]
                    },
                    "LearningAlgorithm_1": {
                        "median_loss_without_regularization": [1.1, 0.8, 0.7],
                        "median_loss_with_regularization": [2, 1.5, 1.1]
                    },
                },
                "single_task_losses": {
                    "PN_classification": {
                        "BinaryBalancedAccuracy": [0.6, 0.5, 0.9]
                    },
                    "Prostate_segmentation": {
                        "DICELoss": [0.7, 0.76, 0.85]
                    },
                }
            },
            "valid": {
                "single_task_metrics": {
                    "PN_classification": {
                        "Accuracy": [0.55, 0.45, 0.85],
                        "AUC": [0.55, 0.65, 0.7]
                    },
                    "Prostate_segmentation": {
                        "DICEMetric": [0.65, 0.71, 0.8]
                    },
                },
                "multi_task_losses": {
                    "LearningAlgorithm_0": {
                        "mean_loss_without_regularization": [0.95, 0.75, 0.65],
                        "mean_loss_with_regularization": [1.15, 0.85, 0.75]
                    },
                    "LearningAlgorithm_1": {
                        "median_loss_without_regularization": [1.15, 0.85, 0.75],
                        "median_loss_with_regularization": [2.05, 1.55, 1.15]
                    },
                },
                "single_task_losses": {
                    "PN_classification": {
                        "BinaryBalancedAccuracy": [0.65, 0.55, 0.95]
                    },
                    "Prostate_segmentation": {
                        "DICELoss": [0.75, 0.81, 0.9]
                    },
                }
            },
        }
    )

    history.plot(show=True)
