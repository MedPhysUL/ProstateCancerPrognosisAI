"""
    @file:              base_model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains an abstract model named BaseModel. All other models need to inherit from
                        this model to ensure consistency will all hyperparameters tuning functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from torch import Tensor

from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.utils.hyperparameters import HP
from src.utils.score_metrics import Direction
from src.utils.tasks import ClassificationTask, TableTask, Task


class Predictions(NamedTuple):
    """
    Predictions named tuple. This tuple is used to separate image/segmentation related predictions and tabular
    predictions.

    Elements
    --------
    imaging : Union[np.array, Tensor]
        (N, T_imaging, ...) tensor or array where N is the number of samples and T is the number of imaging tasks.
    table : Union[np.array, Tensor]
        (N, T_table) tensor or array where N is the number of samples and T is the number of table tasks.
    """
    imaging: Union[np.array, Tensor]
    table: Union[np.array, Tensor]


class Score(NamedTuple):
    """
    Score named tuple.

    Elements
    --------
    task : Task
        Task.
    score : float
        Score.
    """
    task: Task
    value: float


class BaseModel(ABC):
    """
    An abstract class which is used to define a model.
    """

    def __init__(
            self,
            train_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sets the protected attributes of the object.

        Parameters
        ----------
        train_params : Optional[Dict[str, Any]]
            Keyword arguments that are proper to the child model inheriting from this class and that will be used when
            there is a call to the fit method.
        """
        self._train_params = train_params if train_params is not None else {}

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

    @staticmethod
    def is_encoder() -> bool:
        """
        Whether the class is used to create an Encoder.

        Returns
        -------
        is_encoder : bool
            Whether the class is used to create an Encoder.
        """
        return False

    @staticmethod
    @abstractmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model.

        Returns
        -------
        hps : List[HP]
            Hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None
    ) -> Predictions:
        """
        Returns predictions for all samples in a particular set (default = test). For classification tasks, it returns
        the probability of belonging to class 1. For regression tasks, it returns the predicted real-valued target.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : Optional[List[int]]
            List of dataset idx for which we want to predict target/probabilities (default = test).

        Returns
        -------
        predictions : Predictions
            Predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(
            self,
            path: str
    ) -> None:
        """
        Saves the model to the given path.

        Parameters
        ----------
        path : str
            Save path
        """
        raise NotImplementedError

    @abstractmethod
    def _update_pos_scaling_factor(
            self,
            y_train: Union[Tensor, np.array]
    ) -> None:
        """
        Updates the scaling factor that needs to be apply to samples in class 1.

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N, 1) tensor or array containing labels.
        """
        raise NotImplementedError

    def score(
            self,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None
    ) -> List[Score]:
        """
        Returns scores for all samples in a particular set (default = test) and for all tasks in the dataset.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : Optional[List[int]]
            List of dataset idx for which we want to predict target/probabilities (default = test).

        Returns
        -------
        scores : List[Score]
            List of scores.
        """
        pred = self.predict(dataset=dataset, mask=mask)
        pred_imaging, pred_table = pred.imaging, pred.table

        if mask:
            imaging_data, table_data = dataset[mask].imaging, dataset[mask].table
        else:
            imaging_data, table_data = dataset[dataset.test_mask].imaging, dataset[dataset.test_mask].table

        y_table = table_data[1] if table_data else None
        y_imaging = imaging_data[1] if imaging_data else None

        scores = []
        for task_idx, task in enumerate(dataset.tasks):
            if isinstance(task, TableTask):
                task_targets, task_pred = y_table[:, task_idx], pred_table[:, task_idx]
                nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(task_targets)
                task_targets, task_pred = task_targets[nonmissing_targets_idx], task_pred[nonmissing_targets_idx]
            else:
                task_targets, task_pred = y_imaging[:, task_idx], pred_imaging[:, task_idx]

            scores.append(Score(task=task, value=task.metric(task_pred, task_targets)))

        return scores

    def fix_thresholds_to_optimal_values(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        # We retrieve the table dataset from the prostate cancer dataset
        table_ds = dataset.table_dataset

        # We predict targets (or proba) on the training set
        predictions = self.predict(dataset, dataset.train_mask).table
        targets = table_ds.y[dataset.train_mask]

        for task_idx, task in enumerate(table_ds.tasks):
            if isinstance(task, ClassificationTask):
                # For multiple threshold values we calculate the metric
                thresholds = np.linspace(start=0.01, stop=0.95, num=95)

                # Get targets and predictions for current task
                task_targets, task_pred = targets[:, task_idx], predictions[:, task_idx]
                nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(task_targets)
                task_targets, task_pred = task_targets[nonmissing_targets_idx], task_pred[nonmissing_targets_idx]

                # Calculate scores
                scores = np.array([task.metric(task_pred, task_targets, t) for t in thresholds])

                # We save the optimal threshold
                if task.metric.direction == Direction.MINIMIZE:
                    task.metric.threshold = thresholds[np.argmin(scores)]
                else:
                    task.metric.threshold = thresholds[np.argmax(scores)]
