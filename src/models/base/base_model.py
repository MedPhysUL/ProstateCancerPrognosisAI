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

from monai.data import DataLoader
import numpy as np
from torch import FloatTensor, Tensor

from src.data.datasets.prostate_cancer_dataset import DataModel, ProstateCancerDataset
from src.utils.hyperparameters import HP
from src.utils.reductions import MetricReduction
from src.utils.score_metrics import Direction
from src.utils.tasks import Task, TaskType


class Output(NamedTuple):
    predictions: List = []
    targets: List = []


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
        self._tasks = None

    @property
    def tasks(self) -> Optional[List[Task]]:
        return self._tasks

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

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
        self._tasks = dataset.tasks

    @abstractmethod
    def predict(
            self,
            x: DataModel.x
    ) -> DataModel.y:
        """
        Returns predictions for all samples in a particular batch. For classification tasks, it returns the probability
        of belonging to class 1. For regression tasks, it returns the predicted real-valued target. For segmentation
        tasks, it returns the predicted segmentation map.

        Parameters
        ----------
        x : DataElement.x
            Batch data items.

        Returns
        -------
        predictions : DataModel.y
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
            predictions: DataModel.y,
            targets: DataModel.y
    ) -> Dict[str, float]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        predictions : DataModel.y
            Batch data items.
        targets : DataElement.y
            Batch data items.

        Returns
        -------
        scores : Dict[str, float]
            Score for each tasks.
        """
        scores = {}
        for task in self.tasks:
            scores[task.name] = task.metric(predictions[task.name], targets[task.name])

        return scores

    def score_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int]
    ) -> Dict[str, float]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.

        Returns
        -------
        scores : Dict[str, float]
            Score for each tasks.
        """
        subset = dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        scores = {}
        segmentation_scores_dict = {
            task.name: [] for task in self.tasks if task.task_type == TaskType.SEGMENTATION
        }
        non_segmentation_outputs_dict = {
            task.name: Output() for task in self.tasks if task.task_type != TaskType.SEGMENTATION
        }

        for x, targets in data_loader:
            predictions = self.predict(x)

            for task in self.tasks:
                pred, target = predictions[task.name], targets[task.name]

                if task.task_type == TaskType.SEGMENTATION:
                    segmentation_scores_dict[task.name].append(task.metric(pred, target, MetricReduction.NONE))
                else:
                    non_segmentation_outputs_dict[task.name].predictions.append(pred)
                    non_segmentation_outputs_dict[task.name].targets.append(target)

        for task in self.tasks:
            if task.task_type == TaskType.SEGMENTATION:
                scores[task.name] = task.metric.perform_reduction(FloatTensor(segmentation_scores_dict[task.name]))
            else:
                output = non_segmentation_outputs_dict[task.name]
                scores[task.name] = task.metric(output.predictions, output.targets)

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
        subset = dataset[dataset.train_mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        thresholds = np.linspace(start=0.01, stop=0.95, num=95)

        classification_tasks = [task for task in self.tasks if task.task_type == TaskType.CLASSIFICATION]
        outputs_dict = {task.name: Output() for task in classification_tasks}

        for x, targets in data_loader:
            predictions = self.predict(x)

            for task in classification_tasks:
                pred, target = predictions[task.name], targets[task.name]

                outputs_dict[task.name].predictions.append(pred)
                outputs_dict[task.name].targets.append(target)

        for task in classification_tasks:
            output = outputs_dict[task.name]
            scores = np.array([task.metric(output.predictions, output.targets, t) for t in thresholds])

            # We set the threshold to the optimal threshold
            if task.metric.direction == Direction.MINIMIZE.value:
                task.metric.threshold = thresholds[np.argmin(scores)]
            else:
                task.metric.threshold = thresholds[np.argmax(scores)]
