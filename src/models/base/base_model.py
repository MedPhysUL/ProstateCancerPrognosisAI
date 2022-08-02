"""
    @file:              base_model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains an abstract model named BaseModel. All other models need to inherit from
                        this model to ensure consistency will all hyperparameters tuning functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import tensor

from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.utils.hyperparameters import HP
from src.utils.score_metrics import BinaryClassificationMetric, Direction


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
    ) -> Union[tensor, np.array]:
        """
        Returns predictions for all samples in a particular set (default = test). For classification tasks, it returns
        the probability of belonging to class 1. For regression tasks, it returns the predicted real-valued target.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : Optional[List[int]]
            List of dataset idx for which we want to predict target/probabilities.

        Returns
        -------
        predictions : Union[tensor, np.array]
            (T, N,) tensor or array where T is the number of tasks and N is the number of samples.
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
            y_train: Union[tensor, np.array]
    ) -> None:
        """
        Updates the scaling factor that needs to be apply to samples in class 1.

        Parameters
        ----------
        y_train : Union[tensor, np.array]
            (N, 1) tensor or array containing labels.
        """
        raise NotImplementedError

    def fix_thresholds_to_optimal_values(
            self,
            dataset: ProstateCancerDataset,
            metric: BinaryClassificationMetric
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        metric : BinaryClassificationMetric
            Binary classification metric used to find optimal threshold
        """
        # We predict targets (or proba) on the training set
        predictions = self.predict(dataset, dataset.train_mask)

        for single_task_table_dataset, pred in zip(dataset.table_dataset.datasets, predictions):
            if single_task_table_dataset.classification:

                # For multiple threshold values we calculate the metric
                thresholds = np.linspace(start=0.01, stop=0.95, num=95)
                scores = np.array(
                    [
                        metric(pred, single_task_table_dataset.y[single_task_table_dataset.train_mask], t)
                        for t in thresholds
                    ]
                )

                # We save the optimal threshold
                if metric.direction == Direction.MINIMIZE:
                    single_task_table_dataset.thresh = thresholds[np.argmin(scores)]
                else:
                    single_task_table_dataset.thresh = thresholds[np.argmax(scores)]
