"""
    @file:              base_models.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains two abstract models named BinaryClassifier and Regressor. All other models
                        need to inherit from one of these two models to ensure consistency will all hyperparameters
                        tuning functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import tensor

from src.data.datasets.single_task_table_dataset import SingleTaskTableDataset
from src.utils.hyperparameters import HP
from src.utils.score_metrics import BinaryClassificationMetric, Direction


class BinaryClassifier(ABC):
    """
    An abstract class which is used to define a model performing binary classification.
    """

    def __init__(
            self,
            classification_threshold: float = 0.5,
            weight: Optional[float] = None,
            train_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sets the protected attributes of the object.

        Parameters
        ----------
        classification_threshold : float
            The threshold used to classify a sample in class 1.
        weight : Optional[float]
            The weight attributed to class 1 (in [0, 1]).
        train_params : Optional[Dict[str, Any]]
            Keyword arguments that are proper to the child model inheriting from this class and that will be used when
            there is a call to the fit method.
        """
        if weight is not None:
            if not (0 < weight < 1):
                raise ValueError("The weight parameter must be included in range [0, 1]")

        self._thresh = classification_threshold
        self._train_params = train_params if train_params is not None else {}
        self._weight = weight

    @property
    def thresh(self) -> float:
        return self._thresh

    @property
    def train_params(self) -> Dict[str, Any]:
        return self._train_params

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    def _get_scaling_factor(
            self,
            y_train: Union[tensor, np.array]
    ) -> Optional[float]:
        """
        Computes the scaling factor that needs to be apply to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[tensor, np.array]
            (N, 1) tensor or array containing labels.

        Returns
        -------
        scaling_factor : Optional[float]
            Positive scaling factor.
        """
        # If no weight was provided we return None
        if self.weight is None:
            return None

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        return (n0/n1)*(self._weight/(1-self._weight))

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

    def find_optimal_threshold(
            self,
            dataset: SingleTaskTableDataset,
            metric: BinaryClassificationMetric
    ) -> None:
        """
        Finds the optimal classification threshold for a binary classification task according to a given metric.

        Parameters
        ----------
        dataset : SingleTaskTableDataset
            Single task table dataset. Its items are tuples (x, y, idx) where
                - x : (N,D) tensor or array with D-dimensional samples
                - y : (N,) tensor or array with classification labels
                - idx : (N,) tensor or array with idx of samples according to the whole dataset
        metric : BinaryClassificationMetric
            Binary classification metric used to find optimal threshold
        """
        # We predict proba on the training set
        proba = self.predict_proba(dataset, dataset.train_mask)

        # For multiple threshold values we calculate the metric
        thresholds = np.linspace(start=0.01, stop=0.95, num=95)
        scores = np.array([metric(proba, dataset.y[dataset.train_mask], t) for t in thresholds])

        # We save the optimal threshold
        if metric.direction == Direction.MINIMIZE:
            self._thresh = thresholds[np.argmin(scores)]
        else:
            self._thresh = thresholds[np.argmax(scores)]

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
            dataset: SingleTaskTableDataset
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        dataset : SingleTaskTableDataset
            Single task table dataset. Its items are tuples (x, y, idx) where
                - x : (N,D) tensor or array with D-dimensional samples
                - y : (N,) tensor or array with classification labels
                - idx : (N,) tensor or array with idx of samples according to the whole dataset
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
            self,
            dataset: SingleTaskTableDataset,
            mask: Optional[List[int]] = None
    ) -> Union[tensor, np.array]:
        """
        Returns the probabilities of being in class 1 for all samples in a particular set (default = test).

        Parameters
        ----------
        dataset : SingleTaskTableDataset
            Single task table dataset. Its items are tuples (x, y, idx) where
                - x : (N,D) tensor or array with D-dimensional samples
                - y : (N,) tensor or array with classification labels
                - idx : (N,) tensor or array with idx of samples according to the whole dataset
        mask : Optional[List[int]]
            List of dataset idx for which we want to predict probabilities

        Returns
        -------
            (N,) tensor or array
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


class Regressor(ABC):
    """
    An abstract class which is used to define a model performing a regression task.
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
            dataset: SingleTaskTableDataset
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        dataset : SingleTaskTableDataset
            Single task table dataset. Its items are tuples (x, y, idx) where
                - x : (N,D) tensor or array with D-dimensional samples
                - y : (N,) tensor or array with classification labels
                - idx : (N,) tensor or array with idx of samples according to the whole dataset
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            dataset: SingleTaskTableDataset,
            mask: Optional[List[int]] = None
    ) -> Union[tensor, np.array]:
        """
        Returns the predicted real-valued targets for all samples in a particular set (default = test).

        Parameters
        ----------
        dataset : SingleTaskTableDataset
            Single task table dataset. Its items are tuples (x, y, idx) where
                - x : (N,D) tensor or array with D-dimensional samples
                - y : (N,) tensor or array with classification labels
                - idx : (N,) tensor or array with idx of samples according to the whole dataset
        mask : Optional[List[int]]
            List of dataset idx for which we want to predict probabilities

        Returns
        -------
            (N,) tensor or array
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
