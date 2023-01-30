"""
    @file:              base_model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 01/2023

    @Description:       This file contains an abstract model named BaseModel. All other models need to inherit from
                        this model to ensure consistency will all hyperparameters tuning functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.data.datasets.prostate_cancer_dataset import FeaturesType, ProstateCancerDataset, TargetsType
from src.utils.hyperparameters import HP


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
    def losses(
            self,
            predictions: TargetsType,
            targets: TargetsType
    ) -> Dict[str, float]:
        """
        Returns the losses for all samples in a particular batch.

        Parameters
        ----------
        predictions : TargetsType
            Batch data items.
        targets : TargetsType
            Batch data items.

        Returns
        -------
        losses : Dict[str, float]
            Loss for each tasks.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            x: FeaturesType
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular batch. For classification tasks, it returns the probability
        of belonging to class 1. For regression tasks, it returns the predicted real-valued target. For segmentation
        tasks, it returns the predicted segmentation map.

        Parameters
        ----------
        x : FeaturesType
            Batch data items.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int],
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using a mask parameter.
        For classification tasks, it returns the probability of belonging to class 1. For regression tasks, it returns
        the predicted real-valued target.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions (except segmentation map).
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
    def scores(
            self,
            predictions: TargetsType,
            targets: TargetsType
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        predictions : TargetsType
            Batch data items.
        targets : TargetsType
            Batch data items.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def scores_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int]
    ) -> Dict[str, Dict[str, float]]:
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
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
