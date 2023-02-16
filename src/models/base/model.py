"""
    @file:              model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains an abstract model named 'Model'. All other models need to inherit from
                        this model to ensure consistency will all training and tuning tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from torch.nn import Module
from torch import cuda
from torch import device as torch_device

from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ...tuning.hyperparameters import Hyperparameter


def check_if_built(_func):
    def wrapper(*args, **kwargs):
        self = args[0]

        assert self._is_built, (
            f"The model needs to be built using the 'build' method before calling {_func.__name__}."
        )

        return _func(*args, **kwargs)

    return wrapper


class Model(Module, ABC):
    """
    An abstract class which is used to define a model.
    """

    def __init__(
            self,
            device: Optional[torch_device] = None,
            name: Optional[str] = None
    ):
        """
        Sets the model's device and name.

        Parameters
        ----------
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        """
        super().__init__()

        self.device = device if device else torch_device("cuda") if cuda.is_available() else torch_device("cpu")
        self.name = name if name else self.__class__.__name__

        self._dataset: Optional[ProstateCancerDataset] = None
        self._is_built: bool = False

    def build(self, dataset: ProstateCancerDataset) -> None:
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        self._dataset = dataset
        self._is_built = True

    @check_if_built
    @abstractmethod
    def fix_thresholds_to_optimal_values(
            self
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        """
        Executes the forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def predict(
            self,
            features: FeaturesType,
            probability: bool = True
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular batch, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.
            - For segmentation tasks, returns the predicted segmentation map.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def predict_on_dataset(
            self,
            mask: List[int],
            probability: bool = True
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using the 'mask'
        parameter, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        mask : List[int]
            A list of dataset idx for which we want to obtain the predictions.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions (except segmentation map).
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def score(
            self,
            features: FeaturesType,
            targets: TargetsType
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        targets : TargetsType
            Batch data items.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def score_on_dataset(
            self,
            mask: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        raise NotImplementedError
