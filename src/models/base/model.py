"""
    @file:              model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 04/2023

    @Description:       This file contains an abstract model named 'Model'. All other models need to inherit from
                        this model to ensure consistency will all training and tuning tools.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from monai.utils import set_determinism
from sksurv.linear_model.coxph import BreslowEstimator
from torch.nn import Module
from torch import cuda
from torch import device as torch_device

from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ...tasks import TaskList
from ...tools.transforms import to_numpy


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
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Sets the model's device and name.

        Parameters
        ----------
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__()

        self.device = device if device else torch_device("cuda") if cuda.is_available() else torch_device("cpu")
        self.name = name if name else self.__class__.__name__
        self.seed = seed

        self._breslow_estimators = {}
        self._is_built: bool = False
        self._tasks: Optional[TaskList] = None

    @property
    def breslow_estimators(self) -> Dict[str, BreslowEstimator]:
        """
        Breslow estimators for each survival analysis tasks.

        Returns
        -------
        breslow_estimators : Dict[str, BreslowEstimator]
            Breslow estimator functions.
        """
        return self._breslow_estimators

    def build(
            self,
            dataset: ProstateCancerDataset
    ) -> Model:
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : Model
            The current model.
        """
        self._set_seed()
        self._tasks = dataset.tasks
        self._is_built = True

        return self

    @check_if_built
    def fit_breslow_estimators(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fit all survival analysis tasks' breslow estimators given the training dataset. It is recommended to train the
        model before calling this method.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        if dataset.table_dataset:
            predictions = self.predict_on_dataset(dataset, dataset.train_mask)
            targets = dataset.table_dataset[dataset.train_mask].y

            for task in dataset.tasks.survival_analysis_tasks:
                pred, target = to_numpy(predictions[task.name]), to_numpy(targets[task.name])

                nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(target)
                if len(nonmissing_targets_idx) > 0:
                    pred, target = pred[nonmissing_targets_idx], target[nonmissing_targets_idx]
                    task.breslow_estimator.fit(pred[:, 0], target[:, 0], target[:, 1])
                    self._breslow_estimators[task.name] = task.breslow_estimator

    @check_if_built
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
            - For survival analysis tasks, returns the estimated risk of experiencing an event.
            - For segmentation tasks, returns the predicted segmentation map.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression, survival and segmentation tasks predictions.

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
            dataset: ProstateCancerDataset,
            mask: List[int],
            probability: bool = True
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using the 'mask'
        parameter, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.
            - For survival analysis tasks, returns the estimated risk of experiencing an event.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the predictions.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression, survival and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions (except segmentation map).
        """
        raise NotImplementedError

    @check_if_built
    @abstractmethod
    def compute_score(
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
    def compute_score_on_dataset(
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

    def _set_seed(self):
        """
        Sets numpy and torch seed.
        """
        if self.seed is not None:
            set_determinism(self.seed)
