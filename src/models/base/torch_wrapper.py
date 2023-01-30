"""
    @file:              torch_wrapper.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     04/2022
    @Last modification: 09/2022

    @Description:       This file is used to define an abstract class used as wrapper for custom torch models.
"""

import os
from typing import Any, Callable, Dict, List, Optional

from torch import save

from src.data.datasets.prostate_cancer_dataset import FeaturesType, ProstateCancerDataset, TargetsType
from src.models.base.base_model import BaseModel
from src.models.base.custom_torch_base import TorchCustomModel
from src.utils.hyperparameters import HP


class TorchWrapper(BaseModel):
    """
    Class used as a wrapper for model inheriting from TorchCustomModel
    """

    def __init__(
            self,
            model_constructor: Callable,
            model_params: Dict[str, Any],
            train_params: Optional[Dict[str, Any]] = None
    ):
        """
        Sets the model protected attribute and other protected attributes via parent's constructor.

        Parameters
        ----------
        model_constructor : TorchCustomModel
            Model inheriting from TorchCustomModel.
        model_params : Dict[str, Any]
            Parameters used to initialize the classification model inheriting from TorchCustomModel.
        train_params : Optional[Dict[str, Any]]
            Training parameters proper to model for fit function.
        """
        # Initialization of model
        self._model_constructor = model_constructor
        self._model_params = model_params
        self._model = self._model_constructor(**model_params)

        super().__init__(train_params=train_params)

    @property
    def model(self) -> TorchCustomModel:
        return self._model

    def fit(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        dataset: ProstateCancerDataset
            A dataset.
        """
        # Call the fit method
        self._model.fit(dataset, **self.train_params)

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
        return self._model.losses(predictions, targets)

    def plot_evaluations(
            self,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plots the training and valid curves saved.

        Parameters
        ----------
        save_path : Optional[str]
            Path where the figures will be saved.
        """
        self._model.plot_evaluations(save_path=save_path)

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
        return self._model.predict(x)

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
        return self._model.predict_dataset(dataset, mask)

    def save_model(self, path: str) -> None:
        """
        Saves the model to the given path.

        Parameters
        ----------
        path : str
            Save path.
        """
        save(self._model.state_dict(), os.path.join(path, "torch_model.pt"))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model.

        Returns
        -------
        list_hp : List[HP]
            list of hyperparameters
        """
        raise NotImplementedError

    def scores(
            self,
            predictions: TargetsType,
            targets: TargetsType,
            include_evaluation_metrics: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        predictions : TargetsType
            Batch data items.
        targets : TargetsType
            Batch data items.
        include_evaluation_metrics: bool
            Whether to calculate the scores with the evaluation metrics or not.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        return self._model.scores(predictions, targets, include_evaluation_metrics)

    def scores_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int],
            include_evaluation_metrics: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.
        include_evaluation_metrics: bool
            Whether to calculate the scores with the evaluation metrics or not.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        return self._model.scores_dataset(dataset, mask, include_evaluation_metrics)

    def fix_thresholds_to_optimal_values(
            self,
            dataset: ProstateCancerDataset,
            include_evaluation_metrics: bool = False
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        include_evaluation_metrics: bool
            Whether to fix the thresholds of evaluation metrics or not.
        """
        return self._model.fix_thresholds_to_optimal_values(dataset, include_evaluation_metrics)
