"""
    @file:              torch_wrapper.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     04/2022
    @Last modification: 09/2022

    @Description:       This file is used to define an abstract class used as wrapper for custom torch models.
"""

import os
from typing import Any, Dict, List, Optional

from torch import save

from src.data.datasets.prostate_cancer_dataset import DataModel, ProstateCancerDataset
from src.models.base.base_model import BaseModel
from src.models.base.custom_torch_base import TorchCustomModel
from src.utils.hyperparameters import HP


class TorchWrapper(BaseModel):
    """
    Class used as a wrapper for model inheriting from TorchCustomModel
    """

    def __init__(
            self,
            model_constructor: TorchCustomModel,
            model_params: Dict[str, Any],
            train_params: Optional[Dict[str, Any]] = None
    ):
        """
        Sets the model protected attribute and other protected attributes via parent's constructor.

        Parameters
        ----------
        model_constructor : TorchCustomModel
            Function used to create the classification model inheriting from TorchCustomModel.
        model_params : Dict[str, Any]
            Parameters used to initialize the classification model inheriting from TorchCustomModel.
        train_params : Optional[Dict[str, Any]]
            Training parameters proper to model for fit function.
        """
        # Initialization of model
        self._model_constructor = model_constructor
        self._model_params = model_params
        self._model = None

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
        # We set the tasks
        self._tasks = dataset.tasks

        # Call the fit method
        self._model.fit(dataset, **self.train_params)

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
        return self._model.predict(x)

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
