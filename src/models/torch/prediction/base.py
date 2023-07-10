"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define an abstract 'Predictor' model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import auto, StrEnum
from typing import Dict, Mapping, Optional, Sequence, Union
from warnings import warn

from torch import stack, Tensor
from torch import device as torch_device
from torch.nn import Module, ModuleDict

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType


class MultiTaskMode(StrEnum):
    """
    This class is used to define the multi-task mode of the model. It can be either 'separated' or 'fully_shared'.

    Elements
    --------
    SEPARATED : str
        A separate extractor model is used for each task.
    FULLY_SHARED : str
        A fully shared extractor model is used. All layers are shared between the tasks.
    """
    SEPARATED = auto()
    FULLY_SHARED = auto()


class Predictor(TorchModel, ABC):
    """
    Abstract class for all predictors. A predictor is a model that takes as input a set of features and outputs a set of
    predictions.
    """

    def __init__(
            self,
            features_columns: Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
            bayesian: bool = False
    ):
        """
        Initializes the model.

        Parameters
        ----------
        features_columns : Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]]
            The names of the features columns. If a mapping is provided, the keys must be the target columns associated
            to the task.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'separated' or 'fully_shared'. If 'separated', a separate extractor model is used for
            each task. If 'fully_shared', a fully shared extractor model is used. All layers are shared between the
            tasks.
        device : Optional[torch_device]
            The device of the model. Defaults to None.
        name : Optional[str]
            The name of the model. Defaults to None.
        seed : Optional[int]
            Random state used for reproducibility. Defaults to None.
        bayesian : bool
            Whether the model implements variational inference.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.features_columns = [features_columns] if isinstance(features_columns, str) else features_columns
        self.multi_task_mode = MultiTaskMode(multi_task_mode)

        if isinstance(features_columns, Mapping):
            if self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
                warn(
                    "The multi-task mode is set to 'fully_shared' but a mapping of features columns is provided. "
                    "The multi-task mode is automatically set to 'separated'."
                )
            self.multi_task_mode = MultiTaskMode.SEPARATED

        self.bayesian = bayesian

        self.map_from_target_col_to_task_name = None        # TODO
        self.predictor = None

        self.kl_divergence: Optional[Dict[str, Tensor]] = None

    @abstractmethod
    def _build_predictor(self, dataset: ProstateCancerDataset) -> Union[Module, ModuleDict]:
        """
        Returns the predictor module.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used to build the extractor.

        Returns
        -------
        predictor : Union[Module, ModuleDict]
            The predictor module.
        """
        raise NotImplementedError

    def build(self, dataset: ProstateCancerDataset) -> Predictor:
        """
        Builds the model. This method must be called before training the model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : Predictor
            The current model.
        """
        super().build(dataset=dataset)

        if self.features_columns is None:
            self.features_columns = dataset.table_dataset.features_columns
        elif isinstance(self.features_columns, Mapping):
            assert set(self.features_columns.keys()) == set([t.target_column for t in dataset.tasks.table_tasks]), (
                "The features columns mapping must contain the same tasks as the dataset."
            )
            self.map_from_target_col_to_task_name = {t.target_column: t.name for t in dataset.tasks.table_tasks}

        self.predictor = self._build_predictor(dataset)

        return self

    @abstractmethod
    def _get_prediction(self, table_data: Union[Tensor, Dict[str, Tensor]]) -> Union[Dict[str, Tensor], tuple]:
        """
        Returns the prediction.

        Parameters
        ----------
        table_data : Union[Tensor, Dict[str, Tensor]]
            The table data.

        Returns
        -------
        prediction : Union[Dict[str, Tensor], tuple]
            The prediction and its KL divergence (if the model is in bayesian mode).
        """
        raise NotImplementedError

    def _bayesian_forward(self, features: FeaturesType) -> TargetsType:
        """
        Executes a bayesian forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items. Defaults to None.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        if isinstance(self.features_columns, Mapping):
            x_table = {}
            for target_col, feature_cols in self.features_columns.items():
                task_name = self.map_from_target_col_to_task_name[target_col]
                x_table[task_name] = stack([features.table[f] for f in feature_cols], 1).float()
        else:
            x_table = stack([features.table[f] for f in self.features_columns], 1).float()

        prediction, kl_divergence = self._get_prediction(x_table)

        self.kl_divergence = kl_divergence

        return prediction

    def _deterministic_forward(self, features: FeaturesType) -> TargetsType:
        """
        Executes a deterministic forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items. Defaults to None.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        if isinstance(self.features_columns, Mapping):
            x_table = {}
            for target_col, feature_cols in self.features_columns.items():
                task_name = self.map_from_target_col_to_task_name[target_col]
                x_table[task_name] = stack([features.table[f] for f in feature_cols], 1).float()
        else:
            x_table = stack([features.table[f] for f in self.features_columns], 1).float()

        return self._get_prediction(x_table)

    @check_if_built
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Executes the forward pass. Implements different methods depending on whether the model is in bayesian mode.

        Parameters
        ----------
        features : FeaturesType
            Batch data items. Defaults to None.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        if self.bayesian:
            return self._bayesian_forward(features=features)
        else:
            return self._deterministic_forward(features=features)
