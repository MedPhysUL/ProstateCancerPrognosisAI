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
from typing import Dict, Optional, Sequence, Union

from torch import cat, stack, Tensor
from torch import device as torch_device
from torch.nn import Module, ModuleDict

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ....tasks.base import TableTask


class InputMode(StrEnum):
    """
    The input mode of the model. Available modes are 'tabular', 'radiomics' and 'hybrid'. Defaults to 'tabular'.

    Elements
    --------
    HYBRID : str
        The model will take as input both the tabular and radiomics features.
    RADIOMICS : str
        The model will take as input only the radiomics features.
    TABULAR : str
        The model will take as input only the tabular features.
    """
    HYBRID = auto()
    RADIOMICS = auto()
    TABULAR = auto()


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
    predictions. The features can be either tabular or radiomics or both.
    """

    def __init__(
            self,
            features_columns: Optional[Union[str, Sequence[str]]] = None,
            input_mode: Union[str, InputMode] = InputMode.TABULAR,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            n_radiomics: int = 5,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        features_columns : Optional[Union[str, Sequence[str]]]
            The names of the features columns. This parameter is only used when `input_mode` is set to 'tabular' or
            'hybrid'.
        input_mode : Union[str, InputMode]
            The input mode of the model. Available modes are 'tabular', 'radiomics' and 'hybrid'. Defaults to 'tabular'.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'separated' or 'fully_shared'. If 'separated', a separate extractor model is used for
            each task. If 'fully_shared', a fully shared extractor model is used. All layers are shared between the
            tasks.
        n_radiomics : int
            Number of radiomics features. Defaults to 5. This parameter is only used when `input_mode` is set to
            'radiomics' or 'hybrid'.
        device : Optional[torch_device]
            The device of the model. Defaults to None.
        name : Optional[str]
            The name of the model. Defaults to None.
        seed : Optional[int]
            Random state used for reproducibility. Defaults to None.
        """
        super().__init__(device=device, name=name, seed=seed)

        if input_mode == InputMode.RADIOMICS or input_mode == InputMode.HYBRID:
            assert n_radiomics > 0, (
                "n_radiomics must be greater than 0 when input_mode is set to 'radiomics' or 'hybrid'"
            )

        if features_columns is None or isinstance(features_columns, Sequence):
            self.features_columns = features_columns
        else:
            self.features_columns = [features_columns]

        self.input_mode = InputMode(input_mode)
        self.multi_task_mode = MultiTaskMode(multi_task_mode)
        self.n_radiomics = n_radiomics

        self.predictor = None

    @abstractmethod
    def _build_predictor(self, dataset) -> Union[Module, ModuleDict]:
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

        self.predictor = self._build_predictor(dataset)

        return self

    @staticmethod
    def _get_input_in_separated_mode(
            task: TableTask,
            table_data: Optional[Tensor] = None,
            radiomics: Optional[Union[Tensor, Dict[str, Tensor]]] = None
    ) -> Tensor:
        """
        Returns the input of the predictor in separated mode.

        Parameters
        ----------
        task : TableTask
            The table task.
        table_data : Optional[Tensor]
            The table data.
        radiomics : Optional[Union[Tensor, Dict[str, Tensor]]]
            The radiomics data.

        Returns
        -------
        input : Tensor
            The input of the predictor.
        """
        if table_data is not None and radiomics is not None:
            if isinstance(radiomics, Tensor):
                return cat([table_data, radiomics], dim=1)
            else:
                return cat([table_data, radiomics[task.name]], dim=1)
        elif table_data is not None:
            return table_data
        elif radiomics is not None:
            if isinstance(radiomics, Tensor):
                return radiomics
            else:
                return radiomics[task.name]
        else:
            raise ValueError("table_data and radiomics cannot be None at the same time")

    @staticmethod
    def _get_input_in_fully_shared_mode(
            table_data: Optional[Tensor] = None,
            radiomics: Optional[Union[Tensor, Dict[str, Tensor]]] = None
    ) -> Tensor:
        """
        Returns the input of the predictor in fully shared mode.

        Parameters
        ----------
        table_data : Optional[Tensor]
            The table data.
        radiomics : Optional[Union[Tensor, Dict[str, Tensor]]]
            The radiomics data.

        Returns
        -------
        input : Tensor
            The input of the predictor.
        """
        if radiomics is not None:
            radiomics = radiomics if isinstance(radiomics, Tensor) else stack(list(radiomics.values()), dim=1)

        if table_data is not None and radiomics is not None:
            return cat([table_data, radiomics], dim=1)
        elif table_data is not None:
            return table_data
        elif radiomics is not None:
            return radiomics
        else:
            raise ValueError("table_data and radiomics cannot be None at the same time")

    def _get_prediction(
            self,
            table_data: Optional[Tensor],
            radiomics: Optional[Union[Tensor, Dict[str, Tensor]]]
    ) -> Dict[str, Tensor]:
        """
        Returns the prediction.

        Parameters
        ----------
        table_data : Optional[Tensor]
            The table data.
        radiomics : Optional[Union[Tensor, Dict[str, Tensor]]]
            The radiomics data.

        Returns
        -------
        prediction : Dict[str, Tensor]
            The prediction.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            prediction = {}
            for task in self._tasks.table_tasks:
                input_tensor = self._get_input_in_separated_mode(task, table_data, radiomics)
                prediction[task.name] = self.predictor[task.name](input_tensor)[:, 0]
            return prediction
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            input_tensor = self._get_input_in_fully_shared_mode(table_data, radiomics)
            y = self.predictor(input_tensor)
            return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    @check_if_built
    def forward(
            self,
            features: Optional[FeaturesType] = None,
            radiomics: Optional[Union[Tensor, Dict[str, Tensor]]] = None
    ) -> TargetsType:
        """
        Executes the forward pass.

        Parameters
        ----------
        features : Optional[FeaturesType]
            Batch data items. Defaults to None.
        radiomics : Optional[Tensor]
            Radiomics features. Defaults to None.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        if self.input_mode == InputMode.RADIOMICS:
            return self._get_prediction(None, radiomics)
        else:
            x_table = stack([features.table[k] for k in self.features_columns], 1).float()
            if self.input_mode == InputMode.TABULAR:
                pred = self._get_prediction(x_table, None)
                return pred
            elif self.input_mode == InputMode.HYBRID:
                return self._get_prediction(x_table, radiomics)
            else:
                raise ValueError(f"Invalid input_mode: {self.input_mode}")
