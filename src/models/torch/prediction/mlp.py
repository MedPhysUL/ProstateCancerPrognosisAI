"""
    @file:              mlp.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     04/2023
    @Last modification: 07/2023

    @Description:       This file is used to define an 'MLP' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from torch import device as torch_device
from torch import Tensor
from torch.nn import DataParallel, Module, ModuleDict

from .base import MultiTaskMode, Predictor
from ..blocks import BayesianFullyConnectedNet, FullyConnectedNet
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class MLP(Predictor):
    """
    A simple MLP model. The model can also be used to perform table tasks.
    """

    def __init__(
            self,
            features_columns: Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            hidden_channels: Union[str, Sequence[int]] = (25, 25, 25),
            activation: Union[Tuple, str] = "PRELU",
            dropout: Union[Tuple, str, float] = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
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
        hidden_channels : Union[str, Sequence[int]]
            List with number of units in each hidden layer. Defaults to (25, 25, 25).
        activation : Union[Tuple, str]
            Activation function. Defaults to "PRELU".
        dropout : Union[Tuple, str, float]
            Probability of dropout. Defaults to 0.0.
        bias : bool
            If `bias` is True then linear units have a bias term. Defaults to True.
        adn_ordering : str
            A string representing the ordering of activation, dropout, and normalization. Defaults to "NDA".
        device : Optional[torch_device]
            The device of the model. Defaults to None.
        name : Optional[str]
            The name of the model. Defaults to None.
        seed : Optional[int]
            Random state used for reproducibility. Defaults to None.
        bayesian : bool
            Whether the model should implement variational inference.
        """
        super().__init__(
            features_columns=features_columns,
            multi_task_mode=multi_task_mode,
            device=device,
            name=name,
            seed=seed,
            bayesian=bayesian
        )

        self.hidden_channels = literal_eval(hidden_channels) if isinstance(hidden_channels, str) else hidden_channels
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.predictor = None

    def _build_single_predictor(
            self,
            in_channels: int,
            out_channels: int
    ) -> Module:
        """
        Returns a single predictor module.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        predictor : Module
            A single predictor module.
        """
        if self.bayesian:
            fcn = BayesianFullyConnectedNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=self.hidden_channels,
                dropout=self.dropout,
                act=self.activation,
                bias=self.bias,
                adn_ordering=self.adn_ordering
            )
        else:
            fcn = FullyConnectedNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=self.hidden_channels,
                dropout=self.dropout,
                act=self.activation,
                bias=self.bias,
                adn_ordering=self.adn_ordering
            )

        return DataParallel(fcn).to(self.device)

    def _get_in_channels(self, target_column: str) -> int:
        """
        Returns the number of input channels.

        Parameters
        ----------
        target_column : str
            The target column.

        Returns
        -------
        in_channels : int
            The number of input channels.
        """
        if isinstance(self.features_columns, Mapping):
            return len(self.features_columns[target_column])
        else:
            return len(self.features_columns)

    def _build_predictor(self, dataset: ProstateCancerDataset) -> Union[Module, ModuleDict]:
        """
        Builds the model. This method must be called before training the model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : Union[Module, ModuleDict]
            The current model.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            predictor = ModuleDict()
            for task in self._tasks.table_tasks:
                predictor[task.name] = self._build_single_predictor(
                    in_channels=self._get_in_channels(task.target_column),
                    out_channels=1
                )
            return predictor

        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self._build_single_predictor(
                in_channels=len(self.features_columns),
                out_channels=len(self._tasks.table_tasks)
            )

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
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            if self.bayesian:
                prediction = {}
                kl_dict = {}
                for task in self._tasks.table_tasks:
                    if isinstance(table_data, dict):
                        y, kl = self.predictor[task.name](table_data[task.name])
                    else:
                        y, kl = self.predictor[task.name](table_data)
                    prediction[task.name] = y[:, 0]
                    kl_dict[task.name] = kl
                return prediction, kl_dict

            else:
                if isinstance(table_data, dict):
                    return {t.name: self.predictor[t.name](table_data[t.name])[:, 0] for t in self._tasks.table_tasks}
                else:
                    return {t.name: self.predictor[t.name](table_data)[:, 0] for t in self._tasks.table_tasks}

        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            if self.bayesian:
                y, kl = self.predictor(table_data)
                prediction = {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
                kl_dict = {task.name: kl for task in self._tasks.table_tasks}
                return prediction, kl_dict

            else:
                y = self.predictor(table_data)
                return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")
