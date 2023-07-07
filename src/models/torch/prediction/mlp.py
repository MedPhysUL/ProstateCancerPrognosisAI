"""
    @file:              mlp.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define an 'MLP' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from monai.networks.nets import FullyConnectedNet
from torch import device as torch_device
from torch import Tensor
from torch.nn import DataParallel, Module, ModuleDict

from .base import MultiTaskMode, Predictor
from ....data.datasets.prostate_cancer import ProstateCancerDataset
from ....tasks.base import TableTask


class MLP(Predictor):
    """
    A simple MLP model. The model can also be used to perform table tasks.
    """

    def __init__(
            self,
            features_columns: Optional[Union[str, Sequence[str], Mapping[TableTask, Sequence[str]]]] = None,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            hidden_channels: Union[str, Sequence[int]] = (25, 25, 25),
            activation: Union[Tuple, str] = "PRELU",
            dropout: Union[Tuple, str, float] = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        features_columns : Optional[Union[str, Sequence[str], Mapping[TableTask, Sequence[str]]]]
            The names of the features columns.
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
        """
        super().__init__(
            features_columns=features_columns,
            multi_task_mode=multi_task_mode,
            device=device,
            name=name,
            seed=seed
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
        fully_connected_net = FullyConnectedNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            act=self.activation,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        )
        fully_connected_net = DataParallel(fully_connected_net)

        return fully_connected_net.to(self.device)

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
            if isinstance(self.features_columns, Mapping):
                return ModuleDict(
                    {
                        t.name: self._build_single_predictor(
                            in_channels=len(self.features_columns[t.target_column]),
                            out_channels=1
                        )
                        for t in self._tasks.table_tasks
                    }
                )
            else:
                return ModuleDict(
                    {
                        t.name: self._build_single_predictor(
                            in_channels=len(self.features_columns),
                            out_channels=1
                        )
                        for t in self._tasks.table_tasks
                    }
                )
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self._build_single_predictor(
                in_channels=len(self.features_columns),
                out_channels=len(self._tasks.table_tasks)
            )

    def _get_prediction(
            self,
            table_data: Union[Tensor, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
        Returns the prediction.

        Parameters
        ----------
        table_data : Union[Tensor, Dict[str, Tensor]]
            The table data.

        Returns
        -------
        prediction : Dict[str, Tensor]
            The prediction.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            if isinstance(table_data, dict):
                return {t.name: self.predictor[t.name](table_data[t.name])[:, 0] for t in self._tasks.table_tasks}
            else:
                return {t.name: self.predictor[t.name](table_data)[:, 0] for t in self._tasks.table_tasks}
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            y = self.predictor(table_data)
            return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")
