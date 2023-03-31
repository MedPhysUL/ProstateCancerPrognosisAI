"""
    @file:              mlp.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define an 'MLP' model.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

from monai.networks.nets import FullyConnectedNet
from torch import stack
from torch import device as torch_device

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType


class MLP(TorchModel):
    """
    Multilayer perceptron model.
    """

    def __init__(
            self,
            hidden_channels: Sequence[int],
            activation: Optional[Union[Tuple, str]] = None,
            adn_ordering: Optional[str] = None,
            bias: bool = True,
            dropout: Optional[Union[Tuple, str, float]] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Builds the layers of the model and sets other protected attributes.

        Parameters
        ----------
        hidden_channels : Sequence[int]
            List with number of units in each hidden layer.
        activation : str
            Activation function.
        adn_ordering : Optional[str]
            A string representing the ordering of activation, dropout, and normalization. Defaults to "NDA".
        bias : bool
            If `bias` is True then linear units have a bias term.
        dropout : float
            Probability of dropout.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.hidden_channels = hidden_channels
        self.activation = activation
        self.adn_ordering = adn_ordering
        self.bias = bias
        self.dropout = dropout

        self.network = None

    def build(self, dataset: ProstateCancerDataset) -> MLP:
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : MLP
            The current model.
        """
        super().build(dataset=dataset)

        table_input_size, table_output_size = len(dataset.table_dataset.features_cols), len(dataset.table_dataset.tasks)

        self.network = FullyConnectedNet(
            in_channels=table_input_size,
            out_channels=table_output_size,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            act=self.activation,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        ).to(self.device)

        return self

    @check_if_built
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
        x_table = stack(list(features.table.values()), 1)
        y_table = self.network(x_table.float())
        y = {task.name: y_table[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        return y
