"""
    @file:              mlp.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define an 'MLP' model.
"""

from __future__ import annotations
from typing import List, Optional

from torch import stack
from torch import device as torch_device
from torch.nn import Identity, Linear

from ..base.blocks.encoders import MLPEncodingBlock
from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from .torch_model import check_if_built, TorchModel


class MLP(TorchModel):
    """
    Multilayer perceptron model.
    """

    def __init__(
            self,
            activation: str,
            layers: List[int],
            dropout: float = 0,
            device: Optional[torch_device] = None,
            name: Optional[str] = None
    ):
        """
        Builds the layers of the model and sets other protected attributes.

        Parameters
        ----------
        activation : str
            Activation function
        layers : List[int]
            List with number of units in each hidden layer
        dropout : float
            Probability of dropout
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        """
        super().__init__(device=device, name=name)

        self.activation = activation
        self.dropout = dropout
        self.layers = layers

        self._linear_layer = None
        self._main_encoding_block = None

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

        if len(self.layers) > 0:
            self._main_encoding_block = MLPEncodingBlock(
                input_size=table_input_size,
                output_size=self.layers[-1],
                layers=self.layers[:-1],
                activation=self.activation,
                dropout=self.dropout
            )
        else:
            self._main_encoding_block = Identity()
            self.layers.append(table_input_size)

        self._main_encoding_block = self._main_encoding_block.to(self.device)
        self._linear_layer = Linear(self.layers[-1], table_output_size).to(self.device)

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
        # We retrieve the table data only and transform the input dictionary to a tensor
        x_table = stack(list(features.table.values()), 1)

        # We compute the output
        y_table = self._linear_layer(self._main_encoding_block(x_table.float()))

        y = {task.name: y_table[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        return y
