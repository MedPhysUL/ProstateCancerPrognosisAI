"""
    @file:              encoders.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file contains an abstract class named Encoder and an MLPEncodingBlock.
"""

from abc import ABC
from typing import List

from torch import nn, tensor

from src.models.base.blocks.base_block import BaseBlock


class Encoder(ABC):
    """
    Abstract class for encoder.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int
    ):
        """
        Saves the input size and the output size.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of features in the encodings.
        """
        self._input_size = input_size
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size


class MLPEncodingBlock(Encoder, nn.Module):
    """
    An MLP encoding block is basically an MLP without prediction function.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layers: List[int],
            activation: str,
            dropout: float
    ):
        """
        Builds the layers of the encoding model.

        Parameters
        ----------
        input_size : int
            Number of features in the input.
        output_size : int
            Number of nodes in the last layer of the neural network.
        layers : List[int]
            List with number of units in each hidden layer.
        activation : str
            Name of the activation function.
        dropout : float
            Probability of dropout.
        """
        # Call of both parent constructors
        Encoder.__init__(self, input_size=input_size, output_size=output_size)
        nn.Module.__init__(self)

        # We create the layers
        layers.insert(0, input_size)
        layers.append(output_size)
        self.__layers = nn.Sequential(
            *[
                BaseBlock(
                    input_size=layers[i - 1],
                    output_size=layers[i],
                    activation=activation,
                    dropout=dropout
                ) for i in range(1, len(layers))
            ]
        )

    def forward(
            self,
            x: tensor
    ) -> tensor:
        """
        Executes the forward pass.

        Parameters
        ----------
        x : tensor
            (N,D) tensor with D-dimensional samples.

        Returns
        -------
        (N, D') tensor with concatenated embedding
        """
        return self.__layers(x)
