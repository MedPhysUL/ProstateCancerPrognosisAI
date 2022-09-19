"""
    @file:              base_block.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file contains an abstract BaseBlock object.
"""

from torch import nn, tensor


class BaseBlock(nn.Module):
    """
    Linear -> BatchNorm -> Activation -> Dropout
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: str,
            dropout: float = 0
    ):
        """
        Sets the layer attribute.

        Parameters
        ----------
        input_size : int
            Size of input.
        output_size : int
            Size of output.
        activation : str
            Name of the activation function.
        dropout: float
            Dropout probability.
        """
        # Call of parent's constructor
        super().__init__()

        # We save the length of the output
        self.__output_size = output_size

        # We create a list with the modules
        module_list = [
            nn.Linear(in_features=input_size, out_features=output_size),
            getattr(nn, activation)(),
            nn.BatchNorm1d(output_size)
        ]

        if dropout != 0:
            module_list.append(nn.Dropout(dropout))

        # We create a sequential from the list
        self.__layer = nn.Sequential(*module_list)

    @property
    def output_size(self):
        return self.__output_size

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
        return self.__layer(x)
