"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 06/2023

    @Description:       This file contains blocks to build image processing models.
"""

from typing import Sequence, Union

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
import numpy as np
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    A class that contains an encoder block used for segmentation models. Based on Monai.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_res_units: int = 3,
            kernel_size: Union[Sequence[int], int] = 3,
            stride: Union[Sequence[int], int] = 2,
            act: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: float = 0.0
    ):
        """
        Builds the operations used by the encoder block.

        Parameters
        ----------
        input_channels : int
            Number of channels of the input.
        output_channels : int
            Number of channels of the output.
        stride : int
            Stride value of convolution. Implicitly controls downscaling.
        dropout : float
            Dropout probability to be used in NDA layers.
        """
        super().__init__()

        padding = same_padding(kernel_size=kernel_size, dilation=1)

        self.conv = nn.Sequential()
        i = 0

        self.conv.add_module(
            f"conv{i}",
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )

        self.conv.add_module(
            f"adn{i}",
            ADN(
                ordering="NDA",
                in_channels=output_channels,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=1,
                norm_dim=3
            )
        )

        if num_res_units > 0:
            self.residual = nn.Identity()

            for su in range(num_res_units - 1):
                i += 1

                self.conv.add_module(
                    f"conv{i}",
                    nn.Conv3d(
                        in_channels=output_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding
                    )
                )

                self.conv.add_module(
                    f"adn{i}",
                    ADN(
                        ordering="NDA",
                        in_channels=output_channels,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                        dropout_dim=1,
                        norm_dim=3
                    )
                )

        if np.prod(stride) != 1 or input_channels != output_channels:
            res_kernel_size = kernel_size
            res_padding = padding

            if np.prod(stride) == 1:  # if no downscaling, just up-channelling, 1x1 kernel is used with no padding
                res_kernel_size = 1
                res_padding = 0

            self.residual = nn.Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=res_kernel_size,
                stride=stride,
                padding=res_padding
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward method of the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.
        """
        y = self.conv(x)
        y_res = self.residual(x)

        return y + y_res
