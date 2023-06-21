"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     06/2022
    @Last modification: 06/2023

    @Description:       This file contains blocks to build image processing models.
"""

from typing import Sequence, Union

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
import numpy as np
from torch import Tensor
from torch.nn import Conv3d, ConvTranspose3d, Identity, Module, Sequential


class EncoderBlock(Module):
    """
    A class that contains an encoder block used for segmentation models. Based on Monai.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_res_units: int = 3,
            kernel_size: Union[Sequence[int], int] = 3,
            stride: int = 2,
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
        num_res_units : int
            Number of units to be implemented in parallel with the residual path.
        kernel_size : Union[int, Sequence[int]]
            Size of the kernel to be used in convolutions.
        stride : int
            Stride value of convolution. Implicitly controls downscaling.
        act : str
            Activation type to be used.
        norm : str
            Normalization method to be used.
        dropout : float
            Dropout probability to be used in NDA layers.
        """
        super().__init__()

        padding = same_padding(kernel_size=kernel_size, dilation=1)

        self.conv = Sequential()
        i = 0

        self.conv.add_module(
            f"conv{i}",
            Conv3d(
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
            self.residual = Identity()

            for subunit in range(num_res_units - 1):
                i += 1

                self.conv.add_module(
                    f"conv{i}",
                    Conv3d(
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

            self.residual = Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=res_kernel_size,
                stride=stride,
                padding=res_padding
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward method of the encoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        output : Tensor
            Output tensor.
        """
        y = self.conv(x)
        y_res = self.residual(x)

        return y + y_res


class DecoderBlock(Module):
    """
    A class that contains a decoder block used for segmentation models. Based on Monai.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_res_units: int = 3,
            kernel_size: Union[Sequence[int], int] = 3,
            stride: int = 2,
            act: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: float = 0.0,
            is_top: bool = False
    ):
        """
        Builds the operations used by the decoder block.

        Parameters
        ----------
        input_channels : int
            Number of channels of the input.
        output_channels : int
            Number of channels of the output.
        num_res_units : int
            Number of units to be implemented in parallel with the residual path.
        kernel_size : Union[int, Sequence[int]]
            Size of the kernel to be used in convolutions.
        stride : int
            Stride value of convolution. Implicitly controls downscaling.
        act : str
            Activation type to be used.
        norm : str
            Normalization method to be used.
        dropout : float
            Dropout probability to be used in NDA layers.
        is_top : bool
            Whether this block is in the last layer, meaning whether it is to be the last decoder block.
        """
        super().__init__()

        self.num_res_units = num_res_units

        padding = same_padding(kernel_size=kernel_size, dilation=1)
        output_padding = stride - 1

        self.up_conv = Sequential()

        self.up_conv.add_module(
            "up_conv",
            ConvTranspose3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=1,
                bias=True,
                dilation=1
            )
        )

        if not is_top or num_res_units > 0:
            self.up_conv.add_module(
                f"up_adn",
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
            self.residual = Identity()
            self.conv = Sequential()

            self.conv.add_module(
                'conv',
                Conv3d(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding
                )
            )

            if not is_top:
                self.conv.add_module(
                    "adn",
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward method of the decoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        output : Tensor
            Output tensor.
        """
        y = self.up_conv(x)

        if self.num_res_units > 0:
            y_res = self.residual(y)
            y = self.conv(y)
            return y + y_res

        else:
            return y
