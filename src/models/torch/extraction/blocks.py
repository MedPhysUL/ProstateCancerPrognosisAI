"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     06/2022
    @Last modification: 06/2023

    @Description:       This file contains blocks to build image processing models.
"""

from typing import Sequence, Union

import torch
from bayesian_torch.layers.variational_layers.conv_variational import (
    Conv3dReparameterization,
    ConvTranspose3dReparameterization
)
from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
import numpy as np
from torch import cat, sum, Tensor
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

        self.num_res_units = num_res_units

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

        if self.num_res_units > 0:
            y_res = self.residual(x)
            return y + y_res

        else:
            return y


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

        # Residual block following upsizing
        if self.num_res_units > 0:
            y_res = self.residual(y)
            y = self.conv(y)
            return y + y_res

        else:
            return y


class BayesianEncoderBlock(Module):
    """
    A class that contains a bayesian encoder block used for segmentation models. It implements variational inference for
    each convolution used.
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
            prior_mean: float = 0.0,
            prior_variance: float = 0.1,
            posterior_mu_init: float = 0.0,
            posterior_rho_init: float = -3.0
    ):
        """
        Builds the operations used by the bayesian encoder block. Implements variational inference for
        each convolution layer used but not for the NDA layers used.

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
        prior_mean : float
            Mean of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        prior_variance : float
            Variance of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        posterior_mu_init : float
            Initial value of the trainable mu parameter representing the mean of the Gaussian approximate of the
            posterior distribution.
        posterior_rho_init : float
            Initial value of the trainable rho parameter representing the sigma (standard deviation) of the Gaussian
            approximate of the posterior distribution.
        """
        super().__init__()

        self.num_res_units = num_res_units

        padding = same_padding(kernel_size=kernel_size, dilation=1)

        self.conv = Sequential()
        i = 0

        self.conv.add_module(
            f"conv{i}",
            Conv3dReparameterization(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
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
                    Conv3dReparameterization(
                        in_channels=output_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        prior_mean=prior_mean,
                        prior_variance=prior_variance,
                        posterior_mu_init=posterior_mu_init,
                        posterior_rho_init=posterior_rho_init
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

            self.residual = Conv3dReparameterization(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=res_kernel_size,
                stride=stride,
                padding=res_padding,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Defines the forward method of the bayesian encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        return : tuple[Tensor, Tensor]
            Tuple containing the transformed tensor and the sum of the KL divergence of each probabilistic operation.
        """
        kl_list = []

        y = x
        for name, module in self.conv.named_children():
            if name.startswith("conv"):     # Module is a convolution layer
                y, kl = module(y)
                kl_list.append(kl)

            else:                           # Module is a NDA layer
                y = module(y)

        if self.num_res_units > 0:
            y_res, kl = self.residual(x)
            kl_list.append(kl)

            return y + y_res, sum(cat(kl_list))

        else:
            return y, sum(cat(kl_list))


class BayesianDecoderBlock(Module):
    """
    A class that contains a bayesian decoder block used for segmentation models.
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
            is_top: bool = False,
            prior_mean: float = 0.0,
            prior_variance: float = 0.1,
            posterior_mu_init: float = 0.0,
            posterior_rho_init: float = -3.0,
    ):
        """
        Builds the operations used by the bayesian encoder block. Implements variational inference for
        each convolution layer used but not for the NDA layers used.

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
        prior_mean : float
            Mean of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        prior_variance : float
            Variance of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        posterior_mu_init : float
            Initial value of the trainable mu parameter representing the mean of the Gaussian approximate of the
            posterior distribution.
        posterior_rho_init : float
            Initial value of the trainable rho parameter representing the sigma (standard deviation) of the Gaussian
            approximate of the posterior distribution.
        """
        super().__init__()

        self.num_res_units = num_res_units

        padding = same_padding(kernel_size=kernel_size, dilation=1)
        output_padding = stride - 1

        self.up_conv = Sequential()

        self.up_conv.add_module(
            "conv_up",
            ConvTranspose3dReparameterization(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=1,
                bias=True,
                dilation=1,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )
        )

        if not is_top or num_res_units > 0:
            self.up_conv.add_module(
                f"adn_up",
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
                Conv3dReparameterization(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    prior_mean=prior_mean,
                    prior_variance=prior_variance,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Defines the forward method of the bayesian decoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        output : tuple[Tensor, Tensor]
            Tuple containing the transformed tensor and the sum of the KL divergence of each probabilistic operation.
        """
        kl_list = []

        for name, module in self.up_conv.named_children():
            if name.startswith("conv"):     # Module is a convolution layer
                x, kl = module(x)
                kl_list.append(kl)

            else:                           # Module is a NDA layer
                x = module(x)

        # Residual block following upsizing
        if self.num_res_units > 0:
            res = self.residual(x)

            for name, module in self.conv.named_children():
                if name.startswith("conv"):     # Module is a convolution layer
                    x, kl = module(x)
                    kl_list.append(kl)
                else:                           # Module is a NDA layer
                    x = module(x)

            return x + res, sum(cat(kl_list))

        else:
            return x, sum(cat(kl_list))
