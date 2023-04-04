"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2022
    @Last modification: 03/2023

    @Description:       This file contains blocks to build segmentation models.
"""

from bayesian_torch.layers.variational_layers.conv_variational import \
    Conv3dReparameterization, \
    ConvTranspose3dReparameterization
from monai.networks.blocks import ADN, Convolution, ResidualUnit
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    A class that contains an encoder block used for segmentation models.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            stride: int = 2,
            dropout: float = 0.2
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

        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
        )
        self.adn1 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
        )
        self.adn2 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )
        self.conv3 = nn.Conv3d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
        )
        self.adn3 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        # Residual path:
        if stride == 1:
            self.res = nn.Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=1,
                groups=1,
                bias=True
            )
        else:
            self.res = nn.Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                groups=1,
                bias=True
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
        return : torch.Tensor
            Output tensor.
        """
        y = self.conv1(x)
        y = self.adn1(y)
        y = self.conv2(y)
        y = self.adn2(y)
        y = self.conv3(y)
        y = self.adn3(y)
        y_res = self.res(x)

        return y + y_res


class DecoderBlock(nn.Module):
    """
    A class that contains a decoder block used for segmentation models.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            is_top: bool = False,
            dropout: float = 0.2
    ):
        """
        Builds the operations used by the decoder block.

        Parameters
        ----------
        input_channels : int
            Number of channels of the input.
        output_channels : int
            Number of channels of the output.
        is_top : bool
            Whether the instanced block is the top layer.
        dropout : float
            Dropout probability to be used in NDA layers.
        """
        super().__init__()

        self.is_top = is_top

        self.up_conv = nn.ConvTranspose3d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=1,
            bias=True,
            dilation=1
        )
        self.up_adn = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        # Residual Block After Up Scaling:
        self.conv1 = nn.Conv3d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
        )

        if not is_top:
            self.adn1 = ADN(
                ordering="NDA",
                in_channels=output_channels,
                act="PRELU",
                norm="INSTANCE",
                norm_dim=3,
                dropout=dropout,
                dropout_dim=1
            )

        self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward method of the decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        return : torch.Tensor
            Output tensor.
        """
        y0 = self.up_conv(x)
        y0 = self.up_adn(y0)

        y = self.conv1(y0)
        if not self.is_top:
            y = self.adn1(y)

        y_res = self.res(y0)
        return y + y_res


class BayesianEncoderBlock(nn.Module):
    """
    A class that contains a bayesian encoder block used for segmentation models.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            stride: int = 2,
            dropout: float = 0.2
    ):
        """
        Builds the operations used by the bayesian encoder block.

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

        self.conv1 = Conv3dReparameterization(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0
        )
        self.adn1 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        self.conv2 = Conv3dReparameterization(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0
        )
        self.adn2 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        self.conv3 = Conv3dReparameterization(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0
        )
        self.adn3 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        # Residual path:
        if stride == 1:
            self.res = Conv3dReparameterization(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                prior_mean=0,
                prior_variance=1,
                posterior_mu_init=0,
                posterior_rho_init=-3.0
            )
        else:
            self.res = Conv3dReparameterization(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
                prior_mean=0,
                prior_variance=1,
                posterior_mu_init=0,
                posterior_rho_init=-3.0
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Defines the forward method of the bayesian encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        return : tuple[torch.Tensor, float]
            Tuple containing the transformed tensor and the sum of the KL divergence of each probabilistic operation.
        """
        kl_sum = 0

        y, kl = self.conv1(x)
        kl_sum += kl
        y = self.adn1(y)

        y, kl = self.conv2(y)
        kl_sum += kl
        y = self.adn2(y)

        y, kl = self.conv3(y)
        kl_sum += kl
        y = self.adn3(y)

        y_res, kl = self.res(x)
        kl_sum += kl

        return y + y_res, kl_sum


class BayesianDecoderBlock(nn.Module):
    """
    A class that contains a bayesian decoder block used for segmentation models.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            is_top: bool = False,
            dropout: float = 0.2
    ):
        """
        Builds the operations used by the bayesian decoder block.

        Parameters
        ----------
        input_channels : int
            Number of channels of the input.
        output_channels : int
            Number of channels of the output.
        is_top : bool
            Whether the instanced block is the top layer.
        dropout : float
            Dropout probability to be used in NDA layers.
        """
        super().__init__()

        self.is_top = is_top

        self.up_conv = ConvTranspose3dReparameterization(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=1,
            bias=True,
            dilation=1,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0
        )
        self.up_adn = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=dropout,
            dropout_dim=1
        )

        # Residual Block After Up Scaling:
        self.conv1 = Conv3dReparameterization(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0
        )
        if not is_top:
            self.adn1 = ADN(
                ordering="NDA",
                in_channels=output_channels,
                act="PRELU",
                norm="INSTANCE",
                norm_dim=3,
                dropout=dropout,
                dropout_dim=1
            )

        self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Defines the forward method of the bayesian decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        return : tuple[torch.Tensor, float]
            Tuple containing the transformed tensor and the sum of the KL divergence of each probabilistic operation.
        """
        kl_sum = 0

        y0, kl = self.up_conv(x)
        kl_sum += kl
        y0 = self.up_adn(y0)

        y, kl = self.conv1(y0)
        kl_sum += kl
        if not self.is_top:
            y = self.adn1(y)

        y_res = self.res(y0)

        return y + y_res, kl_sum


# Monai Blocks

class _MonaiEncoderBlock(ResidualUnit):
    """
    Unused.
    """

    def __init__(self, input_channels, output_channels, stride=2):
        super().__init__(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=output_channels,
            strides=stride,
            kernel_size=(3, 3, 3),
            subunits=3,
            adn_ordering="NDA",
            act="PRELU",
            norm="INSTANCE",
            dropout=0.2,
            bias=True
        )


class _MonaiDecoderBlock(nn.Sequential):
    """
    Unused.
    """
    def __init__(self, input_channels, output_channels, is_top=False):
        conv = Convolution(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=output_channels,
            strides=2,
            kernel_size=(3, 3, 3),
            adn_ordering="NDA",
            act="PRELU",
            norm="INSTANCE",
            dropout=0.2,
            bias=True,
            conv_only=False,
            is_transposed=True
        )
        ru = ResidualUnit(
                spatial_dims=3,
                in_channels=output_channels,
                out_channels=output_channels,
                strides=1,
                kernel_size=(3, 3, 3),
                subunits=1,
                act="PRELU",
                norm="INSTANCE",
                dropout=0.2,
                bias=True,
                adn_ordering="NDA",
                last_conv_only=is_top
            )

        super().__init__(conv, ru)
