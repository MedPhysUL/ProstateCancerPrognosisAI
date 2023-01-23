"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2022
    @Last modification: 01/2023

    @Description:       Description
"""

from monai.networks.blocks import ADN, Convolution, ResidualUnit
import torch.nn as nn
from bayesian_torch.layers.variational_layers.conv_variational import Conv3dReparameterization, ConvTranspose3dReparameterization


class _MonaiEncoderBlock(ResidualUnit):
    """
    Description.
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
    Description.
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


class EncoderBlock(nn.Module):
    """
    Description.
    """

    def __init__(self, input_channels, output_channels, stride=2):
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
            dropout=0.2,
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
            dropout=0.2,
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
            dropout=0.2,
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

    def forward(self, x):
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
    Description.
    """
    def __init__(self, input_channels, output_channels, is_top=False):
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
            dropout=0.2,
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
                dropout=0.2,
                dropout_dim=1
            )

        self.res = nn.Identity()

    def forward(self, x):
        y0 = self.up_conv(x)
        y0 = self.up_adn(y0)

        y = self.conv1(y0)
        if not self.is_top:
            y = self.adn1(y)

        y_res = self.res(y0)
        return y + y_res

# TODO
a = 0
b = 1


class BayesianEncoderBlock(nn.Module):
    """
    Description.
    """

    def __init__(self, input_channels, output_channels, stride=2):
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
            prior_mean=a,             # TODO
            prior_variance=b,
            posterior_mu_init=a,
            posterior_rho_init=b
        )
        self.adn1 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=0.2,
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
            prior_mean=a,         # TODO
            prior_variance=b,
            posterior_mu_init=a,
            posterior_rho_init=b
        )
        self.adn2 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=0.2,
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
            prior_mean=a,             # TODO
            prior_variance=b,
            posterior_mu_init=a,
            posterior_rho_init=b
        )
        self.adn3 = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=0.2,
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
                prior_mean=a,                 # TODO
                prior_variance=b,
                posterior_mu_init=a,
                posterior_rho_init=b
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
                prior_mean=a,             # TODO
                prior_variance=b,
                posterior_mu_init=a,
                posterior_rho_init=b
            )

    def forward(self, x):
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
        kl_sum += kl                    # TODO -- ok?

        return y + y_res, kl_sum


class BayesianDecoderBlock(nn.Module):
    """
    Description.
    """
    def __init__(self, input_channels, output_channels, is_top=False):
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
            prior_mean=a,             # TODO
            prior_variance=b,
            posterior_mu_init=a,
            posterior_rho_init=b
        )
        self.up_adn = ADN(
            ordering="NDA",
            in_channels=output_channels,
            act="PRELU",
            norm="INSTANCE",
            norm_dim=3,
            dropout=0.2,
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
            prior_mean=a,               # TODO
            prior_variance=b,
            posterior_mu_init=a,
            posterior_rho_init=b
        )
        if not is_top:
            self.adn1 = ADN(
                ordering="NDA",
                in_channels=output_channels,
                act="PRELU",
                norm="INSTANCE",
                norm_dim=3,
                dropout=0.2,
                dropout_dim=1
            )

        self.res = nn.Identity()

    def forward(self, x):
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
