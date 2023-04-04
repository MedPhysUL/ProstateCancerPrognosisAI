"""
    @file:              vi_prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 03/2023

    @Description:       This file contains the VIProstateNet class which is a bayesian segmentation model. The model is
                        a variational inference version of ProstateNet.
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import BayesianDecoderBlock, BayesianEncoderBlock


class VIProstateNet(nn.Module):
    """
    A class that contains a VI segmentation model. The model is based on a UNet architecture and incorporates
    residual paths. Variational inference is used.
    """

    def __init__(
            self,
            channels: tuple[int, int, int, int, int],
            dropout: float = 0.2
    ):
        """
        Builds the layers of the model.

        Parameters
        ----------
        channels : tuple[int, int, int, int, int]
            Tuple of the amount of channels at each of the 5 layers of the model.
        dropout : float
            Dropout probability.
        """
        super().__init__()

        self.enc1 = BayesianEncoderBlock(input_channels=1, output_channels=channels[0], dropout=dropout)
        self.enc2 = BayesianEncoderBlock(input_channels=channels[0], output_channels=channels[1], dropout=dropout)
        self.enc3 = BayesianEncoderBlock(input_channels=channels[1], output_channels=channels[2], dropout=dropout)
        self.enc4 = BayesianEncoderBlock(input_channels=channels[2], output_channels=channels[3], dropout=dropout)
        self.bottom = BayesianEncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1, dropout=dropout)
        self.dec4 = BayesianDecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2], dropout=dropout)
        self.dec3 = BayesianDecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1], dropout=dropout)
        self.dec2 = BayesianDecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0], dropout=dropout)
        self.dec1 = BayesianDecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Defines the forward method of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input of the model (a medical image).

        Returns
        -------
        return : tuple[torch.Tensor, float]
            Tuple containing the transformed tensor (i.e. the segmentation) and the sum of the KL divergence of each
            layer.
        """
        kl_sum = 0

        enc1, kl = self.enc1(x)
        kl_sum += kl

        enc2, kl = self.enc2(enc1)
        kl_sum += kl

        enc3, kl = self.enc3(enc2)
        kl_sum += kl

        enc4, kl = self.enc4(enc3)
        kl_sum += kl

        bottom, kl = self.bottom(enc4)
        kl_sum += kl

        dec4, kl = self.dec4(torch.cat([enc4, bottom], dim=1))
        kl_sum += kl

        dec3, kl = self.dec3(torch.cat([enc3, dec4], dim=1))
        kl_sum += kl

        dec2, kl = self.dec2(torch.cat([enc2, dec3], dim=1))
        kl_sum += kl

        dec1, kl = self.dec1(torch.cat([enc1, dec2], dim=1))
        kl_sum += kl

        return dec1, kl_sum
