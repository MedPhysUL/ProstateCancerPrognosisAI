"""
    @file:              prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 03/2023

    @Description:       This file contains the ProstateNet class which is a deterministic segmentation model. The model
                        is a UNet type architecture incorporating residual units.
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import EncoderBlock, DecoderBlock


class ProstateNet(nn.Module):
    """
    A class that contains a deterministic segmentation model. The model is based on a UNet architecture and incorporates
    residual paths.
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

        self.enc1 = EncoderBlock(input_channels=1, output_channels=channels[0], dropout=dropout)
        self.enc2 = EncoderBlock(input_channels=channels[0], output_channels=channels[1], dropout=dropout)
        self.enc3 = EncoderBlock(input_channels=channels[1], output_channels=channels[2], dropout=dropout)
        self.enc4 = EncoderBlock(input_channels=channels[2], output_channels=channels[3], dropout=dropout)
        self.bottom = EncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1, dropout=dropout)
        self.dec4 = DecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2], dropout=dropout)
        self.dec3 = DecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1], dropout=dropout)
        self.dec2 = DecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0], dropout=dropout)
        self.dec1 = DecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward method of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input of the model (a medical image).

        Returns
        -------
        return : torch.Tensor
            Output of the model (a segmentation).
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottom = self.bottom(enc4)
        dec4 = self.dec4(torch.cat([enc4, bottom], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        return dec1
