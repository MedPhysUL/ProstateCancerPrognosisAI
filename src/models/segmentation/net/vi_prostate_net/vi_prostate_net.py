"""
    @file:              vi_prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       Description
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import BayesianDecoderBlock, BayesianEncoderBlock


class VIProstateNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.enc1 = BayesianEncoderBlock(input_channels=1, output_channels=channels[0])
        self.enc2 = BayesianEncoderBlock(input_channels=channels[0], output_channels=channels[1])
        self.enc3 = BayesianEncoderBlock(input_channels=channels[1], output_channels=channels[2])
        self.enc4 = BayesianEncoderBlock(input_channels=channels[2], output_channels=channels[3])
        self.bottom = BayesianEncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1)
        self.dec4 = BayesianDecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2])
        self.dec3 = BayesianDecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1])
        self.dec2 = BayesianDecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0])
        self.dec1 = BayesianDecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True)

    def forward(self, x):
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
