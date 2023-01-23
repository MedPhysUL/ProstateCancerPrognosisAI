"""
    @file:              vimh_prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       Description
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import EncoderBlock, DecoderBlock, BayesianDecoderBlock


# EXAMPLE DEPTH DEC3 # TODO


class DeterministicBase(nn.Module):
    """
    Description.
    """
    def __init__(self, channels):
        super().__init__()

        self.enc1 = EncoderBlock(input_channels=1, output_channels=channels[0])
        self.enc2 = EncoderBlock(input_channels=channels[0], output_channels=channels[1])
        self.enc3 = EncoderBlock(input_channels=channels[1], output_channels=channels[2])
        self.enc4 = EncoderBlock(input_channels=channels[2], output_channels=channels[3])
        self.bottom = EncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1)
        self.dec4 = DecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2])

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottom = self.bottom(enc4)
        dec4 = self.dec4(torch.cat([enc4, bottom], dim=1))

        return enc1, enc2, enc3, dec4


class BayesianHead(nn.Module):
    """
    Description.
    """

    def __init__(self, channels):
        super().__init__()

        self.dec3 = BayesianDecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1])
        self.dec2 = BayesianDecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0])
        self.dec1 = BayesianDecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True)

    def forward(self, enc1, enc2, enc3, dec4):
        kl_sum = 0

        dec3, kl = self.dec3(torch.cat([enc3, dec4], dim=1))
        kl_sum += kl

        dec2, kl = self.dec2(torch.cat([enc2, dec3], dim=1))
        kl_sum += kl

        dec1, kl = self.dec1(torch.cat([enc1, dec2], dim=1))
        kl_sum += kl

        return dec1, kl_sum


class VIMHProstateNet(nn.Module):
    """
    Description.
    """

    def __init__(self, num_heads, channels):
        super().__init__()

        self.num_heads = num_heads

        self.base = DeterministicBase(channels)
        self.heads = nn.ModuleList([BayesianHead(channels) for _ in range(num_heads)])

    def forward(self, x):
        enc1, enc2, enc3, dec4 = self.base(x)

        y_list = []
        kl_list = []
        for i in range(self.num_heads):
            y, kl = self.heads[i](enc1, enc2, enc3, dec4)

            y_list.append(y)
            kl_list.append(kl)

        return torch.stack(y_list), torch.stack(kl_list)
