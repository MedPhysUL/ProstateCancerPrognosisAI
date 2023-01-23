"""
    @file:              prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       Description
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import EncoderBlock, DecoderBlock


class ProstateNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.enc1 = EncoderBlock(input_channels=1, output_channels=channels[0])
        self.enc2 = EncoderBlock(input_channels=channels[0], output_channels=channels[1])
        self.enc3 = EncoderBlock(input_channels=channels[1], output_channels=channels[2])
        self.enc4 = EncoderBlock(input_channels=channels[2], output_channels=channels[3])
        self.bottom = EncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1)
        self.dec4 = DecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2])
        self.dec3 = DecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1])
        self.dec2 = DecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0])
        self.dec1 = DecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True)

    def forward(self, x):
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
