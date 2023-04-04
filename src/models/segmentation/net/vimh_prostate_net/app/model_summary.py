"""
    @file:              model_summary.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file prints a summary of VIMHProstateNet.
"""

from torchsummary import summary

from src.models.segmentation.net.vimh_prostate_net.vimh_prostate_net import VIMHProstateNet


if __name__ == "__main__":
    net = VIMHProstateNet(
        num_heads=4,
        channels=(4, 8, 16, 32, 64)         # TODO
    )

    print(net)      # Model Architecture
    summary(net)    # Summary
