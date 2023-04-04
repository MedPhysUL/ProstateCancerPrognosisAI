"""
    @file:              model_summary.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file prints a summary of VIProstateNet.
"""

from torchsummary import summary

from src.models.segmentation.net.vi_prostate_net.vi_prostate_net import VIProstateNet


if __name__ == "__main__":
    net = VIProstateNet(
        channels=(4, 8, 16, 32, 64)         # TODO
    )

    print(net)      # Model Architecture
    summary(net)    # Summary
