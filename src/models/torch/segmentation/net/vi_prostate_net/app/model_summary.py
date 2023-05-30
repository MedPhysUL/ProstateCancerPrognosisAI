"""
    @file:              model_summary.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file prints a summary of VIProstateNet.
"""

import torch
from torchsummary import summary

from src.models.torch.segmentation.net.vi_prostate_net.vi_prostate_net import VIProstateNet


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = VIProstateNet(
        channels=(64, 128, 256, 512, 1024)
    ).to(device)

    print(net)      # Model Architecture
    summary(net)    # Summary
