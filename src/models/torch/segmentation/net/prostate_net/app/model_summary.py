"""
    @file:              model_summary.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       This file prints a summary of ProstateNet
"""

import torch
from torchsummary import summary

from src.models.torch.segmentation.net.prostate_net.prostate_net import ProstateNet


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ProstateNet(
        channels=(64, 128, 256, 512, 1024)
    ).to(device)

    print(net)      # Model Architecture
    summary(net)    # Summary
