"""
    @file:              model_summary.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       Description.
"""

from torchsummary import summary

from src.models.segmentation.net.prostate_net.prostate_net import ProstateNet


if __name__ == '__main__':
    net = ProstateNet(
        channels=(32, 64, 128, 256, 512)
    )

    print(net)      # Model Architecture
    summary(net)    # Summary
