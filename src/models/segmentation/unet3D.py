"""
    @file:              unet3D.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define a 'Unet3D' model.
"""

from __future__ import annotations
from typing import Optional

from monai.networks.nets import UNet
from torch import device as torch_device

from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..base import check_if_built, TorchModel


class Unet3D(TorchModel):
    """
    Unet3D.
    """

    def __init__(
            self,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        super().__init__(device=device, name=name, seed=seed)
        self._net = None

    def build(self, dataset: ProstateCancerDataset) -> Unet3D:
        super().build(dataset=dataset)

        self._net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            dropout=0.2
        ).to(device=self.device)

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        y = self._net(features.image["CT"])
        y = {task.name: y for i, task in enumerate(self._tasks.segmentation_tasks)}

        return y
