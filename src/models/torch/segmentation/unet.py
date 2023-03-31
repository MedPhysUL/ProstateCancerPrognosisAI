"""
    @file:              unet.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define a 'Unet3D' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Optional, Sequence, Union

from monai.networks.nets import UNet
from torch import device as torch_device

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, TargetsType


class Unet(TorchModel):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Union[str, Sequence[int]],
            strides: Optional[Sequence[int]] = None,
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 0,
            act: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self.network = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=literal_eval(channels) if isinstance(channels, str) else channels,
            strides=strides if strides else [2] * (len(channels) - 1),
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering
        ).to(device=self.device)

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        y = self.network(features.image["CT"])
        y = {task.name: y for i, task in enumerate(self._tasks.segmentation_tasks)}

        return y
