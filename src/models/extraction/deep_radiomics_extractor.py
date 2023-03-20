"""
    @file:              deep_radiomics_extractor.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define a 'DeepRadiomicsExtractor' model.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union

from monai.networks.nets import Classifier
from torch import cat, where, unsqueeze
from torch import device as torch_device
from torch.nn import Linear

from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..base import check_if_built, TorchModel


class DeepRadiomicsExtractor(TorchModel):
    """
    Deep radiomics extractor.
    """

    def __init__(
            self,
            in_shape: Sequence[int],
            n_radiomics: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 2,
            act: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: Optional[float] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self.channels = channels
        self.in_shape = in_shape
        self.n_radiomics = n_radiomics
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout

        self._linear_layer = None
        self._net = None

    @property
    def mode(self):
        return "mask" if self.in_shape[0] == 1 else "channel"

    def build(self, dataset: ProstateCancerDataset) -> DeepRadiomicsExtractor:
        super().build(dataset=dataset)

        table_output_size = len(dataset.table_dataset.tasks)

        self._net = Classifier(
            in_shape=self.in_shape,
            classes=self.n_radiomics,
            channels=self.channels,
            strides=self.strides,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout
        ).to(self.device)

        self._linear_layer = Linear(
            in_features=self.n_radiomics,
            out_features=table_output_size
        ).to(self.device)

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        if self.mode == "channel":
            x_image = cat(list(features.image.values()), 1)
        elif self.mode == "mask":
            x_image = where(features.image["CT_Prostate"] == 1, features.image["PT"], 0.0)
        else:
            raise AssertionError(f"mode is either 'mask' or 'channel'. Found {self.mode}")

        # We compute the output
        y = self._net(x_image)
        y = self._linear_layer(y)

        y = {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        return y
