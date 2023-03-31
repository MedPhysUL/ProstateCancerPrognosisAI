"""
    @file:              cnn.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define a 'CNN' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Callable, Optional, Sequence, Union

from monai.networks.nets import Classifier
from torch import cat
from torch import device as torch_device
from torch.nn import Linear

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType


class CNN(TorchModel):
    """
    Deep radiomics extractor.
    """

    def __init__(
            self,
            in_shape: Union[str, Sequence[int]],
            n_features: int,
            channels: Union[str, Sequence[int]],
            strides: Optional[Sequence[int]] = None,
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

        self.in_shape = literal_eval(in_shape) if isinstance(in_shape, str) else in_shape
        self.n_features = n_features

        self.extractor = Classifier(
            in_shape=self.in_shape,
            classes=self.n_features,
            channels=literal_eval(channels) if isinstance(channels, str) else channels,
            strides=strides if strides else [2] * (len(channels) - 1),
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout
        ).to(self.device)

        self.linear_layer = None
        self._input_getter = self._build_input_getter()
        self._extraction, self._prediction = False, True

    def extraction(self):
        self._extraction, self._prediction = True, False

    def prediction(self):
        self._extraction, self._prediction = False, True

    def _build_input_getter(self) -> Callable:
        if self.in_shape[0] == 1:
            return lambda features: features.image["PT"]*features.image["CT_Prostate"]
        elif self.in_shape[0] == 2:
            return lambda features: cat(list(features.image.values()), 1)
        else:
            raise AssertionError(f"'in_shape' first element must be either 1 or 2. Got {self.in_shape[0]}.")

    def build(self, dataset: ProstateCancerDataset) -> CNN:
        super().build(dataset=dataset)

        table_output_size = len(dataset.table_dataset.tasks)

        self.linear_layer = Linear(
            in_features=self.n_features,
            out_features=table_output_size
        ).to(self.device)

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        x_image = self._input_getter(features)
        radiomics = self.extractor(x_image)

        if self._prediction:
            y = self.linear_layer(radiomics)
            y = {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
            return y
        elif self._extraction:
            return radiomics
