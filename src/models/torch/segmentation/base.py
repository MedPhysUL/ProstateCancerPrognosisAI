"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 04/2023

    @Description:       This file is used to define an abstract 'Segmentor' model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

from torch import cat, Tensor
from torch import device as torch_device
from torch.nn import Module

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ....tasks import SegmentationTask


class Segmentor(TorchModel, ABC):
    """
    Abstract class for all segmentors. A segmentor is a model that takes as input a set of images and outputs a set
    of segmentations.
    """

    def __init__(
            self,
            image_keys: Union[str, Sequence[str]],
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        image_keys : Union[str, Sequence[str]]
            Sequence of images keys to perform segmentation with.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.image_keys = [image_keys] if isinstance(image_keys, str) else image_keys
        self.segmentor = None

    @abstractmethod
    def _build_segmentor(self, dataset) -> Module:
        """
        Returns the segmentor module.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used to build the extractor.

        Returns
        -------
        segmentor : Module
            The segmentor module.
        """
        raise NotImplementedError

    def build(self, dataset: ProstateCancerDataset) -> Segmentor:
        """
        Builds the model. This method must be called before training the model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset to build the model for.

        Returns
        -------
        self : Segmentor
            The built model.
        """
        super().build(dataset=dataset)

        self.segmentor = self._build_segmentor(dataset)

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> Union[Tensor, TargetsType]:
        """
        Performs a forward pass.

        Parameters
        ----------
        features : FeaturesType
            The features to perform the forward pass on. Must contain the images to perform the segmentation on.

        Returns
        -------
        targets : Union[Tensor, TargetsType]
            The targets of the forward pass.
        """
        segmentation = self.segmentor(cat([features.image[k] for k in self.image_keys], 1))
        return {task.name: segmentation[:, i, None] for i, task in enumerate(self._tasks.segmentation_tasks)}