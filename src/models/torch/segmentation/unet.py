"""
    @file:              unet.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 04/2023

    @Description:       This file is used to define a 'Unet' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Optional, Sequence, Union

from monai.networks.nets import UNet
from torch import device as torch_device
from torch.nn import DataParallel, Module

from .base import Segmentor
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class Unet(Segmentor):
    """
    This class is used to define a 'Unet' model. It is a wrapper around the 'monai.networks.nets.UNet' class.
    """

    def __init__(
            self,
            image_keys: Union[str, Sequence[str]],
            spatial_dims: int = 3,
            channels: Union[str, Sequence[int]] = (64, 128, 256, 512, 1024),
            strides: Optional[Sequence[int]] = None,
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 0,
            activation: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
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
        spatial_dims : int
            Number of spatial dimensions. Default to 3.
        channels : Union[str, Sequence[int]]
            Sequence of integers stating the output channels of each convolutional layer. Can also be given as a string
            containing the sequence.
        strides : Optional[Sequence[int]]
            Sequence of integers stating the stride (downscale factor) of each convolutional layer. Default to 2.
        kernel_size : Union[Sequence[int], int]
            Integer or sequence of integers stating size of convolutional kernels.
        up_kernel_size : Union[Sequence[int], int]
            Upsampling convolution kernel size, the value(s) should be odd. If sequence, its length should equal to
            dimensions.
        num_res_units : int
            Integer stating number of convolutions in residual units, 0 means no residual units.
        activation : str
             Name defining activation layers.
        norm : str
            Name or type defining normalization layers.
        dropout : Optional[float]
            Probability of dropout.
        bias : bool
            Whether to have a bias term in convolution blocks.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(
            image_keys=image_keys,
            device=device,
            name=name,
            seed=seed
        )

        self.spatial_dims = spatial_dims
        self.channels = literal_eval(channels) if isinstance(channels, str) else channels
        self.strides = strides if strides else [2] * (len(channels) - 1)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

    def _build_segmentor(self, dataset: ProstateCancerDataset) -> Module:
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
        unet = UNet(
            spatial_dims=self.spatial_dims,
            in_channels=len(self.image_keys),
            out_channels=len(self._tasks.segmentation_tasks),
            channels=self.channels,
            strides=self.strides,
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            num_res_units=self.num_res_units,
            act=self.activation,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        )
        unet = DataParallel(unet)

        return unet.to(self.device)
