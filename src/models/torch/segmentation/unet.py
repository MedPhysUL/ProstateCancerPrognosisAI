"""
    @file:              unet.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 07/2023

    @Description:       This file is used to define a 'Unet' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Optional, Sequence, Union

from torch import device as torch_device
from torch import cat, stack, sum, Tensor
from torch.nn import DataParallel, Module, ModuleDict, Sequential

from .base import Segmentor
from ..blocks import BayesianDecoderBlock, BayesianEncoderBlock, DecoderBlock, EncoderBlock
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _UNet(Module):
    """
    This class is used to define an UNet used for segmentation. Based on Monai.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[int, Sequence[int]],
            up_kernel_size: Union[int, Sequence[int]],
            num_res_units: int,
            act: str,
            norm: str,
            dropout: float,
            bias: bool,
            adn_ordering: str,
            bayesian: bool
    ):
        """
        Initializes the UNet.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        channels : Sequence[int]
            Sequence of integers stating the output channels of each convolutional layer.
        strides : Sequence[int]
            Sequence of integers stating the stride (downscale factor) of each convolutional layer. Its length needs to
            be the length of channels - 1.
        kernel_size : Union[int, Sequence[int]]
            Integer or sequence of 3 integers stating size of convolutional kernels.
        up_kernel_size : Union[int, Sequence[int]]
            Integer or sequence of 3 integers stating size of transposed convolutional kernels.
        num_res_units : int
            Integer stating number of convolutions in residual units, 0 means no residual units.
        act : str
             Name defining activation layers.
        norm : str
            Name or type defining normalization layers.
        dropout : float
            Probability of dropout.
        bias : bool
            Whether to have a bias term in convolution blocks.
        adn_ordering : str
            Order of operations in ADN layer.
        bayesian : bool
            Whether the model implements variational inference.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.bayesian = bayesian

    def _get_encoder_path(self) -> ModuleDict:
        """
        Returns a ModuleDict of encoder blocks to be used by the UNet in its encoder (down) path.

        Returns
        -------
        encoders : ModuleDict
            A ModuleDict of the encoder blocks to be used by the unet.
        """
        encoders = ModuleDict()
        for i, c in enumerate(self.channels):
            enc = Sequential()

            if self.bayesian:
                conv = BayesianEncoderBlock(
                    input_channels=self.in_channels if i == 0 else self.channels[i - 1],
                    output_channels=c,
                    num_res_units=self.num_res_units,
                    kernel_size=self.kernel_size,
                    stride=1 if i == len(self.channels) - 1 else self.strides[i],   # No downsizing in bottom layer.
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout
                )
            else:
                conv = EncoderBlock(
                    input_channels=self.in_channels if i == 0 else self.channels[i - 1],
                    output_channels=c,
                    num_res_units=self.num_res_units,
                    kernel_size=self.kernel_size,
                    stride=1 if i == len(self.channels) - 1 else self.strides[i],   # No downsizing in bottom layer.
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout
                )

            enc.add_module(name=f"conv{i}", module=conv)

            encoders["bottom" if i == len(self.channels) - 1 else f"layer{i}"] = enc

        return encoders

    def _get_decoder_path(self) -> ModuleDict:
        """
        Returns a ModuleDict of decoder blocks to be used by the UNet in its decoder (up) path.

        Returns
        -------
        decoders : ModuleDict
            A ModuleDict of the decoder blocks to be used by the unet.
        """
        decoders = ModuleDict()
        for i, c in enumerate(self.channels):

            if i < len(self.channels) - 1:
                dec = Sequential()

                if self.bayesian:
                    up_conv = BayesianDecoderBlock(
                        input_channels=c + self.channels[i + 1] if i == len(self.channels) - 2 else c * 2,
                        output_channels=self.out_channels if i == 0 else self.channels[i - 1],
                        num_res_units=self.num_res_units,
                        kernel_size=self.up_kernel_size,
                        stride=self.strides[i],
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        is_top=True if i == 0 else False
                    )
                else:
                    up_conv = DecoderBlock(
                        input_channels=c + self.channels[i + 1] if i == len(self.channels) - 2 else c * 2,
                        output_channels=self.out_channels if i == 0 else self.channels[i - 1],
                        num_res_units=self.num_res_units,
                        kernel_size=self.up_kernel_size,
                        stride=self.strides[i],
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        is_top=True if i == 0 else False
                    )

                dec.add_module(name=f"up_conv{i}", module=up_conv)

                decoders[f"layer{i}"] = dec

        return decoders

    def _bayesian_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass for bayesian UNet.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        output : tuple[Tensor, Tensor]
            A tuple of the segmentation and its KL divergence.
        """
        kl_list = []
        layers_output = {}
        for key, encoder in self.encoders.items():
            x, kl = encoder(x)
            layers_output[key] = x
            kl_list.append(kl)

        for key, decoder in reversed(list(self.decoders.items())):
            x, kl = decoder(cat([layers_output[key], x], dim=1))
            kl_list.append(kl)

        kl_divergence = sum(stack(kl_list))

        return x, kl_divergence

    def _deterministic_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for deterministic model.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        segmentation : Tensor
            The segmentation tensor.
        """
        layers_output = {}
        for key, encoder in self.encoders.items():
            x = encoder(x)
            layers_output[key] = x

        for key, decoder in reversed(list(self.decoders.items())):
            x = decoder(cat([layers_output[key], x], dim=1))

        return x

    def forward(self, input_tensor: Tensor):
        """
        Forward pass. Applies different forward methods depending on whether the model is bayesian.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        output : Union[Tensor, tuple[Tensor, Tensor]]
            The output Tensor of the forward pass. If in bayesian mode, the output is a tuple of the output tensor and
            its KL divergence.
        """
        if self.bayesian:
            return self._bayesian_forward(x=input_tensor)

        else:
            return self._deterministic_forward(x=input_tensor)


class Unet(Segmentor):
    """
    This class is used to define a 'Unet' model. It is a wrapper around the 'monai.networks.nets.UNet' class.
    """

    def __init__(
            self,
            image_keys: Union[str, Sequence[str]],
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
            seed: Optional[int] = None,
            bayesian: bool = False
    ):
        """
        Initializes the model.

        Parameters
        ----------
        image_keys : Union[str, Sequence[str]]
            Sequence of images keys to perform segmentation with.
        channels : Union[str, Sequence[int]]
            Sequence of integers stating the output channels of each convolutional layer. Can also be given as a string
            containing the sequence.
        strides : Optional[Sequence[int]]
            Sequence of integers stating the stride (downscale factor) of each convolutional layer. Has to be the length
            of channels - 1. Defaults to 2.
        kernel_size : Union[Sequence[int], int]
            Integer or sequence of integers stating size of convolutional kernels.
        up_kernel_size : Union[Sequence[int], int]
            Decoder block convolution kernel size, the value(s) should be odd. If sequence, its length should be 3.
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
        bayesian : bool
            Whether the model implements variational inference.
        """
        super().__init__(
            image_keys=image_keys,
            device=device,
            name=name,
            seed=seed,
            bayesian=bayesian
        )

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
        unet = _UNet(
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
            adn_ordering=self.adn_ordering,
            bayesian=self.bayesian
        )

        return DataParallel(unet).to(self.device)
