"""
    @file:              cnn.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 06/2023

    @Description:       This file is used to define a CNN model.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Union

from torch import cat, mean, Tensor
from torch import device as torch_device
from torch.nn import DataParallel, Module, Sequential

from .base import Extractor, ExtractorOutput, ModelMode, MultiTaskMode
from .blocks import EncoderBlock
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _Encoder(Module):
    """
    This class is used to define an encoder.
    """

    def __init__(self, conv_sequence: Sequential):
        """
        Initializes the model.

        Parameters
        ----------
        conv_sequence : Sequential
            The convolutional sequence.
        """
        super().__init__()
        self.conv_sequence = conv_sequence

    def forward(self, input_tensor: Union[Tensor]) -> ExtractorOutput:
        """
        Forward pass.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        output : ExtractorOutput
            The output of the forward pass. It contains the deep features.
        """
        dim = tuple(range(2, len(input_tensor.shape)))
        x = input_tensor
        features = []
        for i, conv in enumerate(self.conv_sequence):
            x = conv(x)
            global_average_pool = mean(x, dim=dim)
            features.append(global_average_pool)

        features = cat(features, dim=1)

        return ExtractorOutput(deep_features=features, segmentation=None)


class CNN(Extractor):
    """
    A convolutional neural network used to extract deep radiomics from 3D medical images. It can also be used to perform
    predictions using extracted radiomics. The model is based on the 'Classifier' model from MONAI.
    """

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            model_mode: Union[str, ModelMode] = ModelMode.PREDICTION,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Union[str, Sequence[int]] = (128, 128, 128),
            n_features: int = 6,
            channels: Union[str, Sequence[int]] = (64, 128, 256, 512, 1024),
            strides: Optional[Sequence[int]] = None,
            kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 3,
            activation: str = "PRELU",
            norm: str = "INSTANCE",
            dropout_cnn: float = 0.0,
            dropout_fnn: float = 0.0,
            hidden_channels_fnn: Optional[Sequence[int]] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        image_keys : Union[str, List[str]]
            Sequence of images keys to extract deep radiomics from.
        model_mode : Union[str, ModelMode]
            Available modes are 'extraction' or 'prediction'. If 'extraction', the function will extract deep radiomics
            from input images. If 'prediction', the function will perform predictions using extracted radiomics.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'partly_shared' or 'fully_shared'. If 'partly_shared', a separate extractor model will
            be used for each task. If 'fully_shared', a fully shared extractor model will be used. All layers will be
            shared between the tasks.
        shape : Union[str, Sequence[int]]
            Sequence of integers stating the dimension of the input tensor (minus batch and channel dimensions). Can
            also be given as a string containing the sequence. Default to (128, 128, 128).
        n_features : int
            Integer stating the dimension of the final output tensor, i.e. the number of deep features to extract from
            the image.
        channels : Union[str, Sequence[int]]
            Sequence of integers stating the output channels of each convolutional layer. Can also be given as a string
            containing the sequence.
        strides : Optional[Sequence[int]]
            Sequence of integers stating the stride (downscale factor) of each convolutional layer. Default to 2.
        kernel_size : Union[Sequence[int], int]
            Integer or sequence of integers stating size of convolutional kernels.
        num_res_units : int
            Integer stating number of convolutions in residual units, 0 means no residual units.
        activation : str
             Name defining activation layers.
        norm : str
            Name or type defining normalization layers.
        dropout_cnn : float
            Dropout rate after each convolutional layer.
        dropout_fnn : float
            Dropout rate after each fully connected layer.
        hidden_channels_fnn : Optional[Sequence[int]]
            Sequence of integers stating the number of hidden units in each fully connected layer.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(
            image_keys=image_keys,
            model_mode=model_mode,
            multi_task_mode=multi_task_mode,
            shape=shape,
            n_features=n_features,
            activation=activation,
            channels=channels,
            dropout_fnn=dropout_fnn,
            hidden_channels_fnn=hidden_channels_fnn,
            device=device,
            name=name,
            seed=seed
        )

        self.strides = strides if strides else [2] * (len(channels) - 1)
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.norm = norm
        self.dropout_cnn = dropout_cnn

    def __get_layer(
            self,
            in_channels: int,
            out_channels: int,
            strides: int
    ) -> EncoderBlock:
        """
        Returns an encoder block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        strides : int
            Stride of the convolution.

        Returns
        -------
        encoder : EncoderBlock
            An encoder block.
        """
        return EncoderBlock(
            input_channels=in_channels,
            output_channels=out_channels,
            num_res_units=self.num_res_units,
            kernel_size=self.kernel_size,
            stride=strides,
            act=self.activation,
            norm=self.norm,
            dropout=self.dropout_cnn
        )

    def _get_conv_sequence(self):
        """
        Returns a convolutional sequence.

        Returns
        -------
        conv_sequence : Sequential
            The convolutional sequence.
        """
        conv_sequence = Sequential()
        for i, c in enumerate(self.channels):
            layer = self.__get_layer(
                in_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                out_channels=c,
                strides=1 if i == len(self.channels) - 1 else self.strides[i]
            )
            conv_sequence.add_module(
                name="conv_%i" % i,
                module=DataParallel(layer).to(self.device)
            )

        return conv_sequence

    def _build_deep_features_extractor(self, dataset: ProstateCancerDataset) -> Module:
        """
        Returns the deep features extractor module.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used to build the extractor.

        Returns
        -------
        extractor : Module
            The extractor module. It should take as input a tensor of shape (batch_size, channels, *spatial_shape) and
            return an ExtractorOutput object. The ExtractorOutput object contains the deep features extracted from the
            images and the segmentation of the images (optional).
        """
        conv_sequence = self._get_conv_sequence()

        return _Encoder(conv_sequence=conv_sequence)
