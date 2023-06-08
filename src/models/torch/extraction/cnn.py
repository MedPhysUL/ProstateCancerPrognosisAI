"""
    @file:              cnn.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 04/2023

    @Description:       This file is used to define a 'CNN' model.
"""

from __future__ import annotations
from ast import literal_eval
from copy import copy
from typing import List, Optional, Sequence, Union

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.nets import FullyConnectedNet
from torch import cat, mean, Tensor
from torch import device as torch_device
from torch.nn import DataParallel, Module, ModuleDict, Sequential

from .base import Extractor, MergingMethod, ModelMode, MultiTaskMode
from ....tasks import SegmentationTask
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _Encoder(Module):

    def __init__(self, conv_sequence: Sequential, linear_module: Module):
        """
        Initializes the model.

        Parameters
        ----------
        conv_sequence : Sequential
            The convolutional sequence.
        linear_module : Module
            The linear module.
        """
        super().__init__()
        self.conv_sequence = conv_sequence
        self.linear_module = linear_module

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor.
        """
        dim = tuple(range(2, len(input_tensor.shape)))
        x = input_tensor
        features = []
        for i, conv in enumerate(self.conv_sequence):
            x = conv(x)
            global_average_pool = mean(x, dim=dim)
            features.append(global_average_pool)

        features = cat(features, dim=1)
        y = self.linear_module(features)

        return y


class CNN(Extractor):
    """
    A convolutional neural network used to extract deep radiomics from 3D medical images. It can also be used to perform
    predictions using extracted radiomics. The model is based on the 'Classifier' model from MONAI.
    """

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            segmentation_key_or_task: Optional[str, SegmentationTask] = None,
            merging_method: Union[str, MergingMethod] = MergingMethod.CONCATENATION,
            model_mode: Union[str, ModelMode] = ModelMode.PREDICTION,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Union[str, Sequence[int]] = (128, 128, 128),
            n_features: int = 6,
            channels: Union[str, Sequence[int]] = (4, 8, 16, 32, 64),
            strides: Optional[Sequence[int]] = None,
            kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 2,
            activation: str = "PRELU",
            norm: str = "INSTANCE",
            dropout_cnn: float = 0.0,
            dropout_fnn: float = 0.0,
            hidden_channels_fnn: Optional[Sequence[int]] = None,
            partly_shared_convolutions: int = 2,
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
        segmentation_key_or_task : Optional[str, SegmentationTask]
            Key of the segmentation to merge with the images. If a segmentation task is given, the segmentation key will
            be extracted from the task. If None, the segmentation will not be merged with the images.
        merging_method : Union[str, MergingMethod]
            Available methods for merging the segmentation with the images are 'concatenation' or 'multiplication'. If
            'concatenation', the segmentation and image features are concatenated along the channel dimension. If
            'multiplication', the segmentation is element-wise multiplied with the image features.
        model_mode : Union[str, ModelMode]
            Available modes are 'extraction' or 'prediction'. If 'extraction', the function will extract deep radiomics
            from input images. If 'prediction', the function will perform predictions using extracted radiomics.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'separated', 'partly_shared' or 'fully_shared'. If 'separated', a separate extractor
            model is used for each task. If 'partly_shared', a partly shared extractor model is used. The first layers
            are shared between the tasks. If 'fully_shared', a fully shared extractor model is used. All layers are
            shared between the tasks.
        shape : Union[str, Sequence[int]]
            Sequence of integers stating the dimension of the input tensor (minus batch and channel dimensions). Can
            also be given as a string containing the sequence. Exemple: (96, 96, 96).
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
        partly_shared_convolutions : int
            Integer stating the number of convolutional layers that are shared between the tasks when using a partly
            shared extractor model. The first layers are shared, the last layers are not shared. Only used when
            multi_task_mode is 'partly_shared'. Default to 2.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(
            image_keys=image_keys,
            segmentation_key_or_task=segmentation_key_or_task,
            merging_method=merging_method,
            model_mode=model_mode,
            multi_task_mode=multi_task_mode,
            shape=shape,
            n_features=n_features,
            device=device,
            name=name,
            seed=seed
        )

        self.channels: Sequence[int] = literal_eval(channels) if isinstance(channels, str) else channels
        self.strides = strides if strides else [2] * (len(channels) - 1)
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.activation = activation
        self.norm = norm
        self.dropout_cnn = dropout_cnn
        self.dropout_fnn = dropout_fnn
        self.partly_shared_convolutions = partly_shared_convolutions

        if hidden_channels_fnn:
            self.hidden_channels_fnn = hidden_channels_fnn
        else:
            self.hidden_channels_fnn = (int(sum(self.channels)/4), int(sum(self.channels)/16))

        self.partly_shared_conv_final_shape = None

    def _get_layer(
            self,
            in_channels: int,
            out_channels: int,
            strides: int,
            is_last: bool = False
    ) -> Union[ResidualUnit, Convolution]:
        """
        Returns a convolutional layer. If the number of residual units is greater than 0, a residual unit is returned.
        Otherwise, a convolutional layer is returned. If the layer is the last one, the activation is not applied.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        strides : int
            Stride of the convolution.
        is_last : bool
            Whether the layer is the last one.
        """
        if self.num_res_units > 0:
            return ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=len(self.shape),
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.activation,
                norm=self.norm,
                dropout=self.dropout_cnn
            )
        else:
            return Convolution(
                conv_only=is_last,
                spatial_dims=len(self.shape),
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.activation,
                norm=self.norm,
                dropout=self.dropout_cnn
            )

    def _build_partly_shared_extractor(self) -> Sequential:
        """
        Builds a partly shared extractor model. The first layers are shared between the tasks. The last layers are
        specific to each task. The number of shared layers is defined by the 'PARTLY_SHARED_CONVOLUTIONS' class
        attribute.

        Returns
        -------
        shared_extractor : Sequential
            The shared extractor model.
        """
        conv_sequence = Sequential()
        partly_shared_final_shape = copy(self.shape)
        for i in range(self.partly_shared_convolutions):
            layer = self._get_layer(
                in_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                out_channels=self.channels[i],
                strides=self.strides[i]
            )
            conv_sequence.add_module(
                name="shared_conv_%i" % i,
                module=DataParallel(layer).to(self.device)
            )
            partly_shared_final_shape = tuple(int(t/self.strides[i]) for t in partly_shared_final_shape)

        self.partly_shared_final_shape = (
            int(self.channels[self.partly_shared_convolutions - 1]),
            *partly_shared_final_shape
        )

        return conv_sequence

    def __get_single_conv_sequence(self):
        """
        Returns a single convolutional sequence.

        Returns
        -------
        conv_sequence : Sequential
            The convolutional sequence.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            in_shape = self.partly_shared_final_shape
            channels = self.channels[self.partly_shared_convolutions:]
            strides = self.strides[self.partly_shared_convolutions:]
        else:
            in_shape = self.in_shape
            channels = self.channels
            strides = self.strides

        conv_sequence = Sequential()
        for i, c in enumerate(channels):
            layer = self._get_layer(
                in_channels=in_shape[0] if i == 0 else channels[i - 1],
                out_channels=c,
                strides=1 if i == len(channels) - 1 else strides[i],
                is_last=i == len(channels) - 1
            )
            conv_sequence.add_module(
                name="conv_%i" % i,
                module=DataParallel(layer).to(self.device)
            )

        return conv_sequence

    def __get_single_linear_module(self):
        """
        Returns a single linear module.

        Returns
        -------
        linear_module : Module
            The linear module.
        """
        linear_module = FullyConnectedNet(
            in_channels=sum(self.channels),
            out_channels=self.n_features,
            hidden_channels=self.hidden_channels_fnn,
            dropout=self.dropout_fnn,
            act=self.activation
        )
        return DataParallel(linear_module).to(self.device)

    def _build_single_extractor(self) -> Module:
        """
        Returns a single extractor module.

        Returns
        -------
        extractor : Module
            The extractor module. It should take as input a tensor of shape (batch_size, channels, *spatial_shape) and
            return a tensor of shape (batch_size, n_features, *spatial_shape).
        """
        conv_sequence = self.__get_single_conv_sequence()
        linear_module = self.__get_single_linear_module()

        return _Encoder(conv_sequence=conv_sequence, linear_module=linear_module)

    def _build_extractor(self, dataset: ProstateCancerDataset) -> Union[Module, ModuleDict]:
        """
        Returns the extractor module.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used to build the extractor.

        Returns
        -------
        extractor : Union[Module, ModuleDict]
            The extractor module. It should take as input a tensor of shape (batch_size, channels, *spatial_shape) and
            return a tensor of shape (batch_size, n_features, *spatial_shape).
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            return ModuleDict({task.name: self._build_single_extractor() for task in self._tasks.table_tasks})
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self._build_single_extractor()
        elif self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            shared_module = self._build_partly_shared_extractor()
            separated_modules = ModuleDict(
                {task.name: self._build_single_extractor() for task in self._tasks.table_tasks}
            )

            partly_shared_module_dict = ModuleDict()
            for name, module in separated_modules.items():
                partly_shared_module_dict[name] = Sequential(shared_module, module)

            return partly_shared_module_dict
