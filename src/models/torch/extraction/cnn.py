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
from monai.networks.nets import Classifier
from torch import device as torch_device
from torch.nn import Module, ModuleDict, Sequential

from .base import Extractor, MergingMethod, ModelMode, MultiTaskMode
from ....tasks import SegmentationTask
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class CNN(Extractor):
    """
    A convolutional neural network used to extract deep radiomics from 3D medical images. It can also be used to perform
    predictions using extracted radiomics. The model is based on the 'Classifier' model from MONAI.
    """

    PARTLY_SHARED_CONVOLUTIONS = 2

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            segmentation_key_or_task: Optional[str, SegmentationTask] = None,
            merging_method: Union[str, MergingMethod] = MergingMethod.CONCATENATION,
            model_mode: Union[str, ModelMode] = ModelMode.PREDICTION,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Union[str, Sequence[int]] = (96, 96, 96),
            n_features: int = 5,
            channels: Union[str, Sequence[int]] = (4, 8, 16, 32, 64),
            strides: Optional[Sequence[int]] = None,
            kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 2,
            activation: str = "PRELU",
            norm: str = "INSTANCE",
            dropout: float = 0.0,
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
            Integer stating the dimension of the final output tensor, i.e. the number of deep radiomics/features to
            extract from the image.
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
        dropout : float
            Probability of dropout.
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
        self.dropout = dropout
        self.partly_shared_conv_final_shape = None

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
        for i in range(self.PARTLY_SHARED_CONVOLUTIONS):
            if self.num_res_units > 0:
                module = ResidualUnit(
                    subunits=self.num_res_units,
                    spatial_dims=len(self.shape),
                    in_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                    out_channels=self.channels[i],
                    strides=self.strides[i],
                    kernel_size=self.kernel_size,
                    act=self.activation,
                    norm=self.norm,
                    dropout=self.dropout
                ).to(self.device)
            else:
                module = Convolution(
                    spatial_dims=len(self.shape),
                    in_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                    out_channels=self.channels[i],
                    strides=self.strides[i],
                    kernel_size=self.kernel_size,
                    act=self.activation,
                    norm=self.norm,
                    dropout=self.dropout
                ).to(self.device)

            conv_sequence.add_module(
                name="shared_conv_%i" % i,
                module=module
            )

            partly_shared_final_shape = tuple(int(t/self.strides[i]) for t in partly_shared_final_shape)

        self.partly_shared_final_shape = (
            int(self.channels[self.PARTLY_SHARED_CONVOLUTIONS - 1]),
            *partly_shared_final_shape
        )

        return conv_sequence

    def _build_single_extractor(self) -> Module:
        """
        Returns a single extractor module.

        Returns
        -------
        extractor : Module
            The extractor module. It should take as input a tensor of shape (batch_size, channels, *spatial_shape) and
            return a tensor of shape (batch_size, n_features, *spatial_shape).
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            in_shape = self.partly_shared_final_shape
            channels = self.channels[self.PARTLY_SHARED_CONVOLUTIONS:]
            strides = self.strides[self.PARTLY_SHARED_CONVOLUTIONS:]
        else:
            in_shape = self.in_shape
            channels = self.channels
            strides = self.strides

        return Classifier(
            in_shape=in_shape,
            classes=self.n_features,
            channels=channels,
            strides=strides,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.activation,
            norm=self.norm,
            dropout=self.dropout
        ).to(self.device)

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
