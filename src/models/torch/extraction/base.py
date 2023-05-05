"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     04/2022
    @Last modification: 04/2023

    @Description:       This file is used to define an abstract 'Extractor' model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from ast import literal_eval
from enum import auto, StrEnum
from typing import Dict, List, Optional, Sequence, Union

from torch import cat, Tensor
from torch import device as torch_device
from torch.nn import Linear, Module, ModuleDict

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ....tasks import SegmentationTask


class MergingMethod(StrEnum):
    """
    This class is used to define the merging method used to merge the extracted radiomics. It can be either
    concatenation or multiplication.

    Elements
    --------
    CONCATENATION : str
        Concatenation merging method.
    MULTIPLICATION : str
        Multiplication merging method.
    """
    CONCATENATION = auto()
    MULTIPLICATION = auto()


class ModelMode(StrEnum):
    """
    This class is used to define the mode of the model. It can be either extraction or prediction. If 'extraction', the
    model will extract deep radiomics from the images. If 'prediction', the model will perform predictions using the
    extracted deep radiomics.

    Elements
    --------
    EXTRACTION : str
        Extraction mode.
    PREDICTION : str
        Prediction mode.
    """
    EXTRACTION = auto()
    PREDICTION = auto()


class MultiTaskMode(StrEnum):
    """
    This class is used to define the multi-task mode of the model. It can be either separated, partly shared or fully
    shared.

    Elements
    --------
    SEPARATED : str
        A separate extractor model is used for each task.
    PARTLY_SHARED : str
        A partly shared extractor model is used. The first layers are shared between the tasks.
    FULLY_SHARED : str
        A fully shared extractor model is used. All layers are shared between the tasks.
    """
    SEPARATED = auto()
    PARTLY_SHARED = auto()
    FULLY_SHARED = auto()


class Extractor(TorchModel, ABC):
    """
    This class is used to define an abstract 'Extractor' model. It can be used to extract deep radiomics from images or
    to perform predictions using the extracted deep radiomics.
    """

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            segmentation_key_or_task: Optional[str, SegmentationTask] = None,
            merging_method: Union[str, MergingMethod] = MergingMethod.CONCATENATION,
            model_mode: Union[str, ModelMode] = ModelMode.PREDICTION,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Union[str, Sequence[int]] = (96, 96, 96),
            n_features: int = 5,
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
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(device=device, name=name, seed=seed)

        if isinstance(segmentation_key_or_task, SegmentationTask):
            self.segmentation_key = segmentation_key_or_task.name
        else:
            self.segmentation_key = segmentation_key_or_task

        self.shape = literal_eval(shape) if isinstance(shape, str) else shape
        self.image_keys = image_keys if isinstance(image_keys, Sequence) else [image_keys]
        self.merging_method = MergingMethod(merging_method)
        self.model_mode = ModelMode(model_mode)
        self.multi_task_mode = MultiTaskMode(multi_task_mode)
        self.n_features = n_features

        self.extractor = None
        self.linear_layer = None

    @abstractmethod
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
        raise NotImplementedError

    @property
    def in_shape(self) -> Sequence[int]:
        """
        Returns the shape of the input tensor expected by the model, excluding batch dimension.

        Returns
        -------
        Sequence[int]
            The shape of the input tensor expected by the model, excluding batch dimension.
        """
        n_images = len(self.image_keys)
        if self.segmentation_key:
            if self.merging_method == MergingMethod.CONCATENATION:
                in_channel = n_images + 1
            elif self.merging_method == MergingMethod.MULTIPLICATION:
                in_channel = n_images
            else:
                raise ValueError(f"{self.merging_method} is not a valid MergingMethod")
        else:
            in_channel = n_images

        return in_channel, *self.shape

    def _build_linear_layer(self) -> Union[Module, ModuleDict]:
        """
        Returns the linear layer module.

        Returns
        -------
        linear_layer : Union[Module, ModuleDict]
            The linear layer module. It should take as input a tensor of shape (batch_size, n_features, *spatial_shape)
            and return a tensor of shape (batch_size, n_features).
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED or self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            return ModuleDict({
                task.name: Linear(in_features=self.n_features, out_features=1).to(self.device)
                for task in self._tasks.table_tasks
            })
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return Linear(in_features=self.n_features, out_features=len(self._tasks.table_tasks)).to(self.device)
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    def build(self, dataset: ProstateCancerDataset) -> Extractor:
        """
        Builds the model. This method must be called before training the model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset to build the model for.

        Returns
        -------
        self : Extractor
            The built model.
        """
        super().build(dataset=dataset)

        self.extractor = self._build_extractor(dataset)
        self.linear_layer = self._build_linear_layer()

        return self

    def _get_input_tensor(self, features: FeaturesType) -> Tensor:
        """
        Returns the input tensor to the extractor.

        Parameters
        ----------
        features : FeaturesType
            The features to use as input to the extractor.

        Returns
        -------
        input: Tensor
            The input tensor to the extractor.
        """
        if self.segmentation_key:
            if self.merging_method == MergingMethod.CONCATENATION:
                image_and_seg_keys = self.image_keys + [self.segmentation_key]
                return cat([features.image[k] for k in image_and_seg_keys], 1)
            elif self.merging_method == MergingMethod.MULTIPLICATION:
                return cat([features.image[k]*features.image[self.segmentation_key] for k in self.image_keys], 1)
            else:
                raise ValueError(f"{self.merging_method} is not a valid MergingMethod")
        else:
            return cat([features.image[k] for k in self.image_keys], 1)

    def _get_radiomics(self, input_tensor: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Returns the radiomics features.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor to the extractor.

        Returns
        -------
        radiomics : Union[Tensor, Dict[str, Tensor]]
            The radiomics features.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED or self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            deep_radiomics = {}
            for task in self._tasks.table_tasks:
                deep_radiomics[task.name] = self.extractor[task.name](input_tensor)
            return deep_radiomics
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self.extractor(input_tensor)
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    def _get_prediction(self, radiomics: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Returns the prediction.

        Parameters
        ----------
        radiomics : Union[Tensor, Dict[str, Tensor]]
            The radiomics features.

        Returns
        -------
        prediction : Dict[str, Tensor]
            The prediction.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED or self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            prediction = {}
            for task in self._tasks.table_tasks:
                prediction[task.name] = self.linear_layer[task.name](radiomics[task.name])[:, 0]
            return prediction
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            y = self.linear_layer(radiomics)
            return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> Union[Tensor, TargetsType]:
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        features : FeaturesType
            The input features.

        Returns
        -------
        targets : Union[Tensor, TargetsType]
            The output targets. If the model is in 'extraction' mode, it will return a tensor of shape
            (batch_size, n_features). If the model is in 'prediction' mode, it will return a dictionary of tensors,
            each of shape (batch_size, 1).
        """
        x_image = self._get_input_tensor(features)
        deep_radiomics = self._get_radiomics(x_image)

        if self.model_mode == ModelMode.EXTRACTION:
            return deep_radiomics
        elif self.model_mode == ModelMode.PREDICTION:
            return self._get_prediction(deep_radiomics)
        else:
            raise ValueError(f"{self.model_mode} is not a valid ModelMode")
