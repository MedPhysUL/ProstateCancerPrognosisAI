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
from typing import Dict, List, Optional, NamedTuple, Sequence, Union

from monai.networks.nets import FullyConnectedNet
from torch import cat, Tensor
from torch import device as torch_device
from torch.nn import DataParallel, Linear, Module, ModuleDict

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType


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
    This class is used to define the multi-task mode of the model. It can be either partly shared, fully shared.

    Elements
    --------
    PARTLY_SHARED : str
        A shared extractor model is used to get a large tensor of radiomics from the images. Then, a separate linear
        model is used for each task to obtain the final (reduced) deep features.
    FULLY_SHARED : str
        A shared extractor model is used to get a large tensor of radiomics from the images. Then, a shared linear
        model is used for each task to obtain the final (reduced) deep features.
    """
    PARTLY_SHARED = auto()
    FULLY_SHARED = auto()


class ExtractorOutput(NamedTuple):
    """
    This class is used to define the output of the extractor model. It contains the deep features extracted from the
    images and the segmentation of the images (optional).

    Elements
    --------
    deep_features : Union[Tensor, Dict[str, Tensor]]
        The deep features extracted from the images.
    segmentation : Optional[Tensor]
        The segmentation of the images. This is optional.
    """
    deep_features: Union[Tensor, Dict[str, Tensor]]
    segmentation: Optional[Tensor] = None


class Extractor(TorchModel, ABC):
    """
    This class is used to define an abstract 'Extractor' model. It can be used to extract deep radiomics from images or
    to perform predictions using the extracted deep radiomics.
    """

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            model_mode: Union[str, ModelMode] = ModelMode.PREDICTION,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Union[str, Sequence[int]] = (128, 128, 128),
            n_features: int = 6,
            activation: str = "PRELU",
            channels: Union[str, Sequence[int]] = (64, 128, 256, 512, 1024),
            dropout_fnn: float = 0.0,
            hidden_channels_fnn: Optional[Sequence[int]] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
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
        channels : Union[str, Sequence[int]]
            Sequence of integers stating the output channels of each convolutional layer. Can also be given as a string
            containing the sequence. Default to (64, 128, 256, 512, 1024).
        shape : Union[str, Sequence[int]]
            Sequence of integers stating the dimension of the input tensor (minus batch and channel dimensions). Can
            also be given as a string containing the sequence. Default to (128, 128, 128).
        n_features : int
            Integer stating the dimension of the final output tensor, i.e. the number of deep features to extract from
            the image.
        activation : str
             Name defining activation layers.
        dropout_fnn : float
            Dropout rate after each fully connected layers.
        hidden_channels_fnn : Optional[Sequence[int]]
            Sequence of integers stating the number of hidden units in each fully connected layer.
        device : Optional[torch_device]
            The device of the model.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.activation = activation
        self.channels: Sequence[int] = literal_eval(channels) if isinstance(channels, str) else channels
        self.dropout_fnn = dropout_fnn
        self.image_keys = image_keys if isinstance(image_keys, Sequence) else [image_keys]
        self.model_mode = ModelMode(model_mode)
        self.multi_task_mode = MultiTaskMode(multi_task_mode)
        self.n_features = n_features
        self.shape = literal_eval(shape) if isinstance(shape, str) else shape

        if hidden_channels_fnn:
            self.hidden_channels_fnn = hidden_channels_fnn
        else:
            self.hidden_channels_fnn = (int(sum(self.channels)/4), int(sum(self.channels)/16))

        self.extractor = None
        self.linear_module = None
        self.prediction_layer = None

    @property
    def in_shape(self) -> Sequence[int]:
        """
        Returns the shape of the input tensor expected by the model, excluding batch dimension.

        Returns
        -------
        Sequence[int]
            The shape of the input tensor expected by the model, excluding batch dimension.
        """
        return len(self.image_keys), *self.shape

    @abstractmethod
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
        raise NotImplementedError

    def _get_single_linear_module(self):
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

    def _build_linear_module(self):
        """
        Returns the linear module. If the model is in extraction mode, the linear module will be a single module. If the
        model is in prediction mode, the linear module will be a ModuleDict containing a separate module for each task.

        Returns
        -------
        linear_module : Union[Module, ModuleDict]
            The linear module.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            return ModuleDict(
                {task.name: self._get_single_linear_module() for task in self._tasks.table_tasks}
            )
        else:
            return self._get_single_linear_module()

    def _build_prediction_layer(self) -> Union[Module, ModuleDict]:
        """
        Returns the prediction layer module.

        Returns
        -------
        prediction_layer : Union[Module, ModuleDict]
            The prediction layer module. It should take as input a tensor of shape (batch_size, n_features) and return a
            tensor of shape (batch_size, n_tasks).
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            return ModuleDict({
                task.name: DataParallel(Linear(in_features=self.n_features, out_features=1)).to(self.device)
                for task in self._tasks.table_tasks
            })
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return DataParallel(
                Linear(in_features=self.n_features, out_features=len(self._tasks.table_tasks))
            ).to(self.device)
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

        self.extractor = self._build_deep_features_extractor(dataset)
        self.linear_module = self._build_linear_module()

        if self.model_mode == ModelMode.PREDICTION:
            self.prediction_layer = self._build_prediction_layer()

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
        return cat([features.image[k] for k in self.image_keys], 1)

    def _get_radiomics(self, deep_features: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Returns the reduced deep features, i.e. the radiomics.

        Parameters
        ----------
        deep_features : Tensor
            The deep features extracted from the images.

        Returns
        -------
        reduced_features : Union[Tensor, Dict[str, Tensor]]
            The reduced deep features.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            reduced_deep_features = {}
            for task in self._tasks.table_tasks:
                reduced_deep_features[task.name] = self.linear_module[task.name](deep_features)
            return reduced_deep_features
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self.linear_module(deep_features)
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
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            prediction = {}
            for task in self._tasks.table_tasks:
                prediction[task.name] = self.prediction_layer[task.name](radiomics[task.name])[:, 0]
            return prediction
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            y = self.prediction_layer(radiomics)
            return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> Union[ExtractorOutput, TargetsType]:
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        features : FeaturesType
            The input features.

        Returns
        -------
        targets : Union[ExtractorOutput, TargetsType]
            The output targets. If the model is in 'extraction' mode, it will return an ExtractorOutput. If the model
            is in 'prediction' mode, it will return a dictionary of tensors, one for each task.
        """
        x_image = self._get_input_tensor(features)
        extractor_output = self.extractor(x_image)

        radiomics = self._get_radiomics(extractor_output.deep_features)
        output = ExtractorOutput(deep_features=radiomics, segmentation=extractor_output.segmentation)

        if self.model_mode == ModelMode.EXTRACTION:
            return output
        elif self.model_mode == ModelMode.PREDICTION:
            tab_dict = self._get_prediction(radiomics)

            segmentation = output.segmentation
            if segmentation:
                seg_dict = {task.name: segmentation[:, i] for i, task in enumerate(self._tasks.segmentation_tasks)}
                return tab_dict | seg_dict
            else:
                return tab_dict
