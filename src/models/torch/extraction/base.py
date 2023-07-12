"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     04/2022
    @Last modification: 07/2023

    @Description:       This file is used to define an abstract 'Extractor' model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from ast import literal_eval
from enum import auto, StrEnum
from typing import Dict, List, Optional, NamedTuple, Sequence, Union

from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization
from torch import cat, Tensor
from torch import device as torch_device
from torch.nn import DataParallel, Linear, Module, ModuleDict

from ..base import check_if_built, TorchModel
from ..blocks import BayesianFullyConnectedNet, FullyConnectedNet
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ....tasks import SegmentationTask


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
    This class is used to define the multitask mode of the model. It can be either partly shared, fully shared.

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


class ExtractorKLDivergence(NamedTuple):
    """
    This class is used to define the KL divergence of the extractor model. It contains the KL divergence associated to
    the extraction of deep features and the KL divergence associated to the segmentation of the images (optional).

    Elements
    --------
    deep_features : Tensor
        The KL divergence associated to the extraction of deep features.
    segmentation : Optional[Tensor]
        The KL divergence associated to the segmentation of the images. This is optional.
    """
    deep_features: Tensor
    segmentation: Optional[Tensor] = None


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
            shape: Sequence[int] = (128, 128, 128),
            n_features: int = 6,
            activation: str = "PRELU",
            channels: Union[str, Sequence[int]] = (64, 128, 256, 512, 1024),
            dropout_fnn: float = 0.0,
            hidden_channels_fnn: Optional[Sequence[int]] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
            bayesian: bool = False
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
        shape : Sequence[int]
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
        bayesian : bool
            Whether the model implements variational inference.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.activation = activation
        self.channels: Sequence[int] = literal_eval(channels) if isinstance(channels, str) else channels
        self.dropout_fnn = dropout_fnn
        self.image_keys = image_keys if isinstance(image_keys, Sequence) else [image_keys]
        self.model_mode = ModelMode(model_mode)
        self.multi_task_mode = MultiTaskMode(multi_task_mode)
        self.n_features = n_features
        self.shape = shape
        self._bayesian = bayesian

        if hidden_channels_fnn:
            self.hidden_channels_fnn = hidden_channels_fnn
        else:
            self.hidden_channels_fnn = (int(sum(self.channels)/4), int(sum(self.channels)/16))

        self.extractor: Optional[Module] = None
        self.linear_module: Optional[Union[Module, ModuleDict]] = None
        self.prediction_layer: Optional[Union[Module, ModuleDict]] = None

        self.kl_divergence: Optional[Dict[str, Tensor]] = None

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
            images and the segmentation of the images (optional). If the model is in bayesian mode, the extractor module
            should return a tuple of an ExtractorOutput object and an ExtractorKLDivergence object.
        """
        raise NotImplementedError

    def _get_single_linear_module(self) -> Module:
        """
        Returns a single linear module.

        Returns
        -------
        linear_module : Module
            The linear module.
        """
        if self.bayesian:
            linear_module = BayesianFullyConnectedNet(
                in_channels=sum(self.channels),
                out_channels=self.n_features,
                hidden_channels=self.hidden_channels_fnn,
                dropout=self.dropout_fnn,
                act=self.activation
            )

        else:
            linear_module = FullyConnectedNet(
                in_channels=sum(self.channels),
                out_channels=self.n_features,
                hidden_channels=self.hidden_channels_fnn,
                dropout=self.dropout_fnn,
                act=self.activation
            )

        return DataParallel(linear_module).to(self.device)

    def _build_linear_module(self) -> Union[Module, ModuleDict]:
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

    def _get_single_prediction_layer(
            self,
            in_features: int,
            out_features: int
    ) -> Module:
        """
        Returns a single linear module.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.

        Returns
        -------
        single_prediction_layer : Module
            The Linear Module used for prediction.
        """
        if self.bayesian:
            return DataParallel(
                LinearReparameterization(
                    in_features=in_features,
                    out_features=out_features,
                    prior_mean=0.0,
                    prior_variance=0.1
                )
            ).to(self.device)

        else:
            return DataParallel(Linear(in_features=in_features, out_features=out_features)).to(self.device)

    def _build_prediction_layer(self) -> Union[Module, ModuleDict]:
        """
        Returns the prediction layer module.

        Returns
        -------
        prediction_layer : Union[Module, ModuleDict]
            The prediction layer module. It should take as input a tensor of shape (batch_size, n_features) and return a
            tensor of shape (batch_size, n_tasks) or a tuple of said tensor and the KL divergence if the model is in
            bayesian mode.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            return ModuleDict(
                {task.name: self._get_single_prediction_layer(in_features=self.n_features, out_features=1)
                 for task in self._tasks.table_tasks}
            )

        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self._get_single_prediction_layer(
                in_features=self.n_features,
                out_features=len(self._tasks.table_tasks)
            )

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

    def _get_radiomics(self, deep_features: Tensor) -> Union[Tensor, Dict[str, Tensor], tuple]:
        """
        Returns the reduced deep features, i.e. the radiomics.

        Parameters
        ----------
        deep_features : Tensor
            The deep features extracted from the images.

        Returns
        -------
        reduced_features : Union[Tensor, Dict[str, Tensor], tuple]
            The reduced deep features. If the model is in bayesian mode, a tuple of the reduced deep features and their
            respective KL divergence is returned.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            if self.bayesian:
                reduced_deep_features = {}
                radiomics_kl = {}
                for t in self._tasks.table_tasks:
                    reduced_deep_features[t.name], radiomics_kl[t.name] = self.linear_module[t.name](deep_features)

                return reduced_deep_features, radiomics_kl

            else:
                return {task.name: self.linear_module[task.name](deep_features) for task in self._tasks.table_tasks}

        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            if self.bayesian:
                reduced_deep_features, kl = self.linear_module(deep_features)
                radiomics_kl = {task.name: kl for task in self._tasks.table_tasks}
                return reduced_deep_features, radiomics_kl

            else:
                return self.linear_module(deep_features)

        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    def _get_prediction(self, radiomics: Union[Tensor, Dict[str, Tensor]]) -> Union[Dict[str, Tensor], tuple]:
        """
        Returns the prediction.

        Parameters
        ----------
        radiomics : Union[Tensor, Dict[str, Tensor]]
            The radiomics features.

        Returns
        -------
        prediction : Union[Dict[str, Tensor], tuple]
            The prediction. If the model is in bayesian mode, a tuple of the prediction and its respective KL divergence
            is returned.
        """
        if self.multi_task_mode == MultiTaskMode.PARTLY_SHARED:
            if self.bayesian:
                prediction = {}
                prediction_kl = {}
                for task in self._tasks.table_tasks:
                    y, kl = self.prediction_layer[task.name](radiomics[task.name])
                    prediction[task.name] = y[:, 0]
                    prediction_kl[task.name] = kl

                return prediction, prediction_kl

            else:
                return {t.name: self.prediction_layer[t.name](radiomics[t.name])[:, 0] for t in self._tasks.table_tasks}

        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            if self.bayesian:
                y, kl = self.prediction_layer(radiomics)
                prediction = {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}
                prediction_kl = {task.name: kl for task in self._tasks.table_tasks}
                return prediction, prediction_kl

            else:
                y = self.prediction_layer(radiomics)
                return {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        else:
            raise ValueError(f"{self.multi_task_mode} is not a valid MultiTaskMode")

    def __get_kl_dict(
            self,
            extractor_kl: ExtractorKLDivergence,
            radiomics_kl: Dict[str, Tensor],
            prediction_kl: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        """
        Loops through the tasks and creates a dictionary of the KL divergence per task.

        Parameters
        ----------
        extractor_kl : ExtractorKLDivergence
            The KL divergence of the extraction process.
        radiomics_kl : Dict[str, Tensor]
            The KL divergence associated to getting the radiomics.
        prediction_kl : Optional[Dict[str, Tensor]]
            The KL divergence of the prediction process. If the model is in extraction mode then prediction_kl is None.

        Returns
        -------
        kl_divergence : Dict[str, Tensor]
            The KL divergence of each task.
        """
        kl_divergence = {}
        for t in self._tasks:
            if isinstance(t, SegmentationTask):
                kl_divergence[t.name] = extractor_kl.segmentation

            else:
                if prediction_kl is not None:   # Prediction mode.
                    kl_divergence[t.name] = extractor_kl.deep_features + radiomics_kl[t.name] + prediction_kl[t.name]
                else:                           # Extraction mode.
                    kl_divergence[t.name] = extractor_kl.deep_features + radiomics_kl[t.name]

        return kl_divergence

    def _bayesian_forward(self, input_tensor: Tensor) -> Union[ExtractorOutput, TargetsType]:
        """
        Performs a variational inference forward pass through the model.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        targets : Union[ExtractorOutput, TargetsType]
            The output targets. If the model is in 'extraction' mode, it will return an ExtractorOutput. If the model
            is in 'prediction' mode, it will return a dictionary of tensors, one for each task.
        """
        extractor_output, extractor_kl = self.extractor(input_tensor)

        radiomics, radiomics_kl = self._get_radiomics(extractor_output.deep_features)
        output = ExtractorOutput(
            deep_features=radiomics,
            segmentation=extractor_output.segmentation
        )

        if self.model_mode == ModelMode.EXTRACTION:
            kl_divergence = self.__get_kl_dict(
                extractor_kl=extractor_kl,
                radiomics_kl=radiomics_kl,
                prediction_kl=None
            )
            self.kl_divergence = kl_divergence
            return output

        elif self.model_mode == ModelMode.PREDICTION:
            tab_dict, prediction_kl = self._get_prediction(radiomics)

            kl_divergence = self.__get_kl_dict(
                extractor_kl=extractor_kl,
                radiomics_kl=radiomics_kl,
                prediction_kl=prediction_kl
            )
            self.kl_divergence = kl_divergence

            seg = output.segmentation

            if seg is not None:
                seg_dict = {task.name: seg[:, i, None] for i, task in enumerate(self._tasks.segmentation_tasks)}
                return tab_dict | seg_dict
            else:
                return tab_dict

    def _deterministic_forward(self, input_tensor: Tensor) -> Union[ExtractorOutput, TargetsType]:
        """
        Performs a deterministic forward pass through the model.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        targets : Union[ExtractorOutput, TargetsType]
            The output targets. If the model is in 'extraction' mode, it will return an ExtractorOutput. If the model
            is in 'prediction' mode, it will return a dictionary of tensors, one for each task.
        """
        extractor_output = self.extractor(input_tensor)

        radiomics = self._get_radiomics(extractor_output.deep_features)
        output = ExtractorOutput(
            deep_features=radiomics,
            segmentation=extractor_output.segmentation
        )

        if self.model_mode == ModelMode.EXTRACTION:
            return output
        elif self.model_mode == ModelMode.PREDICTION:
            tab_dict = self._get_prediction(radiomics)

            seg = output.segmentation

            if seg is not None:
                seg_dict = {task.name: seg[:, i, None] for i, task in enumerate(self._tasks.segmentation_tasks)}
                return tab_dict | seg_dict
            else:
                return tab_dict

    @check_if_built
    def forward(self, features: FeaturesType) -> Union[ExtractorOutput, TargetsType]:
        """
        Performs a forward pass through the model. Applies different methods depending on whether the model is bayesian.

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

        if self.bayesian:
            return self._bayesian_forward(x_image)

        else:
            return self._deterministic_forward(x_image)
