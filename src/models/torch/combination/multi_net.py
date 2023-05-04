"""
    @file:              multi_net.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define an 'MultiNet' model.
"""

from __future__ import annotations
from enum import auto
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Union

from strenum import LowercaseStrEnum
from torch import device as torch_device
from torch import sigmoid, Tensor

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..extraction.base import Extractor, ModelMode
from ..prediction.base import InputMode, Predictor
from ..segmentation.base import Segmentor


class SegmentationMap(LowercaseStrEnum):
    """
    Enum for the segmentation map type.

    Elements
    --------
    LABEL_MAP : str
        The segmentation map is a label map.
    PROBABILITY_MAP : str
        The segmentation map is a probability map.
    """
    LABEL_MAP = auto()
    PROBABILITY_MAP = auto()


class ModelSetup(NamedTuple):
    """
    NamedTuple for the model setup.

    Elements
    --------
    model : Union[Extractor, Predictor, Segmentor]
        The model.
    freeze : bool
        Whether to freeze the model.
    pretrained_model_state : Optional[Mapping[str, Any]]
        The pretrained model state.
    """
    model: Union[Extractor, Predictor, Segmentor]
    freeze: bool = False
    pretrained_model_state: Optional[Mapping[str, Any]] = None


class MultiNet(TorchModel):
    """
    Class for the MultiNet model. A MultiNet model is a model that can perform prediction, extraction and segmentation.
    It can also perform multiple of these tasks at the same time.
    """

    def __init__(
            self,
            predictor_setup: Optional[ModelSetup] = None,
            extractor_setup: Optional[ModelSetup] = None,
            segmentor_setup: Optional[ModelSetup] = None,
            segmentation_map: Union[str, SegmentationMap] = SegmentationMap.PROBABILITY_MAP,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        predictor_setup : Optional[ModelSetup]
            The setup for the predictor model. If None, will not use a predictor model.
        extractor_setup : Optional[ModelSetup]
            The setup for the extractor model. If None, will not use an extractor model.
        segmentor_setup : Optional[ModelSetup]
            The setup for the segmentor model. If None, will not use a segmentor model.
        segmentation_map : Union[str, SegmentationMap]
            The segmentation map type. Can be either 'label_map' or 'probability_map'.
        device : Optional[torch_device]
            The device to use for the model.
        name : Optional[str]
            The name of the model.
        seed: Optional[int]
            The seed to use for the model.
        """
        super().__init__(device=device, name=name, seed=seed)

        self.predictor = predictor_setup.model if predictor_setup else None
        self.extractor = extractor_setup.model if extractor_setup else None
        self.segmentor = segmentor_setup.model if segmentor_setup else None
        self.segmentation_map = SegmentationMap(segmentation_map)

        self._setups = [predictor_setup, extractor_setup, segmentor_setup]
        self._validate_model_devices()
        self._validate_seed()

        self._prepare_models()
        self._forward_pass = self._get_forward_pass()

    @property
    def _map_model_combination_to_forward_pass(self) -> Mapping[tuple, Callable]:
        """
        Returns a mapping from a model combination to the corresponding forward pass function.

        Returns
        -------
        mapping : Mapping[tuple, Callable]
            The mapping from a model combination to the corresponding forward pass function.
        """
        return {
            (True, False, False): self._forward_pass_predictor,
            (False, True, False): self._forward_pass_extractor,
            (False, False, True): self._forward_pass_segmentor,
            (True, True, False): self._forward_pass_extractor_predictor,
            (True, False, True): self._forward_pass_segmentor_predictor,
            (False, True, True): self._forward_pass_segmentor_extractor,
            (True, True, True): self._forward_pass_segmentor_extractor_predictor
        }

    def _prepare_models(self):
        """
        Prepares the models for training.
        """
        if self.predictor and self.extractor:
            self.extractor.model_mode = ModelMode.EXTRACTION
        if self.predictor and (self.extractor or self.segmentor):
            if self.predictor.input_mode == InputMode.TABULAR:
                self.predictor.input_mode = InputMode.HYBRID
        if not self.predictor and (self.extractor and self.segmentor):
            self.extractor.model_mode = ModelMode.PREDICTION

    def _get_forward_pass(self) -> Callable:
        """
        Returns the forward pass function corresponding to the model combination.

        Returns
        -------
        forward_pass : Callable
            The forward pass function corresponding to the model combination.
        """
        combination = (
            True if self.predictor else False,
            True if self.extractor else False,
            True if self.segmentor else False
        )

        return self._map_model_combination_to_forward_pass[combination]

    def _forward_pass_predictor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the predictor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        return self.predictor(features)

    def _forward_pass_extractor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the extractor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        return self.extractor(features)

    def _forward_pass_segmentor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the segmentor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        return self.segmentor(features)

    def _forward_pass_extractor_predictor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the extractor and predictor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        radiomics = self.extractor(features)
        prediction = self.predictor(features, radiomics)
        return prediction

    def _forward_pass_segmentor_predictor(self, features: FeaturesType) -> TargetsType:
        raise NotImplementedError(
            "This particular model needs a segmentor that can provide bottleneck radiomics features. Thus, it is "
            "necessary to modify the unet model to accommodate this requirement."
        )

    def _forward_pass_segmentor_extractor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the segmentor and extractor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        segmentation_dict = self.segmentor(features)
        segmentation_key = self.extractor.segmentation_key

        features = self._get_updated_features_with_segmentation(features, segmentation_dict, segmentation_key)
        table_prediction = self.extractor(features)

        return table_prediction | segmentation_dict

    def _get_updated_features_with_segmentation(
            self,
            features: FeaturesType,
            segmentation_dict: Dict[str, Tensor],
            segmentation_key: str
    ) -> FeaturesType:
        """
        Updates the features with the segmentation.

        Parameters
        ----------
        features : FeaturesType
            The features.
        segmentation_dict : Dict[str, Tensor]
            The segmentation dictionary.
        segmentation_key : str
            The segmentation key.

        Returns
        -------
        features : FeaturesType
            The updated features.
        """
        if self.segmentation_map == SegmentationMap.LABEL_MAP:
            if self._setups[2].freeze:
                features.image[segmentation_key] = segmentation_dict[segmentation_key]
            else:
                features.image[segmentation_key] = segmentation_dict[segmentation_key].detach()
        elif self.segmentation_map == SegmentationMap.PROBABILITY_MAP:
            features.image[segmentation_key] = sigmoid(segmentation_dict[segmentation_key])
        else:
            raise ValueError(f"Invalid segmentation map: {self.segmentation_map}")

        return features

    def _forward_pass_segmentor_extractor_predictor(self, features: FeaturesType) -> TargetsType:
        """
        Performs a forward pass through the segmentor, extractor and predictor model.

        Parameters
        ----------
        features : FeaturesType
            The features.

        Returns
        -------
        prediction : TargetsType
            The prediction.
        """
        segmentation_dict = self.segmentor(features)
        segmentation_key = self.extractor.segmentation_key

        features = self._get_updated_features_with_segmentation(features, segmentation_dict, segmentation_key)
        radiomics = self.extractor(features)
        table_prediction = self.predictor(features, radiomics)

        return table_prediction | segmentation_dict

    def _validate_model_devices(self):
        """
        Validates that all models are on the same device.
        """
        devices = set()
        for setup in self._setups:
            if setup:
                if setup.model.device:
                    devices.add(setup.model.device)

        if self.device:
            devices.add(self.device)

        if len(devices) > 1:
            raise ValueError("All models must be on the same device.")

    def _validate_seed(self):
        """
        Validates that all models have the same seed.
        """
        seeds = set()
        for setup in self._setups:
            if setup:
                if setup.model.seed:
                    seeds.add(setup.model.seed)

        if self.seed:
            seeds.add(self.seed)

        if len(seeds) > 1:
            raise ValueError("All models must have the same seed.")

    def _build_models(self, dataset: ProstateCancerDataset):
        """
        Builds all models. It is used to initialize the model with the information contained in the dataset.
        """
        for setup in self._setups:
            if setup:
                setup.model.build(dataset)

    def _load_pretrained_models(self):
        """
        Loads pretrained models. It is used to initialize the model with the weights of pretrained models.
        """
        for setup in self._setups:
            if setup and setup.pretrained_model_state:
                setup.model.load_state_dict(setup.pretrained_model_state)

    def _freeze_models(self):
        """
        Freezes models. It is used to prevent the model from updating the weights of the pretrained models.
        """
        for setup in self._setups:
            if setup and setup.freeze:
                for param in setup.model.parameters():
                    param.requires_grad = False

                setup.model.eval()

    def build(self, dataset: ProstateCancerDataset) -> MultiNet:
        """
        Builds the model using information contained in the dataset from which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : MultiNet
            The model.
        """
        super().build(dataset=dataset)

        self._build_models(dataset)
        self._load_pretrained_models()
        self._freeze_models()

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        """
        Executes the forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        return self._forward_pass(features)
