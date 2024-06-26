"""
    @file:              unextractor.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     06/2022
    @Last modification: 07/2023

    @Description:       This file is used to define a UNEXtractor model.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Union

from torch import cat, mean, stack, sum, Tensor
from torch import device as torch_device
from torch.nn import DataParallel, Module, ModuleDict, Sequential

from .base import Extractor, ExtractorKLDivergence, ExtractorOutput, MultiTaskMode
from ..blocks import BayesianDecoderBlock, BayesianEncoderBlock, DecoderBlock, EncoderBlock
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _UNet(Module):
    """
    This class is used to define an unet used for segmentation and extraction of deep radiomics.
    """

    def __init__(
            self,
            encoders: ModuleDict,
            decoders: ModuleDict,
            bayesian: bool = False
    ):
        """
        Initializes the model.

        Parameters
        ----------
        encoders : ModuleDict
            A ModuleDict of the encoders to be used by the unet.
        decoders : ModuleDict
            A ModuleDict of the decoders to be used by the unet.
        bayesian : bool
            Whether the model implements variational inference.
        """
        super().__init__()

        self.encoders = encoders
        self.decoders = decoders
        self.bayesian = bayesian

    def _bayesian_forward(self, input_tensor: Tensor) -> tuple[ExtractorOutput, ExtractorKLDivergence]:
        """
        Forward pass for bayesian model.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        output : tuple[ExtractorOutput, ExtractorKLDivergence]
            The deep features and segmentations and their corresponding KL divergence.
        """
        dim = tuple(range(2, len(input_tensor.shape)))
        x = input_tensor
        features = []

        kl_list = []

        layers_output = {}
        for key, encoder in self.encoders.items():
            x, kl = encoder(x)
            layers_output[key] = x
            kl_list.append(kl)

            global_average_pool = mean(x, dim=dim)
            features.append(global_average_pool)

        features = cat(features, dim=1)

        deep_features_kl = sum(stack(kl_list))

        for key, decoder in reversed(list(self.decoders.items())):
            x, kl = decoder(cat([layers_output[key], x], dim=1))
            kl_list.append(kl)

        segmentation_kl = sum(stack(kl_list))

        kl_divergence = ExtractorKLDivergence(deep_features=deep_features_kl, segmentation=segmentation_kl)

        return ExtractorOutput(deep_features=features, segmentation=x), kl_divergence

    def _deterministic_forward(self, input_tensor: Tensor) -> ExtractorOutput:
        """
        Forward pass for deterministic model.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        output : ExtractorOutput
            The output of the forward pass. It contains the deep features and the segmentation.
        """
        dim = tuple(range(2, len(input_tensor.shape)))
        x = input_tensor
        features = []

        layers_output = {}
        for key, encoder in self.encoders.items():
            x = encoder(x)
            layers_output[key] = x

            global_average_pool = mean(x, dim=dim)
            features.append(global_average_pool)

        features = cat(features, dim=1)

        for key, decoder in reversed(list(self.decoders.items())):
            x = decoder(cat([layers_output[key], x], dim=1))

        return ExtractorOutput(deep_features=features, segmentation=x)

    def forward(self, input_tensor: Tensor) -> Union[ExtractorOutput, tuple[ExtractorOutput, ExtractorKLDivergence]]:
        """
        Forward pass. Applies different forward methods depending on whether the model is bayesian.

        Parameters
        ----------
        input_tensor : Tensor
            The input tensor.

        Returns
        -------
        output : Union[ExtractorOutput, tuple[ExtractorOutput, ExtractorKLDivergence]]
            The output of the forward pass. It contains the deep features and segmentations and their respective KL
            divergence if the model is in bayesian mode.
        """
        if self.bayesian:
            return self._bayesian_forward(input_tensor=input_tensor)

        else:
            return self._deterministic_forward(input_tensor=input_tensor)


class UNEXtractor(Extractor):
    """
    This class contains the UNEXtractor Extractor which is used to extract deep radiomics while inferring segmentations.
    """

    def __init__(
            self,
            image_keys: Union[str, List[str]],
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            shape: Sequence[int] = (128, 128, 128),
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
            seed: Optional[int] = None,
            bayesian: bool = False,
            temperature: Optional[Dict[str, float]] = None,
            prior_mean: float = 0.0,
            prior_variance: float = 0.1,
            posterior_mu_init: float = 0.0,
            posterior_rho_init: float = -3.0,
            standard_deviation: float = 0.1
    ):
        """
        Initializes the model.

        Parameters
        ----------
        image_keys : Union[str, List[str]]
            Sequence of images keys to extract deep radiomics from.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'partly_shared' or 'fully_shared'. If 'partly_shared', a separate extractor model will
            be used for each task. If 'fully_shared', a fully shared extractor model will be used. All layers will be
            shared between the tasks.
        shape : Sequence[int]
            Sequence of integers stating the dimension of the input tensor (minus batch and channel dimensions). Can
            also be given as a string containing the sequence. Default to (128, 128, 128).
        n_features : int
            Integer stating the dimension of the final output tensor, i.e. the number of deep features to extract from
            the image.
        channels : Union[str, Sequence[int]]
            Sequence of integers stating the output channels of each convolutional layer. Can also be given as a string
            containing the sequence.
        strides : Optional[Sequence[int]]
            Sequence of integers stating the stride (downscale factor) of each convolutional layer. Has to be the length
            of channels - 1. Defaults to 2.
        kernel_size : Union[int, Sequence[int]]
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
        bayesian : bool
            Whether the model implements variational inference.
        temperature : Optional[Dict[str, float]]
            Dictionary containing the temperature for each tasks. The temperature is the coefficient by which the KL
            divergence is multiplied when the loss is being computed. Keys are the task names and values are the
            temperature for each task.
        prior_mean : float
            Mean of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        prior_variance : float
            Prior variance used to calculate KL divergence.
        posterior_mu_init : float
            Initial value of the trainable mu parameter representing the mean of the Gaussian approximate of the
            posterior distribution.
        posterior_rho_init : float
            Rho parameter for reparametrization for the initial posterior distribution.
        standard_deviation : float
            Standard deviation of the gaussian distribution used to sample the initial posterior mu and initial
            posterior rho for the gaussian distribution from which the initial weights are sampled.
        """
        super().__init__(
            image_keys=image_keys,
            multi_task_mode=multi_task_mode,
            shape=shape,
            n_features=n_features,
            activation=activation,
            channels=channels,
            dropout_fnn=dropout_fnn,
            hidden_channels_fnn=hidden_channels_fnn,
            device=device,
            name=name,
            seed=seed,
            bayesian=bayesian,
            temperature=temperature,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            standard_deviation=standard_deviation
        )

        self.strides = strides if strides else [2] * (len(self.channels) - 1)
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.norm = norm
        self.dropout_cnn = dropout_cnn

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.standard_deviation = standard_deviation

    def _get_encoders_dict(self) -> ModuleDict:
        """
        Returns a ModuleDict of encoder blocks to be used by the UNet in its encoding path.

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
                    input_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                    output_channels=c,
                    num_res_units=self.num_res_units,
                    kernel_size=self.kernel_size,
                    stride=1 if i == len(self.channels) - 1 else self.strides[i],
                    act=self.activation,
                    norm=self.norm,
                    dropout=self.dropout_cnn,
                    prior_mean=self.prior_mean,
                    prior_variance=self.prior_variance,
                    posterior_mu_init=self.posterior_mu_init,
                    posterior_rho_init=self.posterior_rho_init,
                    standard_deviation=self.standard_deviation
                )
            else:
                conv = EncoderBlock(
                    input_channels=self.in_shape[0] if i == 0 else self.channels[i - 1],
                    output_channels=c,
                    num_res_units=self.num_res_units,
                    kernel_size=self.kernel_size,
                    stride=1 if i == len(self.channels) - 1 else self.strides[i],
                    act=self.activation,
                    norm=self.norm,
                    dropout=self.dropout_cnn
                )

            enc.add_module(name=f"conv{i}", module=conv)

            encoders["bottom" if i == len(self.channels) - 1 else f"layer{i}"] = enc

        return encoders

    def _get_decoders_dict(self) -> ModuleDict:
        """
        Returns a ModuleDict of decoder blocks to be used by the UNet in its decoding path.

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
                        output_channels=len(self._tasks.segmentation_tasks) if i == 0 else self.channels[i - 1],
                        num_res_units=self.num_res_units,
                        kernel_size=self.kernel_size,
                        stride=self.strides[i],
                        act=self.activation,
                        norm=self.norm,
                        dropout=self.dropout_cnn,
                        is_top=True if i == 0 else False,
                        prior_mean=self.prior_mean,
                        prior_variance=self.prior_variance,
                        posterior_mu_init=self.posterior_mu_init,
                        posterior_rho_init=self.posterior_rho_init,
                        standard_deviation=self.standard_deviation
                    )
                else:
                    up_conv = DecoderBlock(
                        input_channels=c + self.channels[i + 1] if i == len(self.channels) - 2 else c * 2,
                        output_channels=len(self._tasks.segmentation_tasks) if i == 0 else self.channels[i - 1],
                        num_res_units=self.num_res_units,
                        kernel_size=self.kernel_size,
                        stride=self.strides[i],
                        act=self.activation,
                        norm=self.norm,
                        dropout=self.dropout_cnn,
                        is_top=True if i == 0 else False
                    )

                dec.add_module(name=f"up_conv{i}", module=up_conv)

                decoders[f"layer{i}"] = dec

        return decoders

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
        assert len(self._tasks.segmentation_tasks) > 0, (
            "UNEXtractor requires at least one segmentation task. Found none."
        )

        unet = _UNet(
            encoders=self._get_encoders_dict(),
            decoders=self._get_decoders_dict(),
            bayesian=self.bayesian
        )

        return DataParallel(unet).to(self.device)
