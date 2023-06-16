"""
    @file:              unextractor.py
    @Author:            Raphael Brodeur

    @Creation Date:     06/2022
    @Last modification: 06/2023

    @Description:       This file is used to define a UNEXtractor model.
"""

from monai.networks.nets import FullyConnectedNet
from torch import cat, Tensor
from torch.nn import DataParallel, Module

from .base import Extractor, MergingMethod
from ....data.datasets.prostate_cancer import FeaturesType


class _UNEXtractorModule(Module):
    pass


class UNEXtractor(Extractor):
    """
    Description.
    """

    def __init__(self):
        pass

    def _get_layer(self):
        pass

    def __get_single_conv_sequence(self):
        pass

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

    def _build_single_extractor(self):
        """
        Description.
        """
        linear_module = self.__get_single_linear_module()

        return _UNEXtractorModule(linear_module=linear_module)

    def _build_extractor(self):
        """
        Description.
        """
        return self._build_single_extractor()

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
