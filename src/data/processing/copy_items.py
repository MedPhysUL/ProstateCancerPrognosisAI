"""
    @file:              copy_items.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Description.

"""

from copy import deepcopy
from typing import List, NamedTuple, Union

from monai.data.dataset import ArrayDataset
from monai.transforms import Compose
from torch.utils.data.dataset import Subset
from torch.utils.data import ConcatDataset


class AugmentationTransforms(NamedTuple):
    img_transforms: Compose
    seg_transforms: Compose


class CopyItems:
    """
    """

    def __new__(cls, dataset: Union[ArrayDataset, Subset]):
        """
        """
        if cls.is_dataset_type_valid(dataset) is False:
            raise TypeError(f"Dataset should be an instance of {ArrayDataset} or {Subset}. Given dataset is of type "
                            f"{type(dataset)}.")

        return deepcopy(dataset)

    @staticmethod
    def is_dataset_type_valid(dataset: Union[ArrayDataset, Subset]):
        """
        """
        if isinstance(dataset, ArrayDataset) or isinstance(dataset, Subset):
            return True
        else:
            return False


class Augmentation:
    # Not sure about the name of this class 

    def __init__(self, augmentation_transforms: List[AugmentationTransforms]):
        self._augmentation_transforms = augmentation_transforms

    def get_augmented_dataset(self, dataset: Union[ArrayDataset, Subset]):
        augmented_datasets = []

        for augmentation in self._augmentation_transforms:
            dataset_copy = CopyItems(dataset)
            dataset_copy.dataset.dataset.data[0].transform = augmentation.img_transforms  # Remove recurrent .dataset if splitting with slicing
            dataset_copy.dataset.dataset.data[1].transform = augmentation.seg_transforms

            augmented_datasets.append(dataset_copy)

        return ConcatDataset([dataset] + augmented_datasets)
