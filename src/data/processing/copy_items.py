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


class Augmentations(NamedTuple):
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

    def __init__(self, dataset: Union[ArrayDataset, Subset]):
        self.copy = CopyItems(dataset)

    def apply_augmentations(self, augmentations: List[Augmentations]):
        for augmentation in augmentations:
            self._apply_img_transforms(augmentation.img_transforms)
            self._apply_seg_transforms(augmentation.seg_transforms)

    def _apply_img_transforms(self, img_transforms):
        self.copy.dataset.data[0].transform = img_transforms

    def _apply_seg_transforms(self, seg_transforms):
        self.copy.dataset.data[1].transform = seg_transforms
