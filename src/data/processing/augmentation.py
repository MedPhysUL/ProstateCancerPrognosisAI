"""
    @file:              augmentation.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Description.

"""
from copy import deepcopy


class Augmentation:
    """

    """

    def __call__(self, dataset, img_transform, seg_transform):
        ds_copy = deepcopy(dataset)
        ds_copy.dataset.data[0].transform = img_transform
        ds_copy.dataset.data[1].transform = seg_transform

from torch.utils.data.dataset import Subset


class Dataset(Subset):

    def __init__(self, subset: Subset):
        ds_copy = deepcopy(subset)

    def apply_transforms(self, transforms):
        # transform



new_ds = Dataset(ds_original)

