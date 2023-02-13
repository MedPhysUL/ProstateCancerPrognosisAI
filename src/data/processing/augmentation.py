"""
    @file:              augmentation.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file contains the Augmentation class which makes it possible to manage image/segmentation
                        augmentations.
"""

from copy import deepcopy
from typing import List, NamedTuple, Union

from monai.transforms import Compose
from torch.utils.data import ConcatDataset, Subset

from ..datasets import ImageDataset


class AugmentationTransforms(NamedTuple):
    """
    Tuple of image and segmentation transforms.
    """
    image_transforms: Compose
    segmentation_transforms: Compose


class CopyItems:
    """
    A class used to perform deepcopy to have multiple transform chains in the MONAI way.

    See multiple transform chains : https://docs.monai.io/en/stable/highlights.html#multiple-transform-chains
    """

    def __new__(
            cls,
            dataset: Union[ImageDataset, Subset]
    ) -> Union[ImageDataset, Subset]:
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        dataset : Union[ImageDataset, Subset]
            A dataset that contains images and segmentation maps.

        Returns
        -------
        dataset_copy : Union[ImageDataset, Subset]
            A copy of a dataset that contains images and segmentation maps.
        """
        if cls.is_dataset_type_valid(dataset) is False:
            raise TypeError(f"Dataset should be an instance of {ImageDataset} or {Subset}. Given dataset is of type "
                            f"{type(dataset)}.")

        return deepcopy(dataset)

    @staticmethod
    def is_dataset_type_valid(
            dataset: Union[ImageDataset, Subset]
    ) -> bool:
        """
        Check if given dataset is valid.

        Parameters
        ----------
        dataset : Union[ImageDataset, Subset]
            A dataset that contains images and segmentation maps.

        Returns
        -------
        valid : bool
            Whether dataset type is valid.
        """
        if isinstance(dataset, ImageDataset) or isinstance(dataset, Subset):
            return True
        else:
            return False


class Augmentation:
    """
    A class used to get multiple augmented copies of the original dataset.

    See multiple transform chains : https://docs.monai.io/en/stable/highlights.html#multiple-transform-chains
    """

    def __init__(
            self,
            augmentation_transforms: List[AugmentationTransforms]
    ) -> None:
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        augmentation_transforms : List[AugmentationTransforms]
            A list of augmentations to apply. For each augmentation in this list, a copy of the original dataset is
            created and the corresponding augmentation is applied.
        """
        self._augmentation_transforms = augmentation_transforms

    def get_augmented_dataset(
            self,
            dataset: Union[ImageDataset, Subset]
    ) -> ConcatDataset:
        """
        Gets the augmented dataset. For each augmentation in the self._augmentation_transforms list, a copy of the
        original dataset is created, the corresponding augmentation is applied and the copy is concatenated with the
        original dataset.

        Parameters
        ----------
        dataset : Union[ImageDataset, Subset]
            A dataset that contains images and segmentation maps.

        Returns
        -------
        concat_dataset : ConcatDataset
            A concat dataset that contains the original dataset and multiple augmented copies of the original dataset.
        """
        augmented_datasets = []

        for augmentation in self._augmentation_transforms:
            dataset_copy = CopyItems(dataset)

            if isinstance(dataset, ImageDataset):
                dataset_copy.dataset.dataset.data[0].transform = augmentation.image_transforms
                dataset_copy.dataset.dataset.data[1].transform = augmentation.segmentation_transforms
            else:
                dataset_copy.dataset.data[0].transform = augmentation.image_transforms
                dataset_copy.dataset.data[1].transform = augmentation.segmentation_transforms

            augmented_datasets.append(dataset_copy)

        return ConcatDataset([dataset] + augmented_datasets)
