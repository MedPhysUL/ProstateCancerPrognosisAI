"""
    @file:              augmentation.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 11/2022

    @Description:       This file contains the Augmentation class which make it possible to manage image/segmentation
                        augmentations.
"""

from copy import deepcopy
from typing import List, NamedTuple, Union

from monai.transforms import Compose
from torch.utils.data import ConcatDataset, Subset

from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset


class CopyItems:
    """
    A class used to perform deepcopy to have multiple transform chains in the MONAI way.

    See multiple transform chains : https://docs.monai.io/en/stable/highlights.html#multiple-transform-chains
    """

    def __new__(
            cls,
            dataset: Union[ProstateCancerDataset, Subset]
    ) -> Union[ProstateCancerDataset, Subset]:
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        dataset : Union[ProstateCancerDataset, Subset]
            A dataset that contains images and segmentation maps.

        Returns
        -------
        dataset_copy : Union[ProstateCancerDataset, Subset]
            A copy of a dataset that contains images and segmentation maps.
        """
        if cls.is_dataset_type_valid(dataset) is False:
            raise TypeError(
                f"Dataset should be an instance of {ProstateCancerDataset} or {Subset}. Given dataset is of type "f"{type(dataset)}.")

        return deepcopy(dataset)

    @staticmethod
    def is_dataset_type_valid(
            dataset: Union[ProstateCancerDataset, Subset]
    ) -> bool:
        """
        Checks if given dataset is valid.

        Parameters
        ----------
        dataset : Union[ProstateCancerDataset, Subset]
            A dataset that contains images and segmentation maps.

        Returns
        -------
        valid : bool
            Whether dataset type is valid.
        """
        if isinstance(dataset, ProstateCancerDataset) or isinstance(dataset, Subset):
            return True
        else:
            return False


class Augmentation:
    """
    A class used to get an augmented dataset.
    """

    def __init__(
            self,
            dataset: Union[ProstateCancerDataset, Subset]
    ) -> None:
        """
        Sets protected attributes of our custom dataset class.

        Parameters
        ----------
        dataset : Union[ProstateCancerDataset, Subset]
            The dataset on which to apply augmentations.
        """
        self._dataset = dataset

    def synthetic(
            self,
            augmentation_transforms: List[Compose]
    ) -> ConcatDataset:
        """
        Applies synthetic augmentations. For each augmentation in the augmentation_transforms list, a copy of the
        original dataset is created, the corresponding augmentation is applied and the copy is concatenated with the
        original dataset.

        Parameters
        ----------
        augmentation_transforms : List[Compose]
            A list of augmentations to apply. For each augmentation in this list, a copy of the original dataset is
            created and the corresponding augmentation is applied.

        Returns
        -------
        concat_dataset : ConcatDataset
            A concat dataset that contains the original dataset and the augmented copies.
        """
        augmented_datasets = []

        for augmentation in augmentation_transforms:
            dataset_copy = CopyItems(self._dataset)

            if isinstance(self._dataset, ProstateCancerDataset):
                dataset_copy.image_dataset.transform = augmentation
            if isinstance(self._dataset, Subset):
                dataset_copy.dataset.image_dataset.transform = augmentation

            augmented_datasets.append(dataset_copy)

        return ConcatDataset([self._dataset] + augmented_datasets)

    def live(
            self,
            augmentation_transforms: Compose
    ) -> Union[ProstateCancerDataset, Subset]:
        """
        Applies live augmentation. Applies new transforms to an already existing dataset after its train-validation
        split. Changes initial transforms to randomized transforms for augmentation at each epoch.

        Parameters
        ----------
        augmentation_transforms : Compose
            The new transformations to be made on the dataset.

        Returns
        -------
        augmented_dataset : Union[ProstateCancerDataset, Subset]
            The original dataset with new transformations.
        """
        dataset_copy = CopyItems(self._dataset)

        if isinstance(self._dataset, ProstateCancerDataset):
            dataset_copy.image_dataset.transform = augmentation_transforms
        if isinstance(self._dataset, Subset):
            dataset_copy.dataset.image_dataset.transform = augmentation_transforms

        return dataset_copy
