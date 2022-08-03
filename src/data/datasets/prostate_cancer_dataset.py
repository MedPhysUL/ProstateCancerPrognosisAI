"""
    @file:              prostate_cancer_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom torch dataset named ProstateCancerDataset.
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from src.data.datasets.empty_dataset import EmptyDataset
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.table_dataset import TableDataset


class DataItems(NamedTuple):
    """
    Data items named tuple. This tuple is used to separate images/segmentations data from tabular data.
    """
    image: Tuple[Sequence]
    table: Union[Tuple[np.array, np.array, np.array], Tuple[Tensor, Tensor, Tensor]]


class ProstateCancerDataset(Dataset):
    """
    A custom dataset class used to perform multi-task experiments on tabular AND images data at once. Each
    ProstateCancerDataset contains an ImageDataset and a TableDataset.
    """

    def __init__(
            self,
            image_dataset: Optional[ImageDataset] = None,
            table_dataset: Optional[TableDataset] = None
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        image_dataset : Optional[ImageDataset]
            An ImageDataset.
        table_dataset : Optional[TableDataset]
            A TableDataset.
        """
        if (image_dataset is None) and (table_dataset is None):
            raise AssertionError("At least one image dataset or one table dataset must be provided.")
        elif table_dataset is None:
            self.table_dataset = EmptyDataset()
            self.image_dataset = image_dataset
            self._n = len(image_dataset)
        elif image_dataset is None:
            self.table_dataset = table_dataset
            self.image_dataset = EmptyDataset(to_tensor=table_dataset.to_tensor)
            self._n = len(table_dataset)
        else:
            tab_n, img_n = len(table_dataset), len(image_dataset)
            assert tab_n == img_n, f"Length of image dataset and table dataset must be equal. Image length{img_n}. " \
                                   f"Table length : {tab_n}."
            self.table_dataset = table_dataset
            self.image_dataset = image_dataset
            self._n = len(image_dataset)

        self._train_mask, self._valid_mask, self._test_mask = [], None, []

        # We update current training mask with all the data
        self.update_masks(list(range(self._n)), [], [])

    def __len__(self) -> int:
        """
        Dataset length.

        Returns
        -------
        length : int
            Length of the dataset.
        """
        return self._n

    def __getitem__(
            self,
            index: Union[int, List[int]]
    ) -> DataItems:
        """
        Gets dataset items.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index.

        Returns
        -------
        items : DataItems
            Data items from image and table datasets.
        """
        return DataItems(image=self.image_dataset[index], table=self.table_dataset[index])

    @property
    def test_mask(self) -> List[int]:
        return self._test_mask

    @property
    def train_mask(self) -> List[int]:
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        return self._valid_mask

    def update_masks(
            self,
            train_mask: List[int],
            test_mask: List[int],
            valid_mask: Optional[List[int]] = None
    ) -> None:
        """
        Updates the train, valid and test masks.

        Parameters
        ----------
        train_mask : List[int]
            List of idx in the training set.
        test_mask : List[int]
            List of idx in the test set.
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        """
        # We set the new masks values
        self._train_mask, self._test_mask = train_mask, test_mask
        self._valid_mask = valid_mask if valid_mask is not None else []

        self.table_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)
