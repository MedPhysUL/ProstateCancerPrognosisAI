"""
    @file:              prostate_cancer_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom torch dataset named ProstateCancerDataset.
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from torch import nan, tensor
from torch.utils.data import Dataset

from src.data.processing.image_dataset import ImageDataset
from src.data.processing.multi_task_table_dataset import MultiTaskTableDataset


class EmptyDataset(Dataset):

    def __init__(
            self,
            to_tensor: bool = True
    ):
        self.to_tensor = to_tensor

    def __getitem__(self, index):
        if self.to_tensor:
            return nan
        else:
            return np.nan


class DataItem(NamedTuple):
    image: tuple  # TODO : Improve typing
    table: List[List[Optional[Union[Tuple[np.array, np.array, np.array], Tuple[tensor, tensor, tensor]]]]]


class ProstateCancerDataset(Dataset):
    """
    A custom dataset class used to perform multi-task experiments on tabular AND images data at once. Each
    ProstateCancerDataset contains an ImageDataset and a MultiTaskTableDataset.
    """

    def __init__(
            self,
            image_dataset: Optional[ImageDataset] = None,
            table_dataset: Optional[MultiTaskTableDataset] = None
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        image_dataset : Optional[ImageDataset]
            An ImageDataset.
        table_dataset : Optional[MultiTaskTableDataset]
            A MultiTaskTableDataset.
        """
        if (image_dataset is None) and (table_dataset is None):
            raise AssertionError("At least one image dataset or one table dataset must be provided.")

        if table_dataset is None:
            self.table_dataset = EmptyDataset()
            self.image_dataset = image_dataset
            self._length = len(image_dataset)
        elif image_dataset is None:
            self.table_dataset = table_dataset
            self.image_dataset = EmptyDataset(to_tensor=table_dataset.to_tensor)
            self._length = len(table_dataset)
        else:
            tab_length, img_length = len(table_dataset), len(image_dataset)
            assert tab_length == img_length, f"Length of image dataset and table dataset must be equal. Image length " \
                                             f": {img_length}. Table length : {tab_length}."
            self.table_dataset = table_dataset
            self.image_dataset = image_dataset
            self._length = len(image_dataset)

    def __len__(self):
        return self._length

    def __getitem__(
            self,
            index: Union[int, List[int]]
    ):
        """
        Gets dataset items.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index.

        Returns
        -------
        items : DataItem
            Data items from image and table datasets.
        """
        return DataItem(image=self.image_dataset[index], table=self.table_dataset[index])
