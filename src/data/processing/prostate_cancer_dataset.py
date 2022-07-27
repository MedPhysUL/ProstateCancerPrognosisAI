"""
    @file:              prostate_cancer_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom torch dataset named ProstateCancerDataset.
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from torch import tensor
from torch.utils.data import Dataset

from image_dataset import ImageDataset
from multi_task_table_dataset import MultiTaskTableDataset


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
            image_dataset: ImageDataset,
            table_dataset: MultiTaskTableDataset
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        image_dataset : ImageDataset
            An ImageDataset.
        table_dataset : MultiTaskTableDataset
            A MultiTaskTableDataset.
        """
        self.image_dataset = image_dataset
        self.table_dataset = table_dataset

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
