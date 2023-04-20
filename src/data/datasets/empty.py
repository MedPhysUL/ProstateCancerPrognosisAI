"""
    @file:              empty.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file contains a custom EmptyDataset class.
"""

from enum import Enum
from typing import Dict, List, Union

from torch.utils.data import Dataset

from .table import TableDataModel


class DatasetType(Enum):
    """
    Dataset type.
    """
    TABLE = "table"
    IMAGE = "image"


class EmptyDataset(Dataset):
    """
    A custom empty dataset class.
    """

    def __init__(
            self,
            dataset_type: DatasetType
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        dataset_type : DatasetType
            Dataset type.
        """
        if dataset_type in DatasetType:
            self.dataset_type = dataset_type
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}.")

    def __getitem__(
            self,
            idx: Union[int, List[int]]
    ) -> Union[Dict, TableDataModel]:
        """
        Gets dataset items. The particularity of the EmptyDataset is that this method always returns empty items,
        regardless of the value of the index.

        Parameters
        ----------
        idx : Union[int, List[int]]
            Indexes of items in the dataset.

        Returns
        -------
        item : Union[Dict, TableDataModel]
            Empty dict or data element.
        """
        if self.dataset_type == self.dataset_type.TABLE:
            return TableDataModel(x={}, y={})
        elif self.dataset_type == self.dataset_type.IMAGE:
            return {}
