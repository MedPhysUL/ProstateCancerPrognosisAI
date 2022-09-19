"""
    @file:              empty_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom EmptyDataset class.
"""

from enum import IntEnum
from typing import Dict, List, Union

from torch.utils.data import Dataset

from src.data.datasets.table_dataset import TableDataModel


class DatasetType(IntEnum):
    """
    Dataset type.
    """
    TABLE = 1
    IMAGE = 2


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
        Gets dataset item. This method always returns NaN, regardless of the value of the index.

        Parameters
        ----------
        idx : Union[int, List[int]]
            Index.

        Returns
        -------
        item : Union[Dict, TableDataModel]
            Empty dict or data element.
        """
        if self.dataset_type == self.dataset_type.TABLE:
            return TableDataModel(x={}, y={})
        elif self.dataset_type == self.dataset_type.IMAGE:
            return {}
