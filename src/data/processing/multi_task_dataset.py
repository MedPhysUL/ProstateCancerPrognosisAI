"""
    @file:              multi_task_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom torch dataset named MultiTaskDataset.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch import tensor
from torch.utils.data import Dataset

from src.data.processing.single_task_dataset import SingleTaskDataset


class MultiTaskDataset(Dataset):
    """
    A custom dataset class used to perform multi-task experiments. Each MultiTaskDataset is composed of several
    SingleTaskDatasets. All multi-task logic is integrated into the MultiTaskDataset object and only affects the
    single-task datasets through masks updates. This class composition allows to cover the cases where some patients
    have a missing label for one task while it is available for another.
    """

    def __init__(
            self,
            datasets: Sequence[SingleTaskDataset],
            ids_to_row_idx: Dict[str, int]
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        datasets : Sequence[SingleTaskDataset]
            Sequence of single task datasets. They need to have the same patient IDs column name.
        ids_to_row_idx : Dict[str, int]
            Dictionary that associates patient IDs with their corresponding row indices in the original dataframe.
        """
        self.datasets = datasets
        self.ids_to_row_idx = ids_to_row_idx

        if not self._is_ids_cols_valid():
            raise ValueError("Datasets in a multi-task learning setting need to have the same IDs column name.")
        else:
            self._ids_col = self.datasets[0].ids_col

        if not self._is_ids_to_row_idx_valid():
            raise ValueError("Each patient ID must be associated to a unique row index.")
        else:
            self._row_idx_to_ids = {v: k for k, v in self.ids_to_row_idx.items()}

    def __getitem__(
            self,
            idx: Union[int, List[int]]
    ) -> List[List[Optional[Union[Tuple[np.array, np.array, np.array], Tuple[tensor, tensor, tensor]]]]]:
        """
        Gets dataset items. NB if a given index (corresponding to a specific patient ID) is not available in a
        specific dataset, the None keyword is returned at that specific location. The items are given in the following
        format :

            items = [
                  ┏━ x array of the item with index = 0 in dataset 0.
                  ┃               ┏━ y array of the item with index = 1 in dataset 0.
                  ┃               ┃          ┏━ Item with index = 2 is None in dataset 0.
                [(x, y, idx), (x, y, idx), None, (x, y, idx), ...],              <------ Items from dataset 0.
                [(x, y, idx), (x, y, idx), (x, y, idx), (x, y, idx), ...],       <------ Items from dataset 1.
                ...
            ]

        NOTICE : You might want to access only the items of one of the single task datasets contained in the
                 self.datasets list. To do this, just call the __getitem__ method of that particular dataset, and not
                 the current __getitem__ which is used to access the items of all datasets at once.

        Parameters
        ----------
        idx : Union[int, List[int]]
            Index.

        Returns
        -------
        item : List[List[Optional[Union[Tuple[np.array, np.array, np.array], Tuple[tensor, tensor, tensor]]]]]
            Item.
        """
        item = []
        ids = self._convert_row_idx_to_ids(idx)

        for dataset in self.datasets:
            ds_item = []
            if isinstance(ids, str):
                if ids in list(dataset.ids_to_row_idx.keys()):
                    i = dataset.ids_to_row_idx[ids]
                    ds_item.append(dataset[i])
                else:
                    ds_item.append(None)
            else:
                for id_ in ids:
                    if id_ in list(dataset.ids_to_row_idx.keys()):
                        i = dataset.ids_to_row_idx[id_]
                        ds_item.append(dataset[i])
                    else:
                        ds_item.append(None)

            item.append(ds_item)

        return item

    def _is_ids_cols_valid(
            self
    ) -> bool:
        """
        Checks if all datasets have the same patient IDs column name.

        Returns
        -------
        validity : bool
            Whether given datasets have valid IDs column name.
        """
        if all(ds.ids_col == self.datasets[0].ids_col for ds in self.datasets):
            return True
        else:
            return False

    def _is_ids_to_row_idx_valid(
            self
    ) -> bool:
        """
        Checks if each patient ID is associated to a unique row index.

        Returns
        -------
        validity : bool
            Whether given dictionary contains unique row indexes.
        """
        if len(self.ids_to_row_idx) == len(set(list(self.ids_to_row_idx.values()))):
            return True
        else:
            return False

    def _convert_row_idx_to_ids(
            self,
            idx: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """
        Converts row indexes to patient IDs.

        Parameters
        ----------
        idx: Union[int, List[int]]
            Indexes.

        Returns
        -------
        ids : Union[str, List[str]]
            Patient IDs.
        """
        if isinstance(idx, int):
            return self._row_idx_to_ids[idx]
        else:
            return [self._row_idx_to_ids[i] for i in idx]

    def get_targets(
            self
    ) -> List[np.ndarray]:
        """
        Gets list of target arrays. Those arrays may contain NaN as they are not filtered.

        WARNING! This method should not be used to access specific dataset.y values. Always iterate through
        self.datasets to have access to the true underlying attributes of the single task datasets.

        Returns
        -------
        targets : List[np.ndarray]
            List of target arrays. The length of the arrays is equal to the length of self.ids_to_row_idx, i.e. the
            dictionary containing all patient IDs.
        """
        df = pd.DataFrame({self.datasets[0].ids_col: list(self.ids_to_row_idx.keys())})
        for ds in self.datasets:
            target_df = pd.DataFrame({self._ids_col: ds.ids, ds.target_col: ds.y})
            df = df.merge(target_df, how="outer", on=self._ids_col)

        del df[self._ids_col]

        return [np.array(df[col]) for col in df]

    def update_masks(
            self,
            train_mask: List[int],
            test_mask: List[int],
            valid_mask: Optional[List[int]] = None
    ) -> None:
        """
        Updates the train, valid and test masks and then preprocesses the data available according to the current
        statistics of the training data for each of the single task datasets respectively.

        Parameters
        ----------
        train_mask : List[int]
            List of idx in the training set.
        test_mask : List[int]
            List of idx in the test set.
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        """
        train_ids = self._convert_row_idx_to_ids(train_mask)
        test_ids = self._convert_row_idx_to_ids(test_mask)

        valid_mask = valid_mask if valid_mask is not None else []
        valid_ids = self._convert_row_idx_to_ids(valid_mask)

        for dataset in self.datasets:
            train_idx = [dataset.ids_to_row_idx[id_] for id_ in train_ids if id_ in dataset.ids]
            test_idx = [dataset.ids_to_row_idx[id_] for id_ in test_ids if id_ in dataset.ids]
            valid_idx = [dataset.ids_to_row_idx[id_] for id_ in valid_ids if id_ in dataset.ids]

            dataset.update_masks(train_mask=train_idx, test_mask=test_idx, valid_mask=valid_idx)
