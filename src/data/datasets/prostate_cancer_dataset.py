"""
    @file:              prostate_cancer_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom torch dataset named ProstateCancerDataset.
"""

from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset

from src.data.datasets.empty_dataset import DatasetType, EmptyDataset
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.table_dataset import TableDataset
from src.utils.tasks import Task


TargetsType = Dict[str, Union[np.ndarray, Tensor]]


class FeaturesType(NamedTuple):
    """
    Features element named tuple. This tuple is used to separate images and table features where
        - image : I-dimensional dictionary containing (N, ) tensor or array where I is the number of images used as
                  features.
        - table : D-dimensional dictionary containing (N, ) tensor or array where D is the number of table features.
    """
    image: Dict[str, Union[Tensor]] = {}
    table: Dict[str, Union[np.ndarray, Tensor]] = {}


class DataType(NamedTuple):
    """
    Data element named tuple. This tuple is used to separate features (x) and targets (y) where
        - x : D-dimensional dictionary containing (N, ) tensor or array where D is the number of features.
        - y : T-dimensional dictionary containing (N, ) tensor or array where T is the number of tasks.
    """
    x: FeaturesType
    y: TargetsType


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
        elif table_dataset is None or isinstance(table_dataset, EmptyDataset):
            self.table_dataset = EmptyDataset(DatasetType.TABLE)
            self.image_dataset = image_dataset
            self._n = len(image_dataset)
        elif image_dataset is None or isinstance(image_dataset, EmptyDataset):
            self.table_dataset = table_dataset
            self.image_dataset = EmptyDataset(DatasetType.IMAGE)
            self._n = len(table_dataset)
        else:
            tab_n, img_n = len(table_dataset), len(image_dataset)
            assert tab_n == img_n, f"Length of image dataset and table dataset must be equal. Image length : {img_n}" \
                                   f". Table length : {tab_n}."
            assert not set(table_dataset.tasks).intersection(image_dataset.tasks), f"Tasks in table and image " \
                                                                                   f"datasets should not overlap."
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
    ) -> Union[DataType, Subset]:
        """
        Gets dataset items.

        Parameters
        ----------
        index : Union[int, List[int]]
            Index.

        Returns
        -------
        items : Union[DataType, Subset]
            Data items from image and table datasets.
        """
        if isinstance(index, int):
            imaging_dict = self.image_dataset[index]

            y_imaging, x_imaging = {}, {}
            for key, item in imaging_dict.items():
                if key in [task.name for task in self.image_dataset.tasks]:
                    y_imaging[key] = item
                else:
                    x_imaging[key] = item

            return DataType(
                x=FeaturesType(image=x_imaging, table=self.table_dataset[index].x),
                y=dict(**self.table_dataset[index].y, **y_imaging)
            )
        elif isinstance(index, list):
            return Subset(dataset=self, indices=index)
        else:
            raise ValueError("Index has to be integer or list.")

    @property
    def tasks(self) -> List[Task]:
        if isinstance(self.table_dataset, EmptyDataset):
            return self.image_dataset.tasks
        elif isinstance(self.image_dataset, EmptyDataset):
            return self.table_dataset.tasks
        else:
            return self.table_dataset.tasks + self.image_dataset.tasks

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

        if isinstance(self.table_dataset, TableDataset):
            self.table_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)
