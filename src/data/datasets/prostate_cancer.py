"""
    @file:              prostate_cancer.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file contains a custom torch dataset named 'ProstateCancerDataset'.
"""

from typing import Dict, List, NamedTuple, Optional, TypeAlias, Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset

from .empty import DatasetType, EmptyDataset
from .image import ImageDataset
from .table import TableDataset
from ...tasks import TaskList


TargetsType: TypeAlias = Dict[str, Union[np.ndarray, Tensor]]


class FeaturesType(NamedTuple):
    """
    Features element named tuple. This tuple is used to separate images and table features where
        - image : I-dimensional dictionary containing (N, ) tensor or array where I is the number of images used as
                  features.
        - table : D-dimensional dictionary containing (N, ) tensor or array where D is the number of table features.
    """
    image: Dict[str, Tensor] = {}
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
            assert tab_n == img_n, (
                f"Length of image dataset and table dataset must be equal. Image length : {img_n}. Table length : "
                f"{tab_n}."
            )
            assert not set(table_dataset.tasks).intersection(image_dataset.tasks), (
                f"Tasks in table and image datasets should not overlap."
            )
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
    def tasks(self) -> TaskList:
        """
        Returns the list of tasks.

        Returns
        -------
        tasks : TaskList
            List of tasks.
        """
        if isinstance(self.table_dataset, EmptyDataset):
            return self.image_dataset.tasks
        elif isinstance(self.image_dataset, EmptyDataset):
            return self.table_dataset.tasks
        else:
            return self.table_dataset.tasks + self.image_dataset.tasks

    @property
    def train_mask(self) -> List[int]:
        """
        Returns the train mask.

        Returns
        -------
        train_mask : List[int]
            List of idx in the training set.
        """
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        """
        Returns the valid mask.

        Returns
        -------
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        """
        return self._valid_mask

    @property
    def test_mask(self) -> Optional[List[int]]:
        """
        Returns the test mask.

        Returns
        -------
        test_mask : Optional[List[int]]
            List of idx in the test set.
        """
        return self._test_mask

    def update_masks(
            self,
            train_mask: List[int],
            valid_mask: Optional[List[int]] = None,
            test_mask: Optional[List[int]] = None
    ) -> None:
        """
        Updates the train, valid and test masks and then preprocesses the data available according to the current
        statistics of the training data.

        Parameters
        ----------
        train_mask : List[int]
            List of idx in the training set.
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        test_mask : Optional[List[int]]
            List of idx in the test set.
        """
        # We set the new masks values
        self._train_mask = train_mask
        self._valid_mask = valid_mask if valid_mask is not None else []
        self._test_mask = test_mask if test_mask is not None else []

        if isinstance(self.table_dataset, TableDataset):
            self.table_dataset.update_masks(train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)
