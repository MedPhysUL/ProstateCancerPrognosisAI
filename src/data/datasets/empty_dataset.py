"""
    @file:              empty_dataset.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a custom EmptyDataset class.
"""

from typing import List, Union

import numpy as np
from torch import nan
from torch.utils.data import Dataset


class EmptyDataset(Dataset):
    """
    A custom empty dataset class.
    """

    def __init__(
            self,
            to_tensor: bool = True
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        to_tensor : bool
            Whether we want features and targets in tensor format. False for numpy arrays.
        """
        self.to_tensor = to_tensor

    def __getitem__(
            self,
            idx: Union[int, List[int]]
    ) -> Union[float, nan, np.nan]:
        """
        Gets dataset item. This method always returns NaN, regardless of the value of the index.

        Parameters
        ----------
        idx : Union[int, List[int]]
            Index.

        Returns
        -------
        item : Union[float, nan, np.nan]
            NaN value.
        """
        if self.to_tensor:
            return nan
        else:
            return np.nan
