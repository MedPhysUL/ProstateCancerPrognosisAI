"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 05/2023

    @Description:       This file is used to define the `Transform` abstract class.
"""

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class Transform(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as data transforms. These classes are
    used to transform data before feeding it to the model.
    """

    @abstractmethod
    def __call__(self, df: pd.Series, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        An abstract method that defines the transform logic.

        Parameters
        ----------
        df : pd.Series
            Data to transform.
        kwargs : Dict[str, Any]
            Additional arguments.

        Returns
        -------
        df : Union[pd.DataFrame, pd.Series]
            Transformed data.
        """
        raise NotImplementedError


class Identity(Transform):
    """
    An identity transform that does nothing.
    """

    def __call__(self, df: pd.Series, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Returns the data as is.

        Parameters
        ----------
        df : pd.Series
            Data to transform.
        kwargs : Dict[str, Any]
            Additional arguments.

        Returns
        -------
        df : Union[pd.DataFrame, pd.Series]
            Transformed data.
        """
        return df
