"""
    @file:              categorical.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 05/2023

    @Description:       This file is used to define the `OneHotEncoding`, `OrdinalEncoding` and `MappingEncoding`.
"""

from typing import Hashable, Mapping, Union

import pandas as pd

from .base import Transform


class OneHotEncoding(Transform):
    """
    Callable class that performs one-hot encoding on categorical data.
    """

    def __call__(self, df: pd.Series, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Performs one-hot encoding on categorical data.

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
        return pd.get_dummies(df)


class OrdinalEncoding(Transform):
    """
    Callable class that performs ordinal encoding on categorical data.
    """

    def __call__(self, df: pd.Series, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Performs ordinal encoding on categorical data.

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
        return df.cat.codes


class MappingEncoding(Transform):
    """
    Callable class that performs mapping encoding on categorical data.
    """

    def __init__(self, mapping: Mapping[Hashable, Union[float, int]]):
        """
        Sets protected attributes.

        Parameters
        ----------
        mapping : Mapping[Hashable, Union[float, int]]
            Mapping to use for encoding.
        """
        self.mapping = mapping

    def __call__(self, df: pd.Series, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Performs mapping encoding on categorical data.

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
        return df.map(self.mapping)
