"""
    @file:              transforms.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains two classes, ContinuousTransform and CategoricalTransform, which simply list
                        methods that defines transformations that can be applied on data during preprocessing.
"""

from typing import Optional, Tuple

import pandas as pd
from torch import int64, float32, Tensor

from ...training.transforms import ToTensor


class ContinuousTransform:
    """
    Class of transformations that can be applied to continuous data
    """

    @staticmethod
    def normalize(
            df: pd.DataFrame,
            mean: Optional[pd.Series] = None,
            std: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Applies normalization to columns of a pandas dataframe
        """
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()

    @staticmethod
    def fill_missing(
            df: pd.DataFrame,
            mean: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fills missing values of continuous data columns with mean
        """
        if mean is not None:
            return df.fillna(mean)
        else:
            return df.fillna(df.mean())

    @staticmethod
    def to_tensor(
            df: pd.DataFrame
    ) -> Tensor:
        """
        Takes a dataframe with categorical columns and return a tensor with "longs".

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with categorical columns only.

        Returns
        -------
        df_tensor : Tensor
            Dataframe as a tensor.
        """
        transform = ToTensor(dtype=float32)

        return transform(df)


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """

    @staticmethod
    def one_hot_encode(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        One hot encodes all columns of the dataframe
        """
        return pd.get_dummies(df)

    @staticmethod
    def ordinal_encode(
            df: pd.DataFrame,
            encodings: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Applies ordinal encoding to all columns of the dataframe
        """
        if encodings is None:
            encodings = {}
            for c in df.columns:
                encodings[c] = {v: k for k, v in enumerate(df[c].cat.categories)}
                df[c] = df[c].cat.codes

        else:
            for c in df.columns:
                column_encoding = encodings[c]
                df[c] = df[c].apply(lambda x: column_encoding[x])

        return df, encodings

    @staticmethod
    def fill_missing(
            df: pd.DataFrame,
            mode: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fills missing values of continuous data columns with mode
        """
        if mode is not None:
            return df.fillna(mode)
        else:
            return df.fillna(df.mode().iloc[0])

    @staticmethod
    def to_tensor(
            df: pd.DataFrame
    ) -> Tensor:
        """
        Takes a dataframe with numerical columns and return a tensor with "floats"

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with categorical columns only.

        Returns
        -------
        df_tensor : Tensor
            Dataframe as a tensor.
        """
        transform = ToTensor(dtype=int64)

        return transform(df)
