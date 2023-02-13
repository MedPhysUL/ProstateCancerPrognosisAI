"""
    @file:              tools.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains helpful functions and classes used for pandas dataframe manipulations.
"""

from typing import List, Optional, Tuple

import pandas as pd

from .transforms import ContinuousTransform as ConT
from .transforms import CategoricalTransform as CaT


def is_categorical(
        data: pd.Series
) -> bool:
    """
    Verifies if a variable is categorical using its data.

    Parameters
    ----------
    data : pd.Series
        Pandas series (column of a pandas dataframe).

    Returns
    -------
    categorical : bool
        True if categorical.
    """
    for item in data:
        if isinstance(item, str):
            return True

    if len(data.unique()) > 10:
        return False

    return True


def preprocess_categoricals(
        df: pd.DataFrame,
        encoding: str = "ordinal",
        mode: Optional[pd.Series] = None,
        encodings: Optional[dict] = None
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Applies all categorical transforms to a dataframe containing only continuous data

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all data.
    encoding : str
        One option in ("ordinal", "one-hot").
    mode : Optional[pd.Series]
        Pandas series with modes of columns.
    encodings : Optional[dict]
        Dict of dict with integers to use as encoding for each category's values

    Returns
    -------
    df, encodings : Tuple[pd.DataFrame, Optional[dict]]
        Pandas dataframe, dictionary of encodings.
    """
    if encoding not in ["ordinal", "one-hot"]:
        raise ValueError("Encoding option not available")

    # We ensure that all columns are considered as categories
    df = CaT.fill_missing(df, mode)

    if encoding == "ordinal":
        df, encodings_dict = CaT.ordinal_encode(df, encodings)
        return df, encodings_dict

    else:
        return CaT.one_hot_encode(df), None


def preprocess_continuous(
        df: pd.DataFrame,
        mean: Optional[pd.Series] = None,
        std: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Applies all continuous transforms to a dataframe containing only continuous data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all data.
    mean : Optional[pd.Series]
        Pandas series with mean.
    std : Optional[pd.Series]
        Pandas series with standard deviations

    Returns
    -------
    preprocessed_dataframe : pd.DataFrame
        Dataframe containing data on which all continuous transforms have been applied.
    """
    return ConT.normalize(ConT.fill_missing(df, mean), mean, std)


def retrieve_numerical_var(
        df: pd.DataFrame,
        to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Returns a dataframe containing only numerical variables of a given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    to_keep : Optional[List[str]]
        List of columns to keep in the returned dataframe no matter their types.

    Returns
    -------
    dataframe : pd.DataFrame
        Dataframe
    """
    # We convert the "to_keep" parameter into list if was not provided
    if to_keep is None:
        to_keep = []

    # We identify the columns to check
    cols_to_check = [col for col in df.columns if col not in to_keep]

    # We identify the categorical columns
    categorical_cols = [col for col in cols_to_check if not is_categorical(df[col])]

    return df[categorical_cols + to_keep]
