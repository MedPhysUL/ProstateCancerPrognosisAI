"""
    @file:              tools.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains helpful functions and classes used for pandas dataframe manipulations.
"""

from typing import List, Optional

import pandas as pd


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
