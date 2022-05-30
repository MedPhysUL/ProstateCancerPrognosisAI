"""
    @file:              preprocessing.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains a series of functions related to preprocessing tabular data.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.processing.transforms import ContinuousTransform as ConT
from src.data.processing.transforms import CategoricalTransform as CaT


ENCODINGS = ["ordinal", "one-hot"]


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
    if encoding not in ENCODINGS:
        raise ValueError("Encoding option not available")

    # We ensure that all columns are considered as categories
    df = CaT.fill_missing(df, mode)

    if encoding == "ordinal":
        df, encodings_dict = CaT.ordinal_encode(df, encodings)
        return df, encodings_dict

    else:
        return CaT.one_hot_encode(df), None


def create_groups(
        df: pd.DataFrame,
        cont_col: str,
        nb_group: int
) -> pd.DataFrame:
    """
    Change each value of the column cont_col for its belonging group computed using nb_group quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all data.
    cont_col : str
        Name of the continuous column.
    nb_group : int
        Number of quantiles (group) wanted.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with modified column.
    """

    if nb_group < 2:
        raise ValueError('Must have at least 2 groups')

    # We compute number needed to calculate quantiles
    percentage = np.linspace(0, 1, nb_group+1)[1:-1]

    # We sort values in an ascending way
    df = df.sort_values(by=cont_col)

    # We retrieve the numeric value to calculate the quantiles
    data = df.loc[:, cont_col].astype(float).values

    # We set few local variables
    group = cont_col.upper()
    max_ = df[cont_col].max()
    quantiles = []

    # We compute quantiles
    for p in percentage:
        quantiles.append(round(np.quantile(data, p), 2))

    # We change row values
    j = turn_to_range(df, cont_col, 0, quantiles[0], group=f"{group} <= q1")
    for i in range(1, len(quantiles)):
        j = turn_to_range(df, cont_col, j, quantiles[i], group=f"{group} >q{i-1},<=q{i}")

    _ = turn_to_range(df, cont_col, j, max_+1, group=f"{group} >q{len(quantiles)}")

    return df


def turn_to_range(
        df: pd.DataFrame,
        cont_col: str,
        start_index: int,
        quantile: float,
        group: str
) -> int:
    """
    Changes categorical values of selected index into a string representing a range.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all data.
    cont_col : str
        Name of a continuous column.
    start_index : int
        Index where to start the modification.
    quantile : float
        Quantile to reach.
    group : str
        Name of the group to be assigned.

    Returns
    -------
    Index : int
        Index where the quantile was exceeded.
    """
    # We get the index of the column
    j = df.columns.get_loc(cont_col)

    for i in range(start_index, df.shape[0]):
        if df.iloc[i, j] < quantile:
            df.iloc[i, j] = group
        else:
            return i


def remove_nan(
        record: List[str]
) -> List[str]:
    """
    Removes nans from a record.

    Parameters
    ----------
    record : List[str]
        List of strings.

    Returns
    -------
    curated list : List[str]
        List of strings.
    """
    record = [x for x in record if str(x) != 'nan']

    return record
