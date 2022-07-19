"""
    @file:              generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file stores the procedures that needs to be executed in order to extract descriptive tables
                        with information from all variables of a table.
"""

import os

import pandas as pd

from constants import *
from src.data.processing.cleaning import DataCleaner


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_excel(LEARNING_DATAFRAME_PATH)
    holdout_df = pd.read_excel(HOLDOUT_DATAFRAME_PATH)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Cleaner                                                     #
    # ----------------------------------------------------------------------------------------------------------- #
    data_cleaner = DataCleaner(
        records_path=RECORDS_PATH,
        ids_col=ID,
    )

    data_cleaner(df=learning_df)
    data_cleaner(df=holdout_df)
