"""
    @file:              02_generate_outliers_records.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to generate a folder containing information about outliers.
"""

import pandas as pd

from constants import *
from src.data.processing.cleaning import Cleaner


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)[[ID] + FEATURE_COLUMNS]
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)[[ID] + FEATURE_COLUMNS]

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Cleaner                                                     #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_set_data_cleaner = Cleaner(
        records_path=f"{OUTLIERS_RECORDS_PATH}/learning_set",
        ids_col=ID,
    )

    learning_set_data_cleaner(df=learning_df)

    data_cleaner = Cleaner(
        records_path=f"{OUTLIERS_RECORDS_PATH}/holdout_set",
        ids_col=ID,
    )

    data_cleaner(df=holdout_df)
