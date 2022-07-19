"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Script used to generate a folder containing descriptive analyses of all data.
"""

import pandas as pd

from constants import *
from src.data.processing.cleaning import DataCleaner


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df.describe()
    holdout_df.describe()
