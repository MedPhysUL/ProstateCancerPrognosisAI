"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 04/2023

    @Description:       This script is used to generate a folder containing descriptive analyses of all data.
"""

import pandas as pd

from constants import *
from src.data.datasets import TableDataset
from src.visualization import TableViewer


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=CONTINUOUS_FEATURE_COLUMNS,
        cat_cols=CATEGORICAL_FEATURE_COLUMNS
    )

    table_dataset.update_masks(
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df)))
    )

    table_viewer = TableViewer(dataset=table_dataset)
    table_viewer.save_descriptive_analysis(path_to_save=DESCRIPTIVE_ANALYSIS_PATH)
