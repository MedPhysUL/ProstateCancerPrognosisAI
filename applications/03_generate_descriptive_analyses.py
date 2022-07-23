"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Script used to generate a folder containing descriptive analyses of all data.
"""

import pandas as pd

from constants import *
from src.data.processing.dataset import ProstateCancerDataset
from src.visualization.table_viewer import TableViewer


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv("local_data/learning_table.csv")
    holdout_df = pd.read_csv("local_data/holdout_table.csv")

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    target_cols = [PN, BCR]

    dataset = ProstateCancerDataset(
        df=pd.concat([learning_df, holdout_df]),
        ids_col=ID,
        target_cols=target_cols,
        cont_cols=[column for column, typ in COLUMNS_TYPES.items() if (typ is NUMERIC_TYPE) and (column in learning_df)],
        cat_cols=[column for column, typ in COLUMNS_TYPES.items() if (typ is CATEGORICAL_TYPE) and (column in learning_df) and (column not in target_cols)]
    )
    dataset.update_masks(
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df)))
    )

    TableViewer(dataset=dataset).visualize()
