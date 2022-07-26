"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Script used to generate a folder containing descriptive analyses of all data.
"""

import pandas as pd

from src.data.processing.multi_task_dataset import MultiTaskDataset
from src.data.processing.single_task_dataset import SingleTaskDataset
from src.visualization.table_viewer import TableViewer

from constants import *


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
    target_cols = [PN, BCR]

    single_task_datasets = []
    for target_col in target_cols:
        single_task_datasets.append(
            SingleTaskDataset(
                df=df[df[target_col].notna()],
                ids_col=ID,
                target_col=target_col,
                cont_cols=[AGE, PSA],
                cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
            )
        )

    multi_task_dataset = MultiTaskDataset(
        datasets=single_task_datasets,
        ids_to_row_idx=dict(pd.Series(df.index, index=df[ID]))
    )

    multi_task_dataset.update_masks(
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df)))
    )

    table_viewer = TableViewer(dataset=multi_task_dataset)
    table_viewer.visualize(path_to_save=DESCRIPTIVE_ANALYSIS_PATH)
