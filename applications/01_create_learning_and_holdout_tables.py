"""
    @file:              01_create_learning_and_holdout_tables.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to create the learning and holdout sets' csv tables.
"""

import pandas as pd

from src.data.processing.multi_task_table_dataset import MultiTaskTableDataset
from src.data.processing.sampling import RandomStratifiedSampler
from src.data.processing.single_task_table_dataset import MaskType, SingleTaskTableDataset

from constants import *


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_excel(io=CLINICAL_DATA_PATH, sheet_name="sheet1", header=1)
    # df = df[df[ID].isin(os.listdir(IMAGES_FOLDER_PATH))]  # Replace with list patients in HDF5

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR]

    df = df[[ID] + feature_cols + target_cols]

    single_task_datasets = []
    for target_col in target_cols:
        single_task_datasets.append(
            SingleTaskTableDataset(
                df=df[df[target_col].notna()],
                ids_col=ID,
                target_col=target_col,
                cont_cols=[AGE, PSA],
                cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
            )
        )

    multi_task_dataset = MultiTaskTableDataset(
        datasets=single_task_datasets,
        ids_to_row_idx=dict(pd.Series(df.index, index=df[ID]))
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Sampling                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    rss = RandomStratifiedSampler(
        dataset=multi_task_dataset,
        n_out_split=1,
        n_in_split=0,
        valid_size=0,
        test_size=HOLDOUT_SIZE,
        random_state=SEED
    )

    masks = rss()
    rss.visualize_splits(masks)
    learning_idx, holdout_idx = masks[0][MaskType.TRAIN], masks[0][MaskType.TEST]
    learning_df, holdout_df = df.iloc[learning_idx, :], df.iloc[holdout_idx, :]

    # ----------------------------------------------------------------------------------------------------------- #
    #                                              Saving DataFrames                                              #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df.to_csv(LEARNING_TABLE_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_TABLE_PATH, index=False)
