"""
    @file:              01_create_learning_and_holdout_tables.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to create the learning and holdout sets' csv tables.
"""

import pandas as pd

from constants import *
from src.data.datasets import TableDataset
from src.data.processing.sampling import Mask, Sampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_excel(io=CLINICAL_DATA_PATH, sheet_name="sheet1", header=1)
    # df = df[df[ID].isin(os.listdir(IMAGES_FOLDER_PATH))]

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Sampling                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    rss = Sampler(
        dataset=table_dataset,
        n_out_split=1,
        n_in_split=0,
        valid_size=0,
        test_size=HOLDOUT_SIZE,
        random_state=SEED
    )

    masks = rss()
    rss.visualize_splits(masks)
    learning_idx, holdout_idx = masks[0][Mask.TRAIN], masks[0][Mask.TEST]
    learning_df, holdout_df = df.iloc[learning_idx, :], df.iloc[holdout_idx, :]

    # ----------------------------------------------------------------------------------------------------------- #
    #                                              Saving DataFrames                                              #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df.to_csv(LEARNING_TABLE_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_TABLE_PATH, index=False)
