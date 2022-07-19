"""
    @file:              01_create_learning_and_holdout_tables.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Small script used to create the learning and holdout sets csv tables.
"""

import os

import pandas as pd

from constants import *
from src.data.processing.dataset import MaskType, ProstateCancerDataset
from src.data.processing.sampling import RandomStratifiedSampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_excel(io=CLINICAL_DATA_PATH, sheet_name="sheet1", header=1)
    df = df[df[ID].isin(os.listdir(IMAGES_FOLDER_PATH))]

    targets_cols = [PN, BCR]
    features_cols = [col for col in COLUMNS_TYPES.keys()]

    df = df[features_cols + targets_cols]
    target_df = df[targets_cols]

    dataset = ProstateCancerDataset(
        df=df,
        ids_col=ID,
        targets_col=targets_cols,
        cont_cols=[column for column, typ in COLUMNS_TYPES.items() if typ is NUMERIC_TYPE],
        cat_cols=[column for column, typ in COLUMNS_TYPES.items() if typ is CATEGORICAL_TYPE]
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Sampling                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    rss = RandomStratifiedSampler(
        dataset,
        n_out_split=1,
        n_in_split=0,
        valid_size=0,
        test_size=HOLDOUT_SIZE,
        random_state=SEED
    )

    masks = rss()
    learning_idx, holdout_idx = masks[0][MaskType.TRAIN], masks[0][MaskType.TEST]
    learning_df, holdout_df = df.iloc[learning_idx, :], df.iloc[holdout_idx, :]

    # ----------------------------------------------------------------------------------------------------------- #
    #                                              Saving DataFrames                                              #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df.to_csv(LEARNING_TABLE_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_TABLE_PATH, index=False)
