"""
    @file:              create_holdout_set.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Small script used to create the holdout set.
"""

import os

import pandas as pd

from src.data.processing.dataset import MaskType, ProstateCancerDataset
from src.data.processing.sampling import RandomStratifiedSampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                              Constants                                                      #
    # ----------------------------------------------------------------------------------------------------------- #
    CLINICAL_DATA_PATH = "local_data/clinical_data.xlsx"
    SHEET_NAME = "sheet1"
    IDS_COL = "ID"
    CONT_COLS = ["AGE", "PSA"]
    CAT_COLS = ["GLEASON_GLOBAL", "GLEASON_PRIMARY", "GLEASON_SECONDARY", "CLINICAL_STAGE"]
    TARGETS_COLS = ["PN", "BCR"]

    DATA_FOLDER_NAME = "local_data/database"

    HOLDOUT_SIZE = 0.24096
    SEED = 1010710

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_excel(io=CLINICAL_DATA_PATH, sheet_name=SHEET_NAME, header=1)
    df = df[df[IDS_COL].isin(os.listdir(DATA_FOLDER_NAME))]
    target_df = df[TARGETS_COLS]

    dataset = ProstateCancerDataset(
        df=df,
        ids_col=IDS_COL,
        target_col="PN",
        cont_cols=CONT_COLS,
        cat_cols=CAT_COLS
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

    masks = rss(stratify=target_df.transpose().to_numpy())
    learning_idx, hold_out_idx = masks[0][MaskType.TRAIN], masks[0][MaskType.TEST]
    learning_df, hold_out_df = df.iloc[learning_idx, :], df.iloc[hold_out_idx, :]

    print(hold_out_df)
