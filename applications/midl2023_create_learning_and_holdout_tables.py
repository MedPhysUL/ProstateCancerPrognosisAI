import pandas as pd

from constants import *
from src.data.datasets import TableDataset
from src.data.processing.sampling import Mask, Sampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_csv(LEARNING_TABLE_PATH)

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
    learning_df.to_csv(os.path.join(DATA_PATH, "midl2023_learning_table.csv"), index=False)
    holdout_df.to_csv(os.path.join(DATA_PATH, "midl2023_holdout_table.csv"), index=False)
