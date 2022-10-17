from json import dump
from os.path import join

from constants import *
import pandas as pd

from src.data.datasets.table_dataset import TableDataset
from src.data.processing.sampling import RandomStratifiedSampler


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
    sampler = RandomStratifiedSampler(
        dataset=table_dataset,
        n_out_split=2,
        n_in_split=2,
        random_state=SEED
    )

    masks = sampler()
    sampler.visualize_splits(masks)

    # Mask saving
    with open(join(MASKS_PATH, "masks.json"), "w") as file:
        dump(masks, file, indent=True)
