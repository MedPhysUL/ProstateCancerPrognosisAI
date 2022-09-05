"""
    @file:              04_test.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to generate a folder containing descriptive analyses of all data.
"""

from monai.data import DataLoader
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    KeepLargestConnectedComponentd,
    ToTensord
)
import pandas as pd
import torch

from src.data.datasets.table_dataset import TableDataset
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.data.extraction.local import LocalDatabaseManager

from constants import *


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    table_dataset = TableDataset(
        df=learning_df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    # Defining Transforms
    transforms = Compose([
        AddChanneld(keys=["CT"] + [task.organ for task in IMAGE_TASKS]),
        CenterSpatialCropd(keys=["CT"] + [task.organ for task in IMAGE_TASKS], roi_size=(1000, 160, 160)),
        KeepLargestConnectedComponentd(keys=[task.organ for task in IMAGE_TASKS]),
        ToTensord(keys=["CT"] + [task.organ for task in IMAGE_TASKS], dtype=torch.float32)
    ])

    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(path_to_database=r"D:/ProstateCancerData/learning_set.h5"),
        tasks=IMAGE_TASKS,
        modalities=MODALITIES,
        transforms=transforms
    )

    prostate_cancer_dataset = ProstateCancerDataset(
        image_dataset=image_dataset,
        table_dataset=table_dataset
    )

    data_loader = DataLoader(
        dataset=prostate_cancer_dataset,
        batch_size=2,
        pin_memory=True,
        shuffle=False
    )

    for batch in data_loader:
        print(type(batch.x))
        print(batch.x["CT"].shape)
        print(type(batch.y))
        print(batch.y.keys())

    data_loader = DataLoader(
        dataset=prostate_cancer_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    for batch in data_loader:
        print(type(batch.x))
        print(batch.x["CT"].shape)
        print(type(batch.y))
        print(batch.y["Prostate"].shape)
