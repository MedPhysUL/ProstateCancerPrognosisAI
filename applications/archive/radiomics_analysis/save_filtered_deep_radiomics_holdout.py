"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 07/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import json
import os

from delia.databases import PatientsDatabase
from monai.data import DataLoader
import numpy as np
import pandas as pd
import torch

from constants import (
    BCR_TASK,
    DEEP_RADIOMICS_HOLDOUT_PATH,
    ID,
    HOLDOUT_MASKS_PATH,
    LEARNING_TABLE_PATH,
    LEARNING_SET_PATH,
    PROSTATE_SEGMENTATION_TASK,
    SEED
)
from src.data.processing.sampling import Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch import UNEXtractor


if __name__ == "__main__":
    df = pd.read_csv(LEARNING_TABLE_PATH)

    os.makedirs(DEEP_RADIOMICS_HOLDOUT_PATH, exist_ok=True)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=BCR_TASK
    )

    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))
    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        valid_mask=masks[Mask.VALID]
    )

    model = UNEXtractor(
        image_keys=["CT", "PT"],
        num_res_units=3,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    loaded_state = torch.load(
        r"local_data\records\experiments\HOLDOUT\BCR(UNEXtractor - Deep radiomics)\best_model_checkpoint.pt"
    )["model_state"]

    model.load_state_dict(loaded_state)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=None)
    rads = []
    for features, _ in data_loader:
        rads.append(model.extract_radiomics(features).deep_features.cpu().detach().numpy())
    rads = np.array(rads)[:, 0, :]

    dataframe = df.copy()
    for i in range(6):
        dataframe[f"RADIOMIC_{i + 1}"] = rads[:, i]

    dataframe.to_csv(os.path.join(DEEP_RADIOMICS_HOLDOUT_PATH, f"learning.csv"), index=False)
