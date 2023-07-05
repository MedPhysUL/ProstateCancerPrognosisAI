
import env_apps

import os

from delia.databases import PatientsDatabase
import numpy as np
import torch

from constants import (
    AUTOMATIC_RADIOMICS_MODELS_PATH,
    LEARNING_SET_PATH,
    MASKS_PATH,
    PROSTATE_SEGMENTATION_TASK,
    SEED
)
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.data.processing.sampling import extract_masks, Mask
from src.models.torch.segmentation import Unet


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    scores = []
    for k in range(5):
        model_state = torch.load(
            os.path.join(AUTOMATIC_RADIOMICS_MODELS_PATH, f"outer_split_{k}", "model.pth")
        )["model_state"]

        dataset.update_masks(
            train_mask=masks[k][Mask.TRAIN],
            test_mask=masks[k][Mask.TEST],
            valid_mask=masks[k][Mask.VALID]
        )

        model = Unet(
            image_keys="CT",
            spatial_dims=3,
            num_res_units=3,
            dropout=0.2,
            device=torch.device("cuda"),
            seed=SEED
        ).build(dataset)

        model.load_state_dict(model_state)

        score = model.compute_score_on_dataset(dataset, dataset.test_mask)
        scores.append(list(list(score.values())[0].values())[0])

    print(scores)
    print(np.mean(scores))
    print(np.std(scores))
