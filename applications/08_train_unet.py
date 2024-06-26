"""
    @file:              08_train_unet.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a unet model.
"""

import env_apps

from delia.databases import PatientsDatabase
from monai.transforms import (
    Compose,
    RandGaussianNoiseD,
    RandFlipD,
    RandRotateD,
    ThresholdIntensityD
)
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.data.processing.sampling import extract_masks, Mask
from src.models.torch.segmentation import Unet
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=PROSTATE_SEGMENTATION_TASK,
        augmentations=Compose([
            RandGaussianNoiseD(keys=["CT"], prob=0.5, std=0.05),
            ThresholdIntensityD(keys=["CT"], threshold=0, above=True, cval=0),
            ThresholdIntensityD(keys=["CT"], threshold=1, above=False, cval=1),
            RandFlipD(keys=["CT", "CT_Prostate"], prob=0.5, spatial_axis=2),
            RandRotateD(
                keys=["CT", "CT_Prostate"],
                mode=["bilinear", "nearest"],
                prob=0.5,
                range_x=0.174533
            )
        ]),
        seed=SEED
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = Unet(
        image_keys="CT",
        num_res_units=3,
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-3
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20)
    )
    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(save_freq=20),
        exec_metrics_on_train=True,
        n_epochs=100,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)
