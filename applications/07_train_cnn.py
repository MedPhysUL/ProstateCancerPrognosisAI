"""
    @file:              07_train_cnn.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a cnn model.
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
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    ID,
    LEARNING_SET_PATH,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    SEED,
    TABLE_TASKS,
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch.extraction import CNN
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS
    )

    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"},
        augmentations=Compose([
            RandGaussianNoiseD(keys=["CT", "PT"], prob=0.5, std=0.05),
            ThresholdIntensityD(keys=["CT", "PT"], threshold=0, above=True, cval=0),
            ThresholdIntensityD(keys=["CT", "PT"], threshold=1, above=False, cval=1),
            RandFlipD(keys=["CT", "PT"], prob=0.5, spatial_axis=2),
            RandRotateD(keys=["CT", "PT"], prob=0.5, range_x=0.174533)
        ]),
        seed=SEED
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = CNN(
        image_keys=["CT", "PT"],
        dropout_cnn=0.4,
        dropout_fnn=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.02)
    )
    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(),
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
