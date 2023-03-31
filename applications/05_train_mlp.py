"""
    @file:              05_train_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to train an mlp model.
"""

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction.mlp import MLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import L2Regularizer
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR, BCR_TIME]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

    dataset.update_masks(
        train_mask=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        test_mask=[],
        valid_mask=[20, 21, 22, 23, 24]
    )

    model = MLP(
        activation="PRELU",
        hidden_channels=[20, 20, 20],
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-2
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.999),
        regularizer=L2Regularizer(params=model.named_parameters(), lambda_=0.01),
        early_stopper=MultiTaskLossEarlyStopper()
    )
    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=30,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)
