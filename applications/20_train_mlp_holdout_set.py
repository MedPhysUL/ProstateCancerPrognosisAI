"""
    @file:              20_train_mlp_holdout_set.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train an mlp model. Each task has its own mlp, but the model is trained
                        using a single MeanLoss on all tasks.
"""

import env_apps

import json

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    ID,
    HOLDOUT_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    LEARNING_TABLE_PATH,
    SEED,
    TABLE_TASKS
)
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.models.torch.prediction import MLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))

    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        test_mask=masks[Mask.TEST],
        valid_mask=masks[Mask.VALID]
    )

    model = MLP(
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=0.02
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=10),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.006)
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

    evaluator = ModelEvaluator(model=trained_model, dataset=dataset)
    score = evaluator.compute_score(dataset.valid_mask)
    print(score)

    # evaluator.plot_binary_classification_task_curves(show=True)
    # evaluator.plot_survival_analysis_task_curves(show=True)
