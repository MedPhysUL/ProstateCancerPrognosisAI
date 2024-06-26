"""
    @file:              06_train_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train an mlp model. Each task has its own mlp and its own optimizer.
"""

import env_apps

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    SEED,
    TABLE_TASKS
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.models.torch.prediction import MLP
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
        tasks=TABLE_TASKS,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = MLP(
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    learning_algorithms = []
    for task in TABLE_TASKS:
        task_specific_model = model.predictor[task.name]

        optimizer = Adam(
            params=task_specific_model.parameters(),
            lr=2e-4,
            weight_decay=0.02
        )

        learning_algorithms.append(
            LearningAlgorithm(
                criterion=MeanLoss(tasks=task),
                optimizer=optimizer,
                lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
                early_stopper=MultiTaskLossEarlyStopper(patience=20),
                regularizer=L2Regularizer(task_specific_model.named_parameters(), lambda_=0.01)
            )
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
        learning_algorithms=learning_algorithms
    )

    history.plot(show=True)

    evaluator = ModelEvaluator(model=trained_model, dataset=dataset)
    score = evaluator.compute_score(dataset.test_mask)
    print(score)

    evaluator.plot_binary_classification_task_curves(show=True)
    evaluator.plot_survival_analysis_task_curves(show=True)
