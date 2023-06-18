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

from constants import *
from src.evaluation.single_task.model_evaluator import ModelEvaluator
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import MLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv("/Users/felixdesroches/Downloads/fake_dataset_2.csv")

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_features=CONTINUOUS_FEATURES,
        cat_features=CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = {0: [x for x in range(30)]}

    dataset.update_masks(
        train_mask=masks[0][:len(masks[0])//3],
        test_mask=masks[0][len(masks[0])//3:2*len(masks[0])//3],
        valid_mask=masks[0][2*len(masks[0])//3:3*len(masks[0])//3]
    )
    SEED = 111211211
    model = MLP(
        multi_task_mode="separated",
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cpu"),
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
                # early_stopper=MultiTaskLossEarlyStopper(patience=10),
                regularizer=L2Regularizer(task_specific_model.named_parameters(), lambda_=0.01)
            )
        )

    trainer = Trainer(
        batch_size=16,
        # checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=20,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithms
    )
    history.plot(show=True)
    evaluator = ModelEvaluator(model=model, dataset=dataset, mask=dataset.valid_mask)

    print(evaluator.compute_metrics())
    evaluator.plot_classification_task_curves(True)
    evaluator.plot_survival_analysis_curves(True)

