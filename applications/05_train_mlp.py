"""
    @file:              05_train_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train an mlp model.
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import MLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR, BCR_TIME, METASTASIS, METASTASIS_TIME, EE, SVI, CRPC, CRPC_TIME, DEATH, DEATH_TIME]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=2, l=2)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = MLP(
        multi_task_mode="separated",
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=2e-4,
        weight_decay=0.02
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.01)
    )

    trainer = Trainer(
        batch_size=8,
        checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=50,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)

    # The next part will be integrated in an evaluation tool in the near future.
    model.fix_thresholds_to_optimal_values(dataset)
    score = model.score_on_dataset(dataset, dataset.test_mask)
    print(score)

    for task in dataset.tasks.survival_analysis_tasks:
        breslow_estimator = task.breslow_estimator

        plt.plot(breslow_estimator.unique_times_)
        plt.show()

        cum_baseline_hazard = breslow_estimator.cum_baseline_hazard_
        plt.plot(cum_baseline_hazard.x, cum_baseline_hazard.y)
        plt.show()

        baseline_survival = breslow_estimator.baseline_survival_
        plt.plot(baseline_survival.x, baseline_survival.y)
        plt.show()

        prediction = trained_model.predict_on_dataset(dataset, [0, 1, 2, 3])[task.name].cpu()
        chf_funcs = breslow_estimator.get_cumulative_hazard_function(prediction)
        for fn in chf_funcs:
            plt.step(fn.x, fn(fn.x), where="post")
        plt.show()

        chf_funcs = breslow_estimator.get_survival_function(prediction)
        for fn in chf_funcs:
            plt.step(fn.x, fn(fn.x), where="post")
        plt.show()
