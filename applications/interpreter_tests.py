"""
    @file:              06_train_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train an mlp model. Each task has its own mlp and its own optimizer.
"""

import env_apps
import copy

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.explainer.shap_explainer import TableShapValueExplainer, CaptumWrapper
from src.explainer.survshap_explainer import  TableSurvshapExplainer
from src.models.torch.prediction import MLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer

from src.evaluation.model_comparator import ModelComparator


if __name__ == '__main__':
    df = pd.read_csv("/Users/felixdesroches/Downloads/fake_dataset_2-1.csv")

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = [i for i in range(len(table_dataset))]

    dataset.update_masks(
        train_mask=masks[:(len(masks)//3)],
        test_mask=masks[(len(masks)//3):(2*len(masks)//3)],
        valid_mask=masks[(2*len(masks)//3):(len(masks))]
    )

    model = MLP(
        multi_task_mode="separated",
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cpu"),
        seed=SEED
    ).build(dataset)
    model_backup = copy.deepcopy(model)

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
        n_epochs=10,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithms
    )

    # history.plot(show=True)

    show = True
    # explainer = TableShapValueExplainer(model, dataset)
    # explainer.plot_waterfall(0, 0)
    # explainer.plot_beeswarm(0)
    explainer = TableSurvshapExplainer(model=trained_model, dataset=dataset, mask=(5, 10))
    explainer.plot_shap_average_of_absolute_value(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=True, show=True, tasks=BCR_TASK, path_to_save_folder=None)
    # explainer.plot_shap_average_of_absolute_value(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=False, show=True, tasks=BCR_TASK, path_to_save_folder=None)
    # explainer.plot_shap_lines_for_all_patients(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=True, show=True, tasks=BCR_TASK, path_to_save_folder=None)
    # explainer.plot_shap_lines_for_features(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=True, show=True, tasks=BCR_TASK, path_to_save_folder=None)
    # explainer.plot_shap_lines_for_all_patients(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=False, show=True, tasks=BCR_TASK, path_to_save_folder=None)
    # explainer.plot_shap_lines_for_features(features={("PSA", "AGE"): [0, 1, 2, 3, 4, 4]}, function='sf', normalize=False, show=True, tasks=BCR_TASK,  path_to_save_folder=None)
    # comparator = ModelComparator(model_backup, trained_model, dataset)
    # print(comparator.compute_c_index_p_value())
    # print(comparator.compute_auc_p_value())
    # print(comparator.compute_variance_p_value())
    # print(comparator.compute_mean_p_value())
