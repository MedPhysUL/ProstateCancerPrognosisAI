"""
    @file:              19_train_sequential_net.py
    @Author:            Maxence Larose

    @Creation Date:     08/2023
    @Last modification: 08/2023

    @Description:       This script is used to train a sequential net.
"""

import env_apps

import os

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    SEED,
    BCR_TASK,
    PN_TASK
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.models.torch.prediction import SequentialNet, ModelConfig
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    RADIOMIC_1 = Feature(column="RADIOMIC_PN_1", transform=Normalization(), impute=False)
    RADIOMIC_2 = Feature(column="RADIOMIC_PN_2", transform=Normalization(), impute=False)
    RADIOMIC_3 = Feature(column="RADIOMIC_PN_3", transform=Normalization(), impute=False)
    RADIOMIC_4 = Feature(column="RADIOMIC_PN_4", transform=Normalization(), impute=False)
    RADIOMIC_5 = Feature(column="RADIOMIC_PN_5", transform=Normalization(), impute=False)
    RADIOMIC_6 = Feature(column="RADIOMIC_PN_6", transform=Normalization(), impute=False)
    RADIOMIC_7 = Feature(column="RADIOMIC_BCR_1", transform=Normalization(), impute=False)
    RADIOMIC_8 = Feature(column="RADIOMIC_BCR_2", transform=Normalization(), impute=False)
    RADIOMIC_9 = Feature(column="RADIOMIC_BCR_3", transform=Normalization(), impute=False)
    RADIOMIC_10 = Feature(column="RADIOMIC_BCR_4", transform=Normalization(), impute=False)
    RADIOMIC_11 = Feature(column="RADIOMIC_BCR_5", transform=Normalization(), impute=False)
    RADIOMIC_12 = Feature(column="RADIOMIC_BCR_6", transform=Normalization(), impute=False)

    PN_RADIOMICS = [RADIOMIC_1, RADIOMIC_2, RADIOMIC_3, RADIOMIC_4, RADIOMIC_5, RADIOMIC_6]
    BCR_RADIOMICS = [RADIOMIC_7, RADIOMIC_8, RADIOMIC_9, RADIOMIC_10, RADIOMIC_11, RADIOMIC_12]

    RADIOMICS = PN_RADIOMICS + BCR_RADIOMICS

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=[PN_TASK, BCR_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    state = torch.load(
        os.path.join(EXPERIMENTS_PATH, "PN(MLP - Clinical data and automatic radiomics - Holdout Set)", "model.pt")
    )

    state_copy = state.copy()
    for key in state.keys():
        if key.startswith("predictor."):
            state_copy[key.replace("predictor.", "")] = state[key]
            del state_copy[key]

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]
    model = SequentialNet(
        sequence=[PN_TASK.name, BCR_TASK.name],
        n_layers={PN_TASK.name: 3, BCR_TASK.name: 2},
        n_neurons={PN_TASK.name: 15, BCR_TASK.name: 10},
        features_columns={
            PN_TASK.name: cont_features + [c.column for c in PN_RADIOMICS] + cat_features,
            BCR_TASK.name: cont_features + [c.column for c in BCR_RADIOMICS] + cat_features
        },
        configs={
            PN_TASK.name:
                ModelConfig(
                    freeze=True,
                    pretrained_model_state=state_copy
                )
        },
        dropout=0.20517364273624022,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=0.00246607780843149,
        weight_decay=0.0031553306821593065
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(tasks=BCR_TASK),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        clip_grad_max_norm=3.0,
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.00046930823845460305)
    )

    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(path_to_checkpoint_folder=fr"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\checkpoints{i}"),
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
    score = evaluator.compute_score(dataset.train_mask)
    print(score)
    score = evaluator.compute_score(dataset.valid_mask)
    print(score)
    score = evaluator.compute_score(dataset.test_mask)
    print(score)

    # evaluator.plot_binary_classification_task_curves(show=True)
    # evaluator.plot_survival_analysis_task_curves(show=True)
