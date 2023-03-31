"""
    @file:              06_tune_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to tune an MLP model.
"""

import env_apps

from delia.databases import PatientsDatabase
from optuna.samplers import TPESampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch.extraction import CNN
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.optuna import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter
)
from src.tuning.hyperparameters.torch import (
    CheckpointHyperparameter,
    CriterionHyperparameter,
    EarlyStopperHyperparameter,
    LearningAlgorithmHyperparameter,
    LRSchedulerHyperparameter,
    OptimizerHyperparameter,
    RegularizerHyperparameter,
    TorchModelHyperparameter,
    TrainerHyperparameter,
    TrainMethodHyperparameter
)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, "midl2023_learning_table.csv"))

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR, BCR_TIME]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=PN_TASK,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    database = PatientsDatabase(path_to_database=r"local_data/midl2023_learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT"},
        organs={"CT": {"Prostate"}}
    )

    # Creation of the dataset
    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    search_algo = SearchAlgorithm(
        sampler=TPESampler(
            multivariate=True,
            constant_liar=True,
            seed=SEED
        )
    )

    tuner = Tuner(
        search_algorithm=search_algo,
        recorder=TuningRecorder(save_descriptive_analysis=True),
        n_trials=50,
        seed=SEED
    )

    model_hyperparameter = TorchModelHyperparameter(
        constructor=CNN,
        parameters={
            "in_shape": FixedHyperparameter(name="in_shape", value=(1, 96, 96, 96)),
            "n_features": FixedHyperparameter(name="n_features", value=10),
            "channels": CategoricalHyperparameter(
                name="channels",
                choices=["(2, 4, 8, 16)", "(2, 4, 8, 16, 32)"]
            ),
            "kernel_size": FixedHyperparameter(name="kernel_size", value=3),
            "num_res_units": FixedHyperparameter(name="num_res_units", value=3),
            "dropout": FixedHyperparameter(name="dropout", value=0.8)
        }
    )

    learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            parameters={
                "lr": FloatHyperparameter(name="lr", low=1e-6, high=1e-4),
                "weight_decay": CategoricalHyperparameter(name="weight_decay", choices=[0.001, 0.01, 0.1])
            }
        ),
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": FixedHyperparameter(name="gamma", value=0.99)}
        ),
        regularizer=RegularizerHyperparameter(
           constructor=L2Regularizer,
           parameters={"lambda_": CategoricalHyperparameter(name="alpha", choices=[0.001, 0.01])}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        n_epochs=100,
        checkpoint=CheckpointHyperparameter(save_freq=5)
    )

    train_methode_hyperparameter = TrainMethodHyperparameter(
        model=model_hyperparameter,
        learning_algorithms=learning_algorithm_hyperparameter
    )

    objective = TorchObjective(
        trainer_hyperparameter=trainer_hyperparameter,
        train_method_hyperparameter=train_methode_hyperparameter
    )

    masks = extract_masks(os.path.join(MASKS_PATH, "midl2023_masks.json"), k=5, l=5)

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks
    )
