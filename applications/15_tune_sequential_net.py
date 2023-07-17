"""
    @file:              15_tune_sequential_net.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This script is used to tune a SequentialNet model.
"""

import env_apps

import os

from optuna.integration.botorch import BoTorchSampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    BCR,
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    CRPC,
    DEATH,
    EXPERIMENTS_PATH,
    HTX,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    METASTASIS,
    PN,
    PREDICTOR_CLIP_GRAD_MAX_NORM_DICT,
    SEED,
    TABLE_TASKS
)
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch.prediction import SequentialNet
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.optuna import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter
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
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

    path_to_record_folder = os.path.join(
        EXPERIMENTS_PATH,
        f"MULTI-TASK(SequentialNet - Clinical data only)"
    )

    search_algo = SearchAlgorithm(
        sampler=BoTorchSampler(
            n_startup_trials=5,
            seed=SEED
        ),
        storage="sqlite:///" + os.path.join(path_to_record_folder, "tuning_history.db")
    )

    tuner = Tuner(
        search_algorithm=search_algo,
        recorder=TuningRecorder(path_to_record_folder=path_to_record_folder),
        n_trials=25,
        seed=SEED
    )

    model_hyperparameter = TorchModelHyperparameter(
        constructor=SequentialNet,
        parameters={
            "sequence": [PN, BCR, METASTASIS, HTX, CRPC, DEATH],
            "activation": FixedHyperparameter(name="activation", value="PReLU"),
            "hidden_channels": CategoricalHyperparameter(
                name="hidden_channels",
                choices=["(10, 10, 10)", "(20, 20, 20)", "(30, 30, 30)"]
            ),
            "dropout": FloatHyperparameter(name="dropout", low=0.05, high=0.25)
        }
    )

    learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            parameters={
                "lr": FloatHyperparameter(name="lr", low=1e-4, high=1e-2, log=True),
                "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-4, high=1e-2, log=True)
            }
        ),
        clip_grad_max_norm=3.0,
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
            parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-5, high=1e-2, log=True)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        batch_size=16,
        n_epochs=100
        # checkpoint=CheckpointHyperparameter(save_freq=20)
    )

    train_methode_hyperparameter = TrainMethodHyperparameter(
        model=model_hyperparameter,
        learning_algorithms=learning_algorithm_hyperparameter
    )

    objective = TorchObjective(
        trainer_hyperparameter=trainer_hyperparameter,
        train_method_hyperparameter=train_methode_hyperparameter
    )

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks
    )
