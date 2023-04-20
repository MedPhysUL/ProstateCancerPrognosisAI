"""
    @file:              07_tune_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to tune an MLP model.
"""

import env_apps
from optuna.samplers import TPESampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch.prediction import MLP
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.containers import (
    HyperparameterList
)
from src.tuning.hyperparameters.optuna import (
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

    dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

    search_algo = SearchAlgorithm(
        sampler=TPESampler(
            n_startup_trials=20,
            n_ei_candidates=20,
            multivariate=True,
            constant_liar=True,
            seed=SEED
        )
    )

    tuner = Tuner(
        search_algorithm=search_algo,
        recorder=TuningRecorder(),
        n_trials=100,
        seed=SEED
    )

    model_hyperparameter = TorchModelHyperparameter(
        constructor=MLP,
        parameters={
            "activation": FixedHyperparameter(name="activation", value="PReLU"),
            "hidden_channels": HyperparameterList(
                [
                    IntegerHyperparameter(name="layer_1", low=5, high=30),
                    IntegerHyperparameter(name="layer_2", low=5, high=30),
                    IntegerHyperparameter(name="layer_3", low=5, high=30)
                ]
            ),
            "dropout": FloatHyperparameter(name="dropout", low=0, high=0.25)
        }
    )

    learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            parameters={"lr": FloatHyperparameter(name="lr", low=0.0001, high=0.01)}
        ),
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": 0.999}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            parameters={"lambda_": FloatHyperparameter(name="alpha", low=0.0001, high=0.01)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        n_epochs=50,
        checkpoint=CheckpointHyperparameter()
    )

    train_methode_hyperparameter = TrainMethodHyperparameter(
        model=model_hyperparameter,
        learning_algorithms=learning_algorithm_hyperparameter
    )

    objective = TorchObjective(
        trainer_hyperparameter=trainer_hyperparameter,
        train_method_hyperparameter=train_methode_hyperparameter
    )

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=2, l=2)

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks
    )
