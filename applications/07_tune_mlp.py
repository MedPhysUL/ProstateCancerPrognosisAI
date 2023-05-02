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
    for task in [BCR_TASK, METASTASIS_TASK, CRPC_TASK, DEATH_TASK, PN_TASK, SVI_TASK, EE_TASK]:
        df = pd.read_csv(LEARNING_TABLE_PATH)

        table_dataset = TableDataset(
            df=df,
            ids_col=ID,
            tasks=task,
            cont_features=CONTINUOUS_FEATURES,
            cat_features=CATEGORICAL_FEATURES
        )

        dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

        path_to_record_folder = os.path.join(
            EXPERIMENTS_PATH,
            f"{task.target_column}(MLP - Clinical data only)"
        )

        search_algo = SearchAlgorithm(
            sampler=TPESampler(
                n_startup_trials=5,
                multivariate=True,
                seed=SEED
            ),
            storage="sqlite:///" + os.path.join(path_to_record_folder, "tuning_history.db")
        )

        tuner = Tuner(
            search_algorithm=search_algo,
            recorder=TuningRecorder(path_to_record_folder=path_to_record_folder),
            n_trials=30,
            seed=SEED
        )

        model_hyperparameter = TorchModelHyperparameter(
            constructor=MLP,
            parameters={
                "activation": FixedHyperparameter(name="activation", value="PReLU"),
                "hidden_channels": CategoricalHyperparameter(
                    name="hidden_channels",
                    choices=["(5, 5, 5)", "(10, 10, 10)", "(15, 15, 15)"]
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
                    "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-4, high=1e-1, log=True)
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
                parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-4, high=1e-2, log=True)}
            )
        )

        trainer_hyperparameter = TrainerHyperparameter(
            batch_size=8,
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

        masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=3)

        tuner.tune(
            objective=objective,
            dataset=dataset,
            masks=masks
        )
