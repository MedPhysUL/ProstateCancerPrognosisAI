"""
    @file:              12_tune_unextractor.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This script is used to tune a UNEXtractor model.
"""

import env_apps

import os

from delia.databases import PatientsDatabase
from monai.transforms import (
    Compose,
    RandGaussianNoiseD,
    RandFlipD,
    RandRotateD,
    ThresholdIntensityD
)
from optuna.integration.botorch import BoTorchSampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    EXPERIMENTS_PATH,
    EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT,
    ID,
    LEARNING_SET_PATH,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    PROSTATE_SEGMENTATION_TASK,
    SEED,
    TABLE_TASKS,
    UNEXTRACTOR_LR_LOW_BOUND_DICT,
    UNEXTRACTOR_LR_HIGH_BOUND_DICT
)
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss, WeightedSumLoss
from src.models.torch import UNEXtractor
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.containers import (
    HyperparameterDict
)
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
    for task in TABLE_TASKS:
        df = pd.read_csv(LEARNING_TABLE_PATH)

        table_dataset = TableDataset(
            dataframe=df,
            ids_column=ID,
            tasks=task
        )

        database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

        image_dataset = ImageDataset(
            database=database,
            modalities={"CT", "PT"},
            tasks=PROSTATE_SEGMENTATION_TASK,
            augmentations=Compose([
                RandGaussianNoiseD(keys=["CT", "PT"], prob=0.5, std=0.05),
                ThresholdIntensityD(keys=["CT", "PT"], threshold=0, above=True, cval=0),
                ThresholdIntensityD(keys=["CT", "PT"], threshold=1, above=False, cval=1),
                RandFlipD(keys=["CT", "PT", "CT_Prostate"], prob=0.5, spatial_axis=2),
                RandRotateD(
                    keys=["CT", "PT", "CT_Prostate"],
                    mode=["bilinear", "bilinear", "nearest"],
                    prob=0.5,
                    range_x=0.174533
                )
            ]),
            seed=SEED
        )

        dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

        path_to_record_folder = os.path.join(
            EXPERIMENTS_PATH,
            f"{task.target_column}(UNEXtractor - Deep radiomics)"
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
            recorder=TuningRecorder(
                path_to_record_folder=path_to_record_folder,
                save_inner_splits_best_models=True
            ),
            n_trials=25,
            seed=SEED
        )

        model_hyperparameter = TorchModelHyperparameter(
            constructor=UNEXtractor,
            parameters={
                "image_keys": ["CT", "PT"],
                "channels": FixedHyperparameter(name="channels", value=(64, 128, 256, 512, 1024)),
                "kernel_size": FixedHyperparameter(name="kernel_size", value=3),
                "num_res_units": FixedHyperparameter(name="num_res_units", value=2),
                "dropout_cnn": FloatHyperparameter(name="dropout_cnn", low=0.0, high=0.6),
                "dropout_fnn": FloatHyperparameter(name="dropout_fnn", low=0.0, high=0.3),
                "bayesian": True,
                "temperature": HyperparameterDict({
                    task.name: FloatHyperparameter(name=f"temp{task.target_column}", low=1e-4, high=1e-1, log=True),
                    PROSTATE_SEGMENTATION_TASK.name: FixedHyperparameter(name="tempPROSTATE", value=1e-4)
                }),
                "standard_deviation": 0.01
            }
        )

        learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
            criterion=CriterionHyperparameter(
                constructor=WeightedSumLoss,
                parameters={
                    "tasks": [task, PROSTATE_SEGMENTATION_TASK],
                    "weights": CategoricalHyperparameter(
                        name="weights",
                        choices=[
                            "(0.25, 0.75)", "(0.333, 0.667)", "(0.5, 0.5)", "(0.667, 0.333)", "(0.75, 0.25)"
                        ]
                    )
                }
            ),
            optimizer=OptimizerHyperparameter(
                constructor=Adam,
                model_params_getter=lambda model: model.parameters(),
                parameters={
                    "lr": FloatHyperparameter(
                        name="lr",
                        low=UNEXTRACTOR_LR_LOW_BOUND_DICT[task.target_column],
                        high=UNEXTRACTOR_LR_HIGH_BOUND_DICT[task.target_column],
                        log=True
                    ),
                    "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-3, high=1e-1, log=True)
                }
            ),
            clip_grad_max_norm=EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT[task.target_column],
            early_stopper=EarlyStopperHyperparameter(
                constructor=MultiTaskLossEarlyStopper,
                parameters={
                    "criterion": MeanLoss(tasks=task),
                    "patience": 20
                }
            ),
            lr_scheduler=LRSchedulerHyperparameter(
                constructor=ExponentialLR,
                parameters={"gamma": FixedHyperparameter(name="gamma", value=0.95)}
            ),
            regularizer=RegularizerHyperparameter(
                constructor=L2Regularizer,
                model_params_getter=lambda model: model.named_parameters(),
                parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-4, high=1e-2, log=True)}
            )
        )

        trainer_hyperparameter = TrainerHyperparameter(
            batch_size=16,
            n_epochs=100,
            verbose=False
            # checkpoint=CheckpointHyperparameter(save_freq=20)
        )

        train_method_hyperparameter = TrainMethodHyperparameter(
            model=model_hyperparameter,
            learning_algorithms=learning_algorithm_hyperparameter
        )

        objective = TorchObjective(
            trainer_hyperparameter=trainer_hyperparameter,
            train_method_hyperparameter=train_method_hyperparameter
        )

        masks = extract_masks(MASKS_PATH, k=5, l=5)

        tuner.tune(
            objective=objective,
            dataset=dataset,
            masks=masks
        )
