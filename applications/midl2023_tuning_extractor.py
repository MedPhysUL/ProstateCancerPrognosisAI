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
from src.models.extraction.deep_radiomics_extractor import DeepRadiomicsExtractor
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
        constructor=DeepRadiomicsExtractor,
        parameters={
            "in_shape": CategoricalHyperparameter(name="in_shape", choices=["(1, 96, 96, 96)", "(2, 96, 96, 96)"]),
            "n_radiomics": FixedHyperparameter(name="n_radiomics", value=10),
            "channels": FixedHyperparameter(name="channels", value=(4, 8, 16, 32, 64)),
            "kernel_size": FixedHyperparameter(name="kernel_size", value=3),
            "num_res_units": IntegerHyperparameter(name="num_res_units", low=1, high=3),
            "dropout": FloatHyperparameter(name="dropout", low=0.1, high=0.3)
        }
    )

    learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            parameters={
                "lr": FloatHyperparameter(name="lr", low=5e-6, high=3e-5),
                "weight_decay": FloatHyperparameter(name="weight_decay", low=0.1, high=0.3)
            }
        ),
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 10}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": CategoricalHyperparameter(name="gamma", choices=[0.9, 0.99])}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            parameters={"lambda_": FloatHyperparameter(name="alpha", low=0.1, high=0.3)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        n_epochs=75,
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
