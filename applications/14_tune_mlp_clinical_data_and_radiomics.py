"""
    @file:              14_tune_mlp_clinical_data_and_radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This script is used to tune an MLP model.
"""

import env_apps

from typing import Dict, Union

from optuna.integration.botorch import BoTorchSampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ProstateCancerDataset, TableDataset, Split
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch.prediction import MLP
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


def get_dataframes_dictionary(
        path_to_dataframes_folder: str
) -> Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]:
    """
    This function returns a dictionary of dataframes. The keys of the first level are the outer split indices. The keys
    of the second level are the split indices. The values are the dataframes.

    Parameters
    ----------
    path_to_dataframes_folder : str
        The path to the folder containing the dataframes.

    Returns
    -------
    dataframes_dict : Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]
        The dictionary of dataframes.
    """
    n_outer_splits = len(os.listdir(path_to_dataframes_folder))

    dataframes = {}
    for k in range(n_outer_splits):
        path_to_outer_split_folder = os.path.join(path_to_dataframes_folder, f"outer_split_{k}")

        dataframes[k] = {}
        dataframes[k][str(Split.OUTER)] = pd.read_csv(os.path.join(path_to_outer_split_folder, "outer_split.csv"))
        dataframes[k][str(Split.INNER)] = {}

        path_to_inner_splits_folder = os.path.join(path_to_outer_split_folder, "inner_splits")
        n_inner_splits = len(os.listdir(path_to_inner_splits_folder))
        for l in range(n_inner_splits):
            path_to_inner_split = os.path.join(path_to_inner_splits_folder, f"inner_split_{l}.csv")
            dataframes[k][str(Split.INNER)][l] = pd.read_csv(path_to_inner_split)

    return dataframes


if __name__ == '__main__':
    LR_HIGH_BOUND_DICT = {BCR: 1e-2, CRPC: 5e-3, DEATH: 5e-3, HTX: 1e-2, METASTASIS: 5e-3, PN: 1e-2}

    for task in TABLE_TASKS:
        df_dict = get_dataframes_dictionary(path_to_dataframes_folder=os.path.join(RADIOMICS_PATH, task.target_column))

        table_dataset = TableDataset(
            dataframe=df_dict[0][str(Split.OUTER)],  # Dummy dataset - Never used
            ids_column=ID,
            tasks=task,
            continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS_FEATURES,
            categorical_features=CLINICAL_CATEGORICAL_FEATURES
        )

        dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

        path_to_record_folder = os.path.join(
            EXPERIMENTS_PATH,
            f"{task.target_column}(MLP - Clinical data and radiomics)"
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
            constructor=MLP,
            parameters={
                "activation": FixedHyperparameter(name="activation", value="PReLU"),
                "n_hidden_layers": IntegerHyperparameter(name="n_hidden_layers", low=1, high=3),
                "n_hidden_neurons": IntegerHyperparameter(name="n_hidden_neurons", low=20, high=40, step=10),
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
                    "lr": FloatHyperparameter(
                        name="lr",
                        low=1e-4,
                        high=LR_HIGH_BOUND_DICT[task.target_column],
                        log=True
                    ),
                    "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-4, high=1e-2, log=True)
                }
            ),
            clip_grad_max_norm=PREDICTOR_CLIP_GRAD_MAX_NORM_DICT[task.target_column],
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

        masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

        tuner.tune(
            objective=objective,
            dataset=dataset,
            masks=masks,
            dataframes=df_dict
        )
