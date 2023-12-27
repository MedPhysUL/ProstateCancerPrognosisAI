import env_apps

from typing import Any, Dict, Union
import os

from optuna.integration.botorch import BoTorchSampler
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    BCR_TASK,
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    CRPC_TASK,
    DEATH_TASK,
    EXPERIMENTS_PATH,
    HTX_TASK,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    METASTASIS_TASK,
    PN_TASK,
    PREDICTOR_CLIP_GRAD_MAX_NORM_DICT,
    SEED,
    TABLE_TASKS,
    DEEP_BAYESIAN_FILTERED_RADIOMICS_PATH,
    MULTITASK_RADIOMICS_PATH
)
from src.data.datasets import Feature, ProstateCancerDataset, TableDataset, Split
from src.data.processing.sampling import extract_masks
from src.data.transforms import Normalization
from src.losses.multi_task import MeanLoss
from src.models.torch.prediction import SequentialNet, ModelConfig
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.containers import (
    HyperparameterDict
)
from src.tuning.hyperparameters.optuna import (
    IntegerHyperparameter,
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

def get_dataframes_dictionary(
        path_to_dataframes_folder: str,
        n_outer_loops: int,
        n_inner_loops: int
) -> Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]:
    """
    This function returns a dictionary of dataframes. The keys of the first level are the outer split indices. The keys
    of the second level are the split indices. The values are the dataframes.

    Parameters
    ----------
    path_to_dataframes_folder : str
        The path to the folder containing the dataframes.
    n_outer_loops : int
        The number of outer loops.
    n_inner_loops : int
        The number of inner loops.

    Returns
    -------
    dataframes_dict : Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]
        The dictionary of dataframes.
    """
    dataframes = {}
    for k in range(n_outer_loops):
        path_to_outer_split_folder = os.path.join(path_to_dataframes_folder, f"outer_split_{k}")

        dataframes[k] = {}
        dataframes[k][str(Split.OUTER)] = pd.read_csv(os.path.join(path_to_outer_split_folder, "outer_split.csv"))
        dataframes[k][str(Split.INNER)] = {}

        path_to_inner_splits_folder = os.path.join(path_to_outer_split_folder, "inner_splits")
        for l in range(n_inner_loops):
            path_to_inner_split = os.path.join(path_to_inner_splits_folder, f"inner_split_{l}.csv")
            dataframes[k][str(Split.INNER)][l] = pd.read_csv(path_to_inner_split)

    return dataframes


def get_model_configs_dictionary(
        path_to_outer_splits_folder: Dict[Any, str],
        n_outer_loops: int,
        n_inner_loops: int
) -> Dict[int, Dict[str, Union[Dict[str, ModelConfig], Dict[int, Dict[str, ModelConfig]]]]]:
    """
    This function returns a dictionary of model configurations. The keys of the first level are the outer split indices.
    The keys of the second level are the split indices. The values are the model configurations.

    Parameters
    ----------
    path_to_outer_splits_folder : Dict[Any, str]
        The path to the folder containing the outer splits.
    n_outer_loops : int
        The number of outer loops.
    n_inner_loops : int
        The number of inner loops.

    Returns
    -------
    model_configs_dict : Dict[int, Dict[str, Union[Dict[str, ModelConfig], Dict[int, Dict[str, ModelConfig]]]]]
        The dictionary of model configurations.
    """
    configs = {}
    for k in range(n_outer_loops):
        configs[k] = {}
        configs[k][str(Split.OUTER)] = {}
        configs[k][str(Split.INNER)] = {}
        for l in range(n_inner_loops):
            configs[k][str(Split.INNER)][l] = {}

        for task, path in path_to_outer_splits_folder.items():
            path_to_best_models = os.path.join(path, f"split_{k}", "best_models")

            state = torch.load(os.path.join(path_to_best_models, "outer_split", "best_model.pt"))
            state_copy = state.copy()
            if task == PN_TASK:
                for key in state.keys():
                    if key.startswith("predictor."):
                        state_copy[key.replace("predictor.", "")] = state[key]
                        del state_copy[key]
            else:
                for key in state.keys():
                    if not key.startswith(f"predictor.{task.name}"):
                        del state_copy[key]
                    else:
                        state_copy[key.replace(f"predictor.{task.name}.", "")] = state[key]
                        del state_copy[key]

            configs[k][str(Split.OUTER)][task.name] = ModelConfig(
                freeze=True,
                pretrained_model_state=state_copy
            )

            path_to_inner_splits_folder = os.path.join(path_to_best_models, "inner_splits")
            for l in range(n_inner_loops):
                path_to_inner_split = os.path.join(path_to_inner_splits_folder, f"split_{l}")

                state = torch.load(os.path.join(path_to_inner_split, "best_model.pt"))
                state_copy = state.copy()
                if task == PN_TASK:
                    for key in state.keys():
                        if key.startswith("predictor."):
                            state_copy[key.replace("predictor.", "")] = state[key]
                            del state_copy[key]
                else:
                    for key in state.keys():
                        if not key.startswith(f"predictor.{task.name}"):
                            del state_copy[key]
                        else:
                            state_copy[key.replace(f"predictor.{task.name}.", "")] = state[key]
                            del state_copy[key]

                configs[k][str(Split.INNER)][l][task.name] = ModelConfig(
                    freeze=True,
                    pretrained_model_state=state_copy
                )

    return configs


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

if __name__ == '__main__':
    df_dict = get_dataframes_dictionary(
        path_to_dataframes_folder=MULTITASK_RADIOMICS_PATH,
        n_outer_loops=5,
        n_inner_loops=5
    )

    config_dict = get_model_configs_dictionary(
        path_to_outer_splits_folder={
            PN_TASK: r"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\PN(MLP - Clinical data and automatic radiomics)\outer_splits",
            BCR_TASK: r"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\BCR(SeqNet - Clinical data and deep radiomics)\outer_splits",
            METASTASIS_TASK: r"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\METASTASIS(SeqNet - Clinical data only)\outer_splits",
            HTX_TASK: r"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\HTX(SeqNet - Clinical data only)\outer_splits"
        },
        n_outer_loops=5,
        n_inner_loops=5
    )

    table_dataset = TableDataset(
        dataframe=df_dict[0][str(Split.OUTER)],  # Dummy dataset - Never used
        ids_column=ID,
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

    path_to_record_folder = os.path.join(
        EXPERIMENTS_PATH,
        f"CRPC(SequentialNet0)"
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
        recorder=TuningRecorder(path_to_record_folder=path_to_record_folder, save_inner_splits_best_models=True),
        n_trials=25,
        seed=SEED
    )

    model_hyperparameter = TorchModelHyperparameter(
        constructor=SequentialNet,
        parameters={
            "sequence": [
                PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name
            ],
            "activation": FixedHyperparameter(name="activation", value="PReLU"),
            "n_layers": HyperparameterDict(
                {
                    PN_TASK.name: 3,
                    BCR_TASK.name: 2,
                    METASTASIS_TASK.name: 2,
                    HTX_TASK.name: 1,
                    CRPC_TASK.name: IntegerHyperparameter(
                        name="n_layers",
                        low=1,
                        high=3
                    )
                }
            ),
            "n_neurons": HyperparameterDict(
                {
                    PN_TASK.name: 15,
                    BCR_TASK.name: 10,
                    METASTASIS_TASK.name: 5,
                    HTX_TASK.name: 10,
                    CRPC_TASK.name: IntegerHyperparameter(
                        name="n_neurons",
                        low=5,
                        high=20,
                        step=5
                    )
                }
            ),
            "features_columns": {
                PN_TASK.name: [c.column for c in CLINICAL_CONTINUOUS_FEATURES] + [c.column for c in PN_RADIOMICS] + [
                    c.column for c in CLINICAL_CATEGORICAL_FEATURES],
                BCR_TASK.name: [c.column for c in CLINICAL_CONTINUOUS_FEATURES] + [c.column for c in BCR_RADIOMICS] + [
                    c.column for c in CLINICAL_CATEGORICAL_FEATURES],
                METASTASIS_TASK.name: [c.column for c in CLINICAL_CONTINUOUS_FEATURES] + [c.column for c in CLINICAL_CATEGORICAL_FEATURES],
                HTX_TASK.name: [c.column for c in CLINICAL_CONTINUOUS_FEATURES] + [c.column for c in CLINICAL_CATEGORICAL_FEATURES],
                CRPC_TASK.name: [c.column for c in CLINICAL_CONTINUOUS_FEATURES] + [c.column for c in CLINICAL_CATEGORICAL_FEATURES]
            },
            "dropout": HyperparameterDict(
                {
                    PN_TASK.name: 0.05,
                    BCR_TASK.name: 0.20517364273624022,
                    METASTASIS_TASK.name: 0.21000275742989755,
                    HTX_TASK.name: 0.2257450268562216,
                    CRPC_TASK.name: FloatHyperparameter(name="dropout", low=0.05, high=0.25)
                }
            )
        }
    )

    learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss,
            parameters={"tasks": CRPC_TASK}
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            model_params_getter=lambda model: model.parameters(),
            parameters={
                "lr": FloatHyperparameter(name="lr", low=1e-4, high=1e-2, log=True),
                "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-4, high=1e-2, log=True)
            }
        ),
        clip_grad_max_norm=PREDICTOR_CLIP_GRAD_MAX_NORM_DICT[CRPC_TASK.target_column],
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": 0.99}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            model_params_getter=lambda model: model.named_parameters(),
            parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-5, high=1e-2, log=True)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        batch_size=16,
        n_epochs=100
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
    masks = {k: v for k, v in masks.items() if k == 0}

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks,
        dataframes=df_dict,
        model_configs=config_dict
    )
