"""
    @file:              constants.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file stores helpful constants.
"""

import os

from src.losses.single_task import BCEWithLogitsLoss, DiceLoss, NegativePartialLogLikelihood
from src.metrics.single_task import (
    AUC, BinaryBalancedAccuracy, ConcordanceIndexCensored, DiceMetric, Sensitivity, Specificity
)
from src.tasks import BinaryClassificationTask, SegmentationTask, SurvivalAnalysisTask

# SEED
SEED = 1010710

# SIZE
HOLDOUT_SIZE = 0.15

# PATHS
DATA_PATH = "local_data"
CLINICAL_DATA_PATH = os.path.join(DATA_PATH, "clinical_data.xlsx")
IMAGES_FOLDER_PATH = os.path.join(DATA_PATH, "Images")

LEARNING_TABLE_PATH = os.path.join(DATA_PATH, "learning_table.csv")
HOLDOUT_TABLE_PATH = os.path.join(DATA_PATH, "holdout_table.csv")

MASKS_PATH: str = os.path.join(DATA_PATH, "masks")

RECORDS_PATH = os.path.join("local_data", "records")
OUTLIERS_RECORDS_PATH = os.path.join(RECORDS_PATH, "outliers")
DESCRIPTIVE_ANALYSIS_PATH = os.path.join(RECORDS_PATH, "descriptive_analyses")
EXPERIMENTS_PATH = os.path.join(RECORDS_PATH, "experiments")
CHECKPOINTS_PATH = os.path.join(EXPERIMENTS_PATH, "checkpoints")

# ID COLUMN
ID = "ID"

# FEATURE COLUMNS
AGE = "AGE"
PSA = "PSA"
GLEASON_GLOBAL = "GLEASON_GLOBAL"
GLEASON_PRIMARY = "GLEASON_PRIMARY"
GLEASON_SECONDARY = "GLEASON_SECONDARY"
CLINICAL_STAGE = "CLINICAL_STAGE"
CORES_POSITIVE = "CORES_POSITIVE"
CORES_NEGATIVE = "CORES_NEGATIVE"
CORES_POSITIVE_PERCENTAGE = "CORES_POSITIVE_PERCENTAGE"
CORES_NEGATIVE_PERCENTAGE = "CORES_NEGATIVE_PERCENTAGE"

CONTINUOUS_FEATURE_COLUMNS = [AGE, PSA]
CATEGORICAL_FEATURE_COLUMNS = [GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
FEATURE_COLUMNS = CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

# TARGET COLUMNS
PN = "PN"
BCR = "BCR"
BCR_TIME = "BCR_TIME"
METASTASIS = "METASTASIS"
METASTASIS_TIME = "METASTASIS_TIME"
EE = "EE"
SVI = "SVI"
CRPC = "CRPC"
CRPC_TIME = "CRPC_TIME"
DEATH = "DEATH"
DEATH_TIME = "DEATH_TIME"

TARGET_COLUMNS = [PN, BCR, BCR_TIME, METASTASIS, METASTASIS_TIME, EE, SVI, CRPC, CRPC_TIME, DEATH, DEATH_TIME]

# TABLE TASKS
PN_TASK = BinaryClassificationTask(
    target_column=PN,
    decision_threshold_metric=BinaryBalancedAccuracy(),
    hps_tuning_metric=AUC(),
    evaluation_metrics=[Sensitivity(), Specificity()],
    criterion=BCEWithLogitsLoss()
)
BCR_TASK = SurvivalAnalysisTask(
    event_indicator_column=BCR,
    event_time_column=BCR_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored()
)
METASTASIS_TASK = SurvivalAnalysisTask(
    event_indicator_column=METASTASIS,
    event_time_column=METASTASIS_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored()
)
EE_TASK = BinaryClassificationTask(
    target_column=EE,
    decision_threshold_metric=BinaryBalancedAccuracy(),
    hps_tuning_metric=AUC(),
    evaluation_metrics=[Sensitivity(), Specificity()],
    criterion=BCEWithLogitsLoss()
)
SVI_TASK = BinaryClassificationTask(
    target_column=SVI,
    decision_threshold_metric=BinaryBalancedAccuracy(),
    hps_tuning_metric=AUC(),
    evaluation_metrics=[Sensitivity(), Specificity()],
    criterion=BCEWithLogitsLoss()
)
CRPC_TASK = SurvivalAnalysisTask(
    event_indicator_column=CRPC,
    event_time_column=CRPC_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored()
)
DEATH_TASK = SurvivalAnalysisTask(
    event_indicator_column=DEATH,
    event_time_column=DEATH_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored()
)

TABLE_TASKS = [BCR_TASK, PN_TASK, METASTASIS_TASK, EE_TASK, SVI_TASK, CRPC_TASK, DEATH_TASK]

# IMAGE TASKS
PROSTATE_SEGMENTATION_TASK = SegmentationTask(
    criterion=DiceLoss(),
    hps_tuning_metric=DiceMetric(),
    organ="Prostate",
    modality="CT"
)

IMAGE_TASKS = [PROSTATE_SEGMENTATION_TASK]
