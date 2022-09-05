"""
    @file:              constants.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file stores helpful constants.
"""

import os

from src.utils.score_metrics import AUC, DICE
from src.utils.tasks import ClassificationTask, SegmentationTask

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

RECORDS_PATH = os.path.join("local_data", "records")
OUTLIERS_RECORDS_PATH = os.path.join(RECORDS_PATH, "outliers")
DESCRIPTIVE_ANALYSIS_PATH = os.path.join(RECORDS_PATH, "descriptive_analyses")

# COLUMNS
ID = "ID"
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
PN = "PN"
BCR = "BCR"

# DATA TYPE
DATE_TYPE = "date"
NUMERIC_TYPE = "numeric"
CATEGORICAL_TYPE = "text"

# TYPE DICT
COLUMNS_TYPES = {
    AGE: NUMERIC_TYPE,
    PSA: NUMERIC_TYPE,
    GLEASON_GLOBAL: CATEGORICAL_TYPE,
    GLEASON_PRIMARY: CATEGORICAL_TYPE,
    GLEASON_SECONDARY: CATEGORICAL_TYPE,
    CLINICAL_STAGE: CATEGORICAL_TYPE,
    CORES_POSITIVE: NUMERIC_TYPE,
    CORES_NEGATIVE: NUMERIC_TYPE,
    CORES_POSITIVE_PERCENTAGE: NUMERIC_TYPE,
    CORES_NEGATIVE_PERCENTAGE: NUMERIC_TYPE,
    PN: CATEGORICAL_TYPE,
    BCR: CATEGORICAL_TYPE
}

# SET OF MODALITIES
MODALITIES = {"CT"}

# TASKS
TABLE_TASKS = [ClassificationTask(target_col=PN, metric=AUC()), ClassificationTask(target_col=BCR, metric=AUC())]
IMAGE_TASKS = [SegmentationTask(metric=DICE(), organ="Prostate", modality="CT")]
