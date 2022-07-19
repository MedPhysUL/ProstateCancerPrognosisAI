"""
    @file:              constants.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file is used to store helpful constants.
"""

# HOLDOUT SET
SEED = 1010710
HOLDOUT_SIZE = 0.15

# PATHS
CLINICAL_DATA_PATH = "local_data/clinical_data.xlsx"

LEARNING_DATAFRAME_PATH = "local_data/learning_dataframe.xlsx"
HOLDOUT_DATAFRAME_PATH = "local_data/holdout_dataframe.xlsx"

PATIENTS_FOLDER_PATH = "local_data/patients"
RECORDS_PATH = "local_data/records.json"

# COLUMNS
ID = "ID"
AGE = "AGE"
PSA = "PSA"
GLEASON_GLOBAL = "GLEASON_GLOBAL"
GLEASON_PRIMARY = "GLEASON_PRIMARY"
GLEASON_SECONDARY = "GLEASON_SECONDARY"
CLINICAL_STAGE = "CLINICAL_STAGE"
PN = "PN"
BCR = "BCR"

# DATA TYPE
DATE_TYPE = "date"
NUMERIC_TYPE = "numeric"
CATEGORICAL_TYPE = "text"

# TYPE DICT
COLUMNS_TYPES = {
    ID: CATEGORICAL_TYPE,
    AGE: NUMERIC_TYPE,
    PSA: NUMERIC_TYPE,
    GLEASON_GLOBAL: CATEGORICAL_TYPE,
    GLEASON_PRIMARY: CATEGORICAL_TYPE,
    GLEASON_SECONDARY: CATEGORICAL_TYPE,
    CLINICAL_STAGE: CATEGORICAL_TYPE
}
