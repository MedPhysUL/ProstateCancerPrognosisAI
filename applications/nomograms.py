"""
    @file:              nomograms.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the clinical data used for nomograms.
"""

import env_apps

from typing import Dict, List, Optional, Union

import pandas as pd

from constants import *
from src.data.datasets import Feature, Mask, TableDataset
from src.data.processing.sampling import extract_masks
from src.data.transforms import MappingEncoding


class NomogramDataframe:
    """
    This class allows to generate the dataframes. It is used to generate the dataframes used for nomograms.
    """

    FINAL_SET_KEY = "final_set"
    OUTER_SPLIT_KEY = "outer_split"

    def __init__(
            self,
            table_dataset: TableDataset,
    ):
        """
        Constructor. It initializes the class.

        Parameters
        ----------
        table_dataset : TableDataset
            The table dataset.
        """
        self.table_dataset = table_dataset

    def _get_imputed_dataframe(
            self,
            train_mask: List[int],
            test_mask: List[int],
            clinical_stage_column: Optional[str] = None,
            mapping: Optional[Dict[Union[float, int], str]] = None,
            valid_mask: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        This method returns the imputed dataframe.

        Parameters
        ----------
        train_mask : List[int]
            The train mask.
        test_mask : List[int]
            The test mask.
        clinical_stage_column : Optional[str]
            The clinical stage column.
        mapping : Optional[Dict[Union[float, int], str]]
            The mapping.
        valid_mask : Optional[List[int]]
            The valid mask.

        Returns
        -------
        dataframe : pd.DataFrame
            The imputed dataframe.
        """
        self.table_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

        dataframe = self.table_dataset.imputed_dataframe.copy()

        if mapping and clinical_stage_column:
            dataframe[clinical_stage_column] = dataframe[clinical_stage_column].map(mapping)

        if valid_mask:
            named_masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
        else:
            named_masks = {"train": train_mask, "test": test_mask}

        return pd.concat(
            objs=[dataframe.iloc[mask].assign(SETS=name) for name, mask in named_masks.items()],
            ignore_index=True
        )

    def save_outer_splits_dataframes(
            self,
            path_to_folder: str,
            masks: dict,
            clinical_stage_column: Optional[str] = None,
            mapping: Optional[Dict[Union[float, int], str]] = None
    ) -> None:
        """
        This method saves the outer splits dataframes.

        Parameters
        ----------
        path_to_folder : str
            The path to the folder.
        masks : dict
            The masks.
        clinical_stage_column : Optional[str]
            The clinical stage column.
        mapping : Optional[Dict[Union[float, int], str]]
            The mapping.
        """
        for k, v in masks.items():
            train_mask, valid_mask, test_mask, inner_masks = v[Mask.TRAIN], v[Mask.VALID], v[Mask.TEST], v[Mask.INNER]

            dataframe = self._get_imputed_dataframe(
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask,
                clinical_stage_column=clinical_stage_column,
                mapping=mapping
            )

            dataframe.to_csv(os.path.join(path_to_folder, f"{self.OUTER_SPLIT_KEY}_{k}.csv"), index=False)

    def save_final_dataframe(
            self,
            path_to_folder: str,
            train_mask: List[int],
            test_mask: List[int],
            clinical_stage_column: Optional[str] = None,
            mapping: Optional[Dict[Union[float, int], str]] = None
    ):
        """
        This method saves the final dataframe.

        Parameters
        ----------
        path_to_folder : str
            The path to the folder.
        train_mask : List[int]
            The train mask.
        test_mask : List[int]
            The test mask.
        clinical_stage_column : Optional[str]
            The clinical stage column.
        mapping : Optional[Dict[Union[float, int], str]]
            The mapping.
        """
        self.table_dataset.update_masks(train_mask=train_mask, test_mask=test_mask)

        dataframe = self._get_imputed_dataframe(
            train_mask=train_mask,
            test_mask=test_mask,
            clinical_stage_column=clinical_stage_column,
            mapping=mapping
        )

        dataframe.to_csv(os.path.join(path_to_folder, f"{self.FINAL_SET_KEY}.csv"), index=False)


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    CLINICAL_STAGE_MSKCC = Feature(
        column="CLINICAL_STAGE_MSKCC_STYLE",
        transform=MappingEncoding(
            {"T1c": 0, "T2": 0.2, "T2a": 0.2, "T2b": 0.4, "T2c": 0.6, "T3": 0.8, "T3a": 0.8, "T3b": 1}
        )
    )
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    # MSKCC
    mskcc_learning_df, mskcc_holdout_df = pd.read_csv(MSKCC_LEARNING_TABLE_PATH), pd.read_csv(MSKCC_HOLDOUT_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=pd.concat([mskcc_learning_df, mskcc_holdout_df], ignore_index=True),
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE_MSKCC, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    nomogram_dataframe = NomogramDataframe(table_dataset=table_dataset)
    nomogram_dataframe.save_outer_splits_dataframes(
        path_to_folder="local_data/nomograms/MSKCC/",
        masks=masks,
        clinical_stage_column=CLINICAL_STAGE_MSKCC.column,
        mapping={0: "T1c", 0.2: "T2a", 0.4: "T2b", 0.6: "T2c", 0.8: "T3a", 1: "T3b"}
    )
    nomogram_dataframe.save_final_dataframe(
        path_to_folder="local_data/nomograms/MSKCC/",
        train_mask=list(range(len(mskcc_learning_df))),
        test_mask=list(range(len(mskcc_learning_df), len(mskcc_learning_df) + len(mskcc_holdout_df))),
        clinical_stage_column=CLINICAL_STAGE_MSKCC.column,
        mapping={0: "T1c", 0.2: "T2a", 0.4: "T2b", 0.6: "T2c", 0.8: "T3a", 1: "T3b"}
    )

    # CAPRA
    learning_df, holdout_df = pd.read_csv(LEARNING_TABLE_PATH), pd.read_csv(HOLDOUT_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=pd.concat([learning_df, holdout_df], ignore_index=True),
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    nomogram_dataframe = NomogramDataframe(table_dataset=table_dataset)
    nomogram_dataframe.save_outer_splits_dataframes(
        path_to_folder="local_data/nomograms/CAPRA/",
        masks=masks,
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )
    nomogram_dataframe.save_final_dataframe(
        path_to_folder="local_data/nomograms/CAPRA/",
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df))),
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )

    # CUSTOM
    nomogram_dataframe.save_outer_splits_dataframes(
        path_to_folder="local_data/nomograms/CUSTOM/clinical",
        masks=masks
    )
    nomogram_dataframe.save_final_dataframe(
        path_to_folder="local_data/nomograms/CUSTOM/clinical",
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df)))
    )
