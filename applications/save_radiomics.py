"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from constants import (
    HOLDOUT_TABLE_PATH,
    ID,
    LEARNING_TABLE_PATH,
    MANUAL_EXTRACTED_RADIOMICS_PATH,
    MANUAL_FILTERED_RADIOMICS_PATH,
    MASKS_PATH,
    TABLE_TASKS
)
from src.data.datasets import Feature, Mask, TableDataset
from src.data.processing.sampling import extract_masks
from src.data.transforms import MappingEncoding


class RadiomicsDataframe:
    """
    This class allows to generate the dataframes. It is used to generate the dataframes used for radiomics analysis.
    """

    FINAL_SET_KEY = "final_set"
    OUTER_SPLIT_KEY = "outer_split"
    INNER_SPLIT_KEY = "inner_split"

    RADIOMIC_NAME = "RADIOMIC"

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

    @staticmethod
    def get_radiomics_dataframe(
            path_to_dataframe: str,
            modality: str
    ) -> pd.DataFrame:
        """
        This method returns the radiomics dataframe.

        Parameters
        ----------
        path_to_dataframe : str
            The path to the radiomics dataframe.
        modality : str
            The modality.

        Returns
        -------
        radiomics : pd.DataFrame
            The radiomics dataframe.
        """
        radiomics_df = pd.read_csv(path_to_dataframe)
        radiomics_df = radiomics_df.sort_values([ID])

        columns = []
        for column in radiomics_df.columns:
            if modality == "PT" and column.startswith("original_shape"):
                continue
            elif column.startswith("diagnostics") or column == ID:
                continue
            else:
                columns.append(column)

        radiomics_df = radiomics_df[columns]
        radiomics_df = radiomics_df.rename(columns=lambda x: f"{modality}_{x}")

        return radiomics_df

    def _get_fitted_random_forest(
            self,
            radiomics_df: pd.DataFrame,
            task: str,
            train_mask: List[int],
            valid_mask: Optional[List[int]] = None,
    ) -> RandomForestClassifier:
        """
        This method returns the fitted random forest. It is used to compute the feature importance.

        Parameters
        ----------
        radiomics_df : pd.DataFrame
            The radiomics dataframe.
        task : str
            The task.
        train_mask : List[int]
            The train mask.
        valid_mask : Optional[List[int]]
            The valid mask.

        Returns
        -------
        forest : RandomForestClassifier
            The fitted random forest.
        """
        mask = train_mask + valid_mask if valid_mask is not None else train_mask
        clinical_df = self.table_dataset.imputed_dataframe.copy()
        targets = clinical_df.loc[:, [task]]
        targets = np.array(targets).ravel()
        targets = targets[mask]
        nan_mask = np.isnan(targets)

        y_train = targets[~nan_mask]
        x_train = radiomics_df.iloc[mask][~nan_mask]

        forest = RandomForestClassifier(n_estimators=10000, random_state=0)
        forest.fit(x_train, y_train)

        return forest

    def _get_filtered_radiomics_dataframe(
            self,
            radiomics_df: pd.DataFrame,
            forest: RandomForestClassifier
    ) -> pd.DataFrame:
        """
        This method filters the radiomics dataframe. It keeps only the most important features.

        Parameters
        ----------
        radiomics_df : pd.DataFrame
            The radiomics dataframe.
        forest : RandomForestClassifier
            The fitted random forest.

        Returns
        -------
        radiomics_df : pd.DataFrame
            The filtered radiomics dataframe.
        """
        importances = forest.feature_importances_
        forest_importances = pd.Series(importances, index=radiomics_df.columns)
        most_important_features = list(forest_importances.nlargest(n=6).index)
        radiomics = radiomics_df[most_important_features].copy()
        new_columns = {old_col: f"{self.RADIOMIC_NAME}_{i + 1}" for i, old_col in enumerate(radiomics.columns)}

        return radiomics.rename(columns=new_columns)

    def _get_imputed_dataframe(
            self,
            radiomics_df: pd.DataFrame,
            target_col: str,
            train_mask: List[int],
            test_mask: List[int],
            clinical_stage_column: Optional[str] = None,
            mapping: Optional[Dict[Union[float, int], str]] = None,
            valid_mask: Optional[List[int]] = None,
            show: bool = False
    ) -> pd.DataFrame:
        """
        This method returns the imputed dataframe. It is used to compute the feature importance.

        Parameters
        ----------
        radiomics_df : pd.DataFrame
            The radiomics dataframe.
        target_col : str
            The target column.
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
        show : bool
            If True, it shows the feature importance.

        Returns
        -------
        dataframe : pd.DataFrame
            The imputed dataframe.
        """
        self.table_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

        forest = self._get_fitted_random_forest(radiomics_df, target_col, train_mask, valid_mask)

        if show:
            self._plot_features_importance(forest, radiomics_df.columns)

        radiomics = self._get_filtered_radiomics_dataframe(radiomics_df, forest)

        clinical_df = self.table_dataset.imputed_dataframe.copy()
        if mapping and clinical_stage_column:
            clinical_df[clinical_stage_column] = clinical_df[clinical_stage_column].map(mapping)

        dataframe = pd.concat([clinical_df, radiomics], axis=1)

        return dataframe

    def save_outer_and_inner_splits_dataframes(
            self,
            path_to_folder: str,
            radiomics_df: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
            masks: dict,
            clinical_stage_column: str,
            mapping: Dict[Union[float, int], str],
            show: bool = False
    ):
        """
        This method saves the outer and inner splits dataframes.

        Parameters
        ----------
        path_to_folder : str
            The path to the folder where to save the dataframes.
        radiomics_df : Union[pd.DataFrame, Dict[int, pd.DataFrame]]
            The radiomics dataframe.
        masks : dict
            The masks.
        clinical_stage_column : str
            The clinical stage column.
        mapping : Dict[Union[float, int], str]
            The mapping.
        show : bool
            If True, it shows the feature importance.
        """
        os.makedirs(path_to_folder, exist_ok=True)
        for task in TABLE_TASKS:
            os.makedirs(os.path.join(path_to_folder, task.target_column), exist_ok=True)
            for k, v in masks.items():
                outer_split_path = os.path.join(path_to_folder, task.target_column, f"{self.OUTER_SPLIT_KEY}_{k}")
                os.makedirs(outer_split_path, exist_ok=True)

                df = radiomics_df[k] if isinstance(radiomics_df, dict) else radiomics_df

                dataframe = self._get_imputed_dataframe(
                    radiomics_df=df,
                    target_col=task.target_column,
                    train_mask=v[Mask.TRAIN],
                    clinical_stage_column=clinical_stage_column,
                    mapping=mapping,
                    valid_mask=v[Mask.VALID],
                    test_mask=v[Mask.TEST],
                    show=show
                )

                dataframe.to_csv(os.path.join(outer_split_path, f"{self.OUTER_SPLIT_KEY}.csv"), index=False)

                path_to_inner_splits = os.path.join(outer_split_path, f"{self.INNER_SPLIT_KEY}s")
                os.makedirs(path_to_inner_splits, exist_ok=True)
                for idx, inner_mask in v[Mask.INNER].items():
                    dataframe = self._get_imputed_dataframe(
                        radiomics_df=df,
                        target_col=task.target_column,
                        train_mask=inner_mask[Mask.TRAIN],
                        clinical_stage_column=clinical_stage_column,
                        mapping=mapping,
                        valid_mask=inner_mask[Mask.VALID],
                        test_mask=inner_mask[Mask.TEST],
                        show=show
                    )

                    dataframe.to_csv(
                        os.path.join(path_to_inner_splits, f"{self.INNER_SPLIT_KEY}_{idx}.csv"), index=False
                    )

    def save_final_dataframe(
            self,
            path_to_folder: str,
            radiomics_df: pd.DataFrame,
            train_mask: List[int],
            test_mask: List[int],
            clinical_stage_column: str,
            mapping: Dict[Union[float, int], str],
            show: bool = False
    ):
        """
        This method saves the final dataframe.

        Parameters
        ----------
        path_to_folder : str
            The path to the folder where to save the dataframes.
        radiomics_df : pd.DataFrame
            The radiomics dataframe.
        train_mask : List[int]
            The train mask.
        test_mask : List[int]
            The test mask.
        clinical_stage_column : str
            The clinical stage column.
        mapping : Dict[Union[float, int], str]
            The mapping.
        show : bool
            If True, it shows the feature importance.

        Returns
        -------
        dataframe : pd.DataFrame
            The imputed dataframe.
        """
        os.makedirs(path_to_folder, exist_ok=True)
        for task in TABLE_TASKS:
            path_to_task = os.path.join(path_to_folder, task.target_column)
            os.makedirs(path_to_task, exist_ok=True)

            dataframe = self._get_imputed_dataframe(
                radiomics_df=radiomics_df,
                target_col=task.target_column,
                train_mask=train_mask,
                test_mask=test_mask,
                clinical_stage_column=clinical_stage_column,
                mapping=mapping,
                show=show
            )

            dataframe.to_csv(os.path.join(path_to_task, f"{self.FINAL_SET_KEY}.csv"), index=False)

    @staticmethod
    def _plot_features_importance(
            forest: RandomForestClassifier,
            columns: List[str],
            n_features: int = 6,
            figsize=(12, 6)
    ):
        """
        This method plots the features importance.

        Parameters
        ----------
        forest : RandomForestClassifier
            The random forest classifier.
        columns : List[str]
            The columns.
        n_features : int
            The number of features to plot.
        figsize : tuple
            The figure size.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        importances = forest.feature_importances_

        tree_importance_sorted_idx = np.argsort(importances)
        tree_indices = np.arange(0, len(importances)) + 0.5

        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_], axis=0
        )[tree_importance_sorted_idx][-n_features:]

        ax.barh(tree_indices[:n_features], importances[tree_importance_sorted_idx][-n_features:], height=0.7, xerr=std)
        ax.set_yticks(tree_indices[:n_features])
        ax.set_yticklabels(columns[tree_importance_sorted_idx][-n_features:])
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        ax.set_ylim((0, len(importances[-n_features:])))
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    # Outer and inner split
    table_dataset = TableDataset(
        dataframe=pd.read_csv(LEARNING_TABLE_PATH),
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    radiomics_dataframe = RadiomicsDataframe(table_dataset=table_dataset)

    ct_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_ct_radiomics.csv"), "CT"
    )
    pt_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_pt_radiomics.csv"), "PT"
    )
    radiomics_df = pd.concat([ct_radiomics_df, pt_radiomics_df], axis=1)

    radiomics_dataframe.save_outer_and_inner_splits_dataframes(
        path_to_folder=MANUAL_FILTERED_RADIOMICS_PATH,
        radiomics_df=radiomics_df,
        masks=masks,
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )

    # Final set
    learning_df, holdout_df = pd.read_csv(LEARNING_TABLE_PATH), pd.read_csv(HOLDOUT_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=pd.concat([learning_df, holdout_df], ignore_index=True),
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    radiomics_dataframe = RadiomicsDataframe(table_dataset=table_dataset)

    learning_ct_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_ct_radiomics.csv"), "CT"
    )
    learning_pt_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_pt_radiomics.csv"), "PT"
    )
    learning_radiomics_df = pd.concat([learning_ct_radiomics_df, learning_pt_radiomics_df], axis=1)

    holdout_ct_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "holdout_ct_radiomics.csv"), "CT"
    )
    holdout_pt_radiomics_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "holdout_pt_radiomics.csv"), "PT"
    )
    holdout_radiomics_df = pd.concat([holdout_ct_radiomics_df, holdout_pt_radiomics_df], axis=1)

    radiomics_df = pd.concat([learning_radiomics_df, holdout_radiomics_df], ignore_index=True)

    radiomics_dataframe.save_final_dataframe(
        path_to_folder=MANUAL_FILTERED_RADIOMICS_PATH,
        radiomics_df=radiomics_df,
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df))),
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )
