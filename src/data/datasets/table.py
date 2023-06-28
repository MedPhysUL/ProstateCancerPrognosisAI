"""
    @file:              table.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 05/2023

    @Description:       This file contains a custom torch dataset named TableDataset. We follow
                        https://ieeexplore.ieee.org/document/8892612 setting for multi-output learning. This class
                        allows to cover the cases where some patients have a missing label for one task (or should I
                        say, for one output) while it is available for another.
"""

from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from torch import cat, from_numpy, stack, Tensor
from torch.utils.data import Dataset

from ..transforms.base import Identity, Transform
from ..transforms.enums import CategoricalTransform, ContinuousTransform
from ...tasks import BinaryClassificationTask, TaskList, SurvivalAnalysisTask
from ...tasks.base import TableTask


class TableDataModel(NamedTuple):
    """
    Data element named tuple. This tuple is used to separate features (x) and targets (y) where
        - x : D-dimensional dictionary containing (N, ) tensor or array where D is the number of features.
        - y : T-dimensional dictionary containing (N, ) tensor or array where T is the number of tasks. Note that it can
            also contain (N, 2) tensor or array in the case of survival analysis, where the first column corresponds to
            events indicators and the second to events times.
    """
    x: Dict[str, Union[np.ndarray, Tensor]]
    y: Dict[str, Union[np.ndarray, Tensor]]


class Feature(NamedTuple):
    """
    Feature column named tuple. This tuple is used to store the name of a column, its associated transform and whether
    it should be used for imputation.

    Elements
    --------
    column : str
        Name of the column.
    impute : bool
        Whether the column should be used for imputation.
    transform : Optional[Transform]
        Transform to apply to the column. If None, the identity transform is used.
    """
    column: str
    impute: bool = True
    transform: Optional[Transform] = Identity()


class TableDataset(Dataset):
    """
    A custom dataset class used to perform experiments on tabular data. It is designed to be used with a
    torch.utils.data.DataLoader. It allows to cover the cases where some patients have a missing label for one task
    (or should I say, for one output) while it is available for another. It also allows to perform experiments with
    multiple tasks.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            ids_column: str,
            tasks: Union[TableTask, TaskList, List[TableTask]],
            continuous_features: Optional[List[Feature]] = None,
            categorical_features: Optional[List[Feature]] = None,
            to_tensor: bool = False,
            random_state: int = 0
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with the original data.
        ids_column : str
            Name of the column containing the patient ids.
        tasks : Union[TableTask, TaskList, List[TableTask]]
            List of tasks.
        continuous_features : Optional[List[Feature]]
            List of column names associated with continuous feature data.
        categorical_features : Optional[List[Feature]]
            List of column names associated with categorical feature data.
        to_tensor : bool
            Whether we want the features and targets in tensors. False for numpy arrays.
        random_state : int
            Random state for the iterative imputer.
        """
        super(TableDataset).__init__()

        self._cat_features = categorical_features if categorical_features else []
        self._cat_features_cols = [f.column for f in self._cat_features]
        self._cont_features = continuous_features if continuous_features else []
        self._cont_features_cols = [f.column for f in self._cont_features]
        self._ids_col = ids_column
        self._imputed_df = None
        self._imputer_features_cols = [f.column for f in self.features if f.impute is True]
        self._imputer_cat_features_cols = list(set(self._cat_features_cols).intersection(self._imputer_features_cols))
        self._iterative_imputer = IterativeImputer(
            estimator=RandomForestRegressor(random_state=random_state),
            tol=1e-2,
            random_state=random_state
        )
        self._original_df = dataframe
        self._tasks = TaskList(tasks)
        self._target_cols = [task.target_column for task in self._tasks.table_tasks]
        self._to_tensor = to_tensor
        self._train_mask, self._valid_mask, self._test_mask = list(range(len(self))), [], []
        self._x_cat, self._x_cont = None, None

        self._initialize_dataset()
        self.update_masks(self._train_mask, self._valid_mask, self._test_mask)

    def __len__(self) -> int:
        """
        Returns the number of data elements in the dataset. This number is equal to the number of patients.

        Returns
        -------
        n : int
            Number of data elements in the dataset.
        """
        return self._original_df.shape[0]

    def __getitem__(
            self,
            idx: Union[int, List[int]]
    ) -> TableDataModel:
        """
        Gets dataset item. In the multi-output learning setting, the output variables, i.e the targets, share the same
        training features (See https://ieeexplore.ieee.org/document/8892612).

        Parameters
        ----------
        idx : Union[int, List[int]]
            Index of the data element to get. If idx is an integer, the corresponding data element is returned. If idx
            is a list of integers, a list of data elements is returned.

        Returns
        -------
        item : TableDataModel
            A data element. It is a named tuple containing the features (x) and targets (y) of the data element. The
            features are a dictionary containing the continuous and categorical features. The targets are a dictionary
            containing the targets of each task. The keys of the dictionaries are the names of the columns.
        """
        x = dict((col, self.x[idx, i]) for i, col in enumerate(self.features_columns))
        y = dict((col, y_task[idx]) for col, y_task in self.y.items())

        return TableDataModel(x=x, y=y)

    @property
    def categorical_features(self) -> List[Feature]:
        """
        Returns the list of categorical features.

        Returns
        -------
        cat_features : List[Feature]
            List of categorical features.
        """
        return self._cat_features

    @property
    def categorical_features_columns(self) -> List[str]:
        """
        Returns the list of column names associated with categorical feature data.

        Returns
        -------
        cat_features_cols : List[str]
            List of column names associated with categorical feature data.
        """
        return self._cat_features_cols

    @property
    def columns(self) -> List[str]:
        """
        Returns the list of column names associated with the data. The list of column names is equal to the list of
        feature column names plus the list of target column names.

        Returns
        -------
        columns : List[str]
            List of column names associated with the data.
        """
        return self.features_columns + self.target_columns

    @property
    def continuous_features(self) -> List[Feature]:
        """
        Returns the list of continuous features.

        Returns
        -------
        cont_features : List[Feature]
            List of continuous features.
        """
        return self._cont_features

    @property
    def continuous_features_columns(self) -> List[str]:
        """
        Returns the list of column names associated with continuous feature data.

        Returns
        -------
        cont_features_cols : List[str]
            List of column names associated with continuous feature data.
        """
        return self._cont_features_cols

    @property
    def features(self) -> List[Feature]:
        """
        Returns the list of features. The list of features is equal to the list of continuous features plus the list of
        categorical features.

        Returns
        -------
        features : List[Feature]
            List of features.
        """
        return self._cont_features + self._cat_features

    @property
    def features_columns(self) -> List[str]:
        """
        Returns the list of column names associated with feature data. The list of column names is equal to the list of
        continuous feature column names plus the list of categorical feature column names.

        Returns
        -------
        features_cols : List[str]
            List of column names associated with feature data.
        """
        return self._cont_features_cols + self._cat_features_cols

    @property
    def ids(self) -> List[str]:
        """
        Returns the list of ids associated with the data.

        Returns
        -------
        ids : List[str]
            List of ids associated with the data.
        """
        return list(self._original_df[self._ids_col].values)

    @property
    def ids_column(self) -> str:
        """
        Returns the name of the column containing the ids.

        Returns
        -------
        ids_col : str
            Name of the column containing the ids.
        """
        return self._ids_col

    @property
    def ids_to_row_idx(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping ids to row indices.

        Returns
        -------
        ids_to_row_idx : Dict[str, int]
            Dictionary mapping ids to row indices.
        """
        return {id_: i for i, id_ in enumerate(self.ids)}

    @property
    def imputed_dataframe(self) -> pd.DataFrame:
        """
        Returns the imputed data.

        Returns
        -------
        imputed_df : pd.DataFrame
            Imputed data.
        """
        return self._imputed_df

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Returns the original data.

        Returns
        -------
        original_df : pd.DataFrame
            Original data.
        """
        return self._original_df

    @property
    def target_columns(self) -> List[str]:
        """
        Returns the list of column names associated with target data.

        Returns
        -------
        target_cols : List[str]
            List of column names associated with target data.
        """
        return self._target_cols

    @property
    def tasks(self) -> TaskList:
        """
        Returns the list of tasks.

        Returns
        -------
        tasks : TaskList
            List of tasks.
        """
        return self._tasks

    @property
    def test_mask(self) -> Optional[List[int]]:
        """
        Returns the list of indices associated with test data.

        Returns
        -------
        test_mask : Optional[List[int]]
            List of indices associated with test data.
        """
        return self._test_mask

    @property
    def to_tensor(self) -> bool:
        """
        Returns whether the data should be converted to tensors.

        Returns
        -------
        to_tensor : bool
            Whether the data should be converted to tensors.
        """
        return self._to_tensor

    @property
    def train_mask(self) -> List[int]:
        """
        Returns the list of indices associated with training data.

        Returns
        -------
        train_mask : List[int]
            List of indices associated with training data.
        """
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        """
        Returns the list of indices associated with validation data.

        Returns
        -------
        valid_mask : Optional[List[int]]
            List of indices associated with validation data.
        """
        return self._valid_mask

    @property
    def x(self) -> Union[Tensor, np.array]:
        """
        Returns the feature data.

        Returns
        -------
        x : Union[Tensor, np.array]
            Feature data.
        """
        return self._get_features()

    @property
    def x_cat(self) -> Optional[Union[np.array, Tensor]]:
        """
        Returns the categorical feature data.

        Returns
        -------
        x_cat : Optional[Union[np.array, Tensor]]
            Categorical feature data.
        """
        return self._x_cat

    @property
    def x_cont(self) -> Optional[Union[np.array, Tensor]]:
        """
        Returns the continuous feature data.

        Returns
        -------
        x_cont : Optional[Union[np.array, Tensor]]
            Continuous feature data.
        """
        return self._x_cont

    @property
    def y(self) -> Dict[str, Union[np.array, Tensor]]:
        """
        Returns the target data.

        Returns
        -------
        y : Dict[str, Union[np.array, Tensor]]
            Target data.
        """
        return self._y

    def _initialize_dataset(self):
        """
        Initializes the dataset. Validates the features and tasks provided by the user. Initializes the features and
        targets. Updates the masks.
        """
        self._validate_features()
        self._validate_tasks()
        self._initialize_features()
        self._initialize_targets()

    def _validate_features(self):
        """
        Validates the features provided by the user. Raises an error if the features are not valid. If no features are
        provided, raises an error. If both continuous and categorical features are provided, raises an error. If the
        features are valid, does nothing.
        """
        for features, cont in zip([self._cont_features, self._cat_features], [True, False]):
            self._check_features_validity(features, cont)

    def _check_features_validity(
            self,
            features: Optional[List[Feature]] = None,
            continuous: bool = True
    ) -> None:
        """
        Checks that the given features are valid. If not, raises a ValueError.

        Parameters
        ----------
        features : Optional[List[Feature]]
            List of features to check.
        continuous : bool
            True if the features are continuous, false if they are categorical.
        """
        if features is not None:
            dataframe_columns = list(self._original_df.columns.values)
            for f in features:
                assert f.column in dataframe_columns, f"Column {f.column} is not part of the given dataframe."

                if f.transform:
                    if continuous:
                        assert isinstance(f.transform, tuple(t.value for t in ContinuousTransform)), (
                            f"Transform {f.transform} is not a continuous transform. Available transforms are: "
                            f"{[t.name for t in ContinuousTransform]}."
                        )
                    else:
                        assert isinstance(f.transform, tuple(t.value for t in CategoricalTransform)), (
                            f"Transform {f.transform} is not a categorical transform. Available transforms are: "
                            f"{[t.name for t in CategoricalTransform]}."
                        )

    def _validate_tasks(self):
        """
        Validates the tasks provided by the user. Raises an error if the tasks are not valid. If no tasks are provided,
        raises an error. If the tasks are valid, does nothing.
        """
        assert all(isinstance(task, TableTask) for task in TaskList(self._tasks)), (
            f"All tasks must be instances of 'TableTask'."
        )

        dataframe_columns = list(self._original_df.columns.values)
        for task in self._tasks:
            assert task.target_column in dataframe_columns, (
                f"Column {task.target_column} is not part of the given dataframe."
            )

    def _initialize_features(self):
        """
        Initializes the features in a proper format.
        """
        if self._cat_features_cols:
            self._original_df[self._cat_features_cols] = self._original_df[self._cat_features_cols].astype('category')
        if self._cont_features_cols:
            self._original_df[self._cont_features_cols] = self._original_df[self._cont_features_cols].astype('float')

    def _initialize_targets(self):
        """
        Initializes the targets in a proper format.
        """
        targets = {}
        for task in self._tasks:
            t = self._original_df[task.target_column].to_numpy(dtype=float)

            if (not isinstance(task, BinaryClassificationTask)) and self._to_tensor:
                t = from_numpy(t).float()
            elif isinstance(task, (BinaryClassificationTask, SurvivalAnalysisTask)):
                t = np.nan_to_num(t, nan=-1e9)
                if self._to_tensor:
                    t = from_numpy(t).long()
                else:
                    t = t.astype(int)

            if isinstance(task, SurvivalAnalysisTask):
                event_time = self._original_df[task.event_time_column].to_numpy(dtype=float)
                if self._to_tensor:
                    event_time = from_numpy(event_time).float()
                    t = stack([t, event_time], dim=1)
                else:
                    t = np.stack([t, event_time], axis=1)

            targets[task.name] = t

        self._y = targets

    def _get_features(self) -> Union[Tensor, np.array]:
        """
        Returns the feature data. If both continuous and categorical features are provided, concatenates them.

        Returns
        -------
        x : Union[Tensor, np.array]
            Feature data.
        """
        if self._cont_features_cols is None:
            return self.x_cat
        elif self._cat_features_cols is None:
            return self.x_cont
        else:
            if not self._to_tensor:
                return np.concatenate((self.x_cont, self.x_cat), axis=1)
            else:
                return cat((self.x_cont, self.x_cat), dim=1)

    def update_dataframe(
            self,
            dataframe: pd.DataFrame,
            update_masks: bool = True
    ) -> None:
        """
        Updates the dataframe and then preprocesses the data available according to the current statistics of the
        training data. If update_masks is True, also updates the masks.

        Parameters
        ----------
        dataframe : pd.DataFrame
            New dataframe. Must have the same columns as the original dataframe.
        update_masks : bool
            If True, also updates the masks.
        """
        self._original_df = dataframe
        self._initialize_dataset()
        if update_masks:
            self.update_masks(self._train_mask, self._valid_mask, self._test_mask)

    def update_masks(
            self,
            train_mask: List[int],
            valid_mask: Optional[List[int]] = None,
            test_mask: Optional[List[int]] = None
    ) -> None:
        """
        Updates the train, valid and test masks and then preprocesses the data available according to the current
        statistics of the training data.

        Parameters
        ----------
        train_mask : List[int]
            List of idx in the training set.
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        test_mask : Optional[List[int]]
            List of idx in the test set.
        """
        self._imputed_df = self._original_df.copy()

        self._train_mask = train_mask
        self._valid_mask = valid_mask if valid_mask is not None else []
        self._test_mask = test_mask if test_mask is not None else []

        if self._cat_features or self._cont_features:
            self._preprocess_cat_features()

            if self._imputer_features_cols:
                self._impute_missing_features()

            self._preprocess_cont_features()

        self._set_features()
        self._set_scaling_factors()

    def _get_current_train_stats(self) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Returns the current statistics and encodings related to the training data.

        Returns
        -------
        stats : Tuple[Optional[pd.Series], Optional[pd.Series]]
            Tuple containing the current statistics and encodings related to the training data :
                mean : Optional[pd.Series]
                    Means of the numerical columns according to the training mask.
                std : Optional[pd.Series]
                    Standard deviations of the numerical columns according to the training mask.
        """
        train_data = self._original_df.iloc[self._train_mask]

        if self._cont_features_cols is None:
            return None, None
        else:
            return train_data[self._cont_features_cols].mean(), train_data[self._cont_features_cols].std()

    def _preprocess_cat_features(self):
        """
        Preprocesses the categorical features according to the transforms provided by the user. If no transforms are
        provided, does nothing.
        """
        if self._cat_features_cols:
            for feature in self._cat_features:
                self._imputed_df[feature.column] = feature.transform(df=self._imputed_df[feature.column])

            self._imputed_df[self._cat_features_cols] = self._imputed_df[self._cat_features_cols].astype('float')

    def _impute_missing_features(self):
        """
        Imputes the missing values of the features using an iterative imputer. For categorical features, we need an
        additional step to map the imputed values to the closest category, as an iterative imputer only works with
        float values.
        """
        df = self._imputed_df[self._imputer_features_cols]

        self._iterative_imputer.fit(df.iloc[self._train_mask])
        data = self._iterative_imputer.transform(df)
        temp_df = pd.DataFrame(data, columns=self._imputer_features_cols)

        for column in self._imputer_cat_features_cols:
            original_array = np.array(df[column].dropna())
            categories = np.unique(original_array)
            bins = self._convert_categories_to_bins(categories)

            imputed_array = np.array(temp_df[column])
            indices = np.digitize(imputed_array, bins)
            result = categories[indices - 1]
            temp_df[column] = result

        self._imputed_df[self._imputer_features_cols] = temp_df

    @staticmethod
    def _convert_categories_to_bins(categories: np.ndarray) -> np.ndarray:
        """
        Converts a list of categories to bins. The bins are defined as the midpoints between each category.

        Parameters
        ----------
        categories : np.ndarray
            List of categories.

        Returns
        -------
        bins : np.ndarray
            List of bins.
        """
        min_value, max_value = np.min(categories), np.max(categories)
        mid_values = np.linspace(min_value, max_value, num=len(categories) + 1)[1:-1]
        bins = np.concatenate(([-np.inf], mid_values, [np.inf]))
        return bins

    def _preprocess_cont_features(self):
        """
        Preprocesses the continuous features according to the transforms provided by the user. If no transforms are
        provided, does nothing.
        """
        mean, std = self._get_current_train_stats()
        if self._cont_features_cols:
            for feature in self._cont_features:
                col = feature.column
                self._imputed_df[col] = feature.transform(df=self._imputed_df[col], mean=mean[col], std=std[col])

    def _set_features(self):
        """
        Sets the features of the dataset. If the dataset is to be converted to tensors, converts the features to
        tensors.
        """
        self._x_cat = self._imputed_df[self._cat_features_cols].to_numpy(dtype=float)
        self._x_cont = self._imputed_df[self._cont_features_cols].to_numpy(dtype=float)

        if self._to_tensor:
            self._x_cat = from_numpy(self._x_cat)
            self._x_cont = from_numpy(self._x_cont)

    def _set_scaling_factors(self):
        """
        Sets scaling factor of all binary classification tasks.
        """
        for task in self.tasks.binary_classification_tasks:
            for metric in task.metrics:
                metric.update_scaling_factor(y_train=self.y[task.name][self.train_mask])

            if task.criterion:
                task.criterion.update_scaling_factor(y_train=self.y[task.name][self.train_mask])
