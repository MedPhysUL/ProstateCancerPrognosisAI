"""
    @file:              sampling.py
    @Author:            Maxence Larose, Nicolas Raymond, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains the 'Sampler' class used to separate test sets and valid sets from train
                        sets. Also contains few functions used to extract specific datasets.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from ..datasets import Mask, TableDataset


class Sampler:
    """
    A random stratified sampler used to generate lists of indexes to use as train, valid and test masks for outer and
    inner validation loops.
    """

    def __init__(
            self,
            dataset: TableDataset,
            n_out_split: int,
            n_in_split: int,
            valid_size: float = 0.20,
            test_size: float = 0.20,
            random_state: Optional[int] = None,
            alpha: int = 4,
            patience: int = 100
    ):
        """
        Sets private and public attributes of the sampler.

        Parameters
        ----------
        dataset : TableDataset
            Custom dataset.
        n_out_split : int
            Number of outer splits to produce.
        n_in_split : int
            Number of inner splits to produce.
        valid_size : float
            Percentage of data taken to create the validation indexes set.
        test_size : float
            Percentage of data taken to create the test indexes set
        random_state : Optional[int]
            Random state integer.
        alpha : int
            IQR multiplier used to check numerical variable range validity of the masks created
        patience : int
            Number of tries that the sampler has to make a single valid split
        """
        if n_out_split <= 0:
            raise ValueError('Number of outer split must be greater than 0')
        if n_in_split < 0:
            raise ValueError('Number of inner split must be greater or equal to 0')
        if not (0 <= valid_size < 1):
            raise ValueError('Validation size must be in the range [0, 1)')
        if not (0 < test_size < 1):
            raise ValueError('Test size must be in the range (0, 1)')
        if valid_size + test_size >= 1:
            raise ValueError('Train size must be non null')

        # Private attributes
        self.__dataset = dataset
        self.__unique_encodings = {}

        targets = []
        for value in dataset.y.values():
            if dataset.to_tensor:
                value = value.numpy()
            if value.ndim == 1:
                targets.append(value)
            else:
                targets.append(value[:, 0])

        self.__targets = np.array(targets)

        # Public attributes
        self.alpha = alpha
        self.n_out_split = n_out_split
        self.n_in_split = n_in_split
        self.patience = patience
        self.random_state = random_state

        # Public method
        self.split = self.__define_split_function(test_size, valid_size)

    def __call__(
            self
    ) -> Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]]:
        """
        Returns lists of indexes to use as train, valid and test masks for outer and inner validation loops.
        The proportion of each class is conserved within each split.

        Returns
        -------
        masks : Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]]
            Dictionary of dictionaries with list of indexes.

        Example
        -------
            {0:
                {
                    'train': [..],
                    'valid': [..],
                    'test': [..],
                    'inner':
                        {0:
                            {
                                'train': [..],
                                'valid': [..],
                                'test': [..]
                            }
                        }
                },
            ...
            }
        """
        # We set targets to use for stratification
        targets = [target if self.is_categorical(target) else self.mimic_classes(target) for target in self.__targets]

        # We set the random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # We initialize the dict that will contain the results and the list of indexes to use
        masks, idx = {}, np.array(range(len(targets[0])))

        # We save a copy of targets in an array
        targets_c = deepcopy(targets)

        with tqdm(total=(self.n_out_split + self.n_out_split*self.n_in_split)) as bar:
            for i in range(self.n_out_split):

                # We create outer split masks
                masks[i] = {**self.split(idx, targets_c), Mask.INNER: {}}
                bar.update()

                for j in range(self.n_in_split):

                    # We create the inner split masks
                    masks[i][Mask.INNER][j] = self.split(masks[i][Mask.TRAIN], targets_c)
                    bar.update()

        # We turn arrays of idx into lists of idx
        self.serialize_masks(masks)

        return masks

    def __define_split_function(
            self,
            test_size: float,
            valid_size: float
    ) -> Callable:
        """
        Defines the split function according to the valid size.

        Parameters
        ----------
        test_size : float
            Percentage of data taken to create the train indexes set.
        valid_size : float
            Percentage of data taken to create the validation indexes set.

        Returns
        -------
        split : Callable
            Split function.
        """

        if valid_size > 0:

            # Split must extract train, valid and test masks
            def split(idx: np.array, targets: List[np.array]) -> Dict[str, np.array]:
                train_mask, valid_mask, test_mask = None, None, None
                # We make a pseudo-multilabel
                targets_str = targets[0].astype(str)
                for i in range(len(targets)-1):
                    targets_str = np.char.add(targets_str, targets[i+1].astype(str))

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    remaining_idx, test_mask = train_test_split(idx, stratify=targets_str[idx], test_size=test_size)
                    train_mask, valid_mask = train_test_split(remaining_idx, stratify=targets_str[remaining_idx],
                                                              test_size=valid_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask, valid_mask)
                    nb_tries_remaining -= 1

                if not mask_ok:
                    raise Exception("The sampler could not find a proper train, valid and test split.")

                return {Mask.TRAIN: train_mask, Mask.VALID: valid_mask, Mask.TEST: test_mask}
        else:
            # Split must extract train and test masks only
            def split(idx: np.array, targets: List[np.array]) -> Dict[str, np.array]:
                train_mask, test_mask = None, None
                # We make a pseudo-multilabel
                targets_str = targets[0].astype(str)
                for i in range(len(targets)-1):
                    targets_str = np.char.add(targets_str, targets[i+1].astype(str))

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    train_mask, test_mask = train_test_split(idx, stratify=targets_str[idx], test_size=test_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask)
                    nb_tries_remaining -= 1

                if not mask_ok:
                    raise Exception("The sampler could not find a proper train and test split")

                return {Mask.TRAIN: train_mask, Mask.VALID: None, Mask.TEST: test_mask}

        return split

    def check_masks_validity(
            self,
            train_mask: List[int],
            test_mask: List[int],
            valid_mask: Optional[List[int]] = None
    ) -> bool:
        """
        Valid if categorical and numerical variables of other masks are out of the range of train mask.

        Parameters
        ----------
        train_mask : List[int]
            List of idx to use for training.
        test_mask : List[int]
            List of idx to use for test.
        valid_mask : Optional[List[int]]
            List of idx to use for validation.

        Returns
        -------
        valid : bool
            Whether the masks are valid.
        """
        # We update the masks of the dataset
        self.__dataset.update_masks(train_mask, test_mask, valid_mask)

        # We extract train dataframe
        imputed_df = self.__dataset.get_imputed_dataframe()
        train_df = imputed_df.iloc[train_mask]

        # We check if all categories of categorical columns are in the training set
        for cat, values in self.__unique_encodings.items():
            for c in train_df[cat].unique():
                if c not in values:
                    return False

        # We save q1 and q3 of each numerical columns
        train_quantiles = {c: (train_df[c].quantile(0.25), train_df[c].quantile(0.75))
                           for c in self.__dataset.cont_features_cols}

        # We validate the other masks
        other_masks = [m for m in [valid_mask, test_mask] if m is not None]
        for mask in other_masks:

            # We extract the subset
            subset_df = imputed_df.iloc[mask]

            # We check if all numerical values are not extreme outliers according to the train mask
            for cont_col, (q1, q3) in train_quantiles.items():
                iqr = q3 - q1
                other_min, other_max = (subset_df[cont_col].min(), subset_df[cont_col].max())
                if other_min < q1 - self.alpha * iqr or other_max > q3 + self.alpha * iqr:
                    return False

            return True

    @staticmethod
    def is_categorical(
            targets: Union[Tensor, np.array]
    ) -> bool:
        """
        Checks if the number of unique values is lower than 15% of the length of the targets sequence.

        Parameters
        ----------
        targets : Union[Tensor, np.array]
            Sequence of float/int used for stratification.

        Returns
        -------
        categorical : bool
            Whether the target is a categorical variable.
        """
        target_list_copy = targets.tolist()
        return len(set(target_list_copy)) < 0.15*len(target_list_copy)

    @staticmethod
    def mimic_classes(
            targets: Union[Tensor, np.array, List[Any]]
    ) -> np.array:
        """
        Creates fake classes array out of real-valued targets sequence using quartiles.

        Parameters
        ----------
        targets : Union[Tensor, np.array, List[Any]]
            Sequence of float/int used for stratification.

        Returns
        -------
        fake_classes : np.array
            Array with fake classes.
        """
        return pd.qcut(np.array(targets), 2, labels=False)

    @staticmethod
    def serialize_masks(
            masks: Dict[int, Dict[str, Union[np.array, Dict[str, np.array]]]]
    ) -> None:
        """
        Turns all numpy arrays of idx into lists of idx.

        Parameters
        ----------
        masks : Dict[int, Dict[str, Union[np.array, Dict[str, np.array]]]]
            Dictionary of masks
        """
        mask_keys = [Mask.TRAIN, Mask.VALID, Mask.TEST]
        for k, v in masks.items():
            for t1 in mask_keys:
                masks[k][t1] = v[t1].tolist() if v[t1] is not None else None
            for in_k, in_v in masks[k][Mask.INNER].items():
                for t2 in mask_keys:
                    masks[k][Mask.INNER][in_k][t2] = in_v[t2].tolist() if in_v[t2] is not None else None

    @staticmethod
    def visualize_splits(
            datasets: dict
    ) -> None:
        """
        Details the data splits for the experiment.

        Parameters
        ----------
        datasets : dict
            Dict with all the masks obtained from the sampler.
        """
        print("#----------------------------------#")
        for k, v in datasets.items():
            print(f"Split {k+1} \n")
            print(f"Outer :")
            valid = v[Mask.VALID] if v[Mask.VALID] is not None else []
            print(f"Train {len(v[Mask.TRAIN])} - Valid {len(valid)} - Test {len(v[Mask.TEST])}")
            if v[Mask.INNER]:
                print(f"{Mask.INNER} :")
            for k1, v1 in v[Mask.INNER].items():
                valid = v1[Mask.VALID] if v1[Mask.VALID] is not None else []
                print(f"{k+1}.{k1} -> Train {len(v1[Mask.TRAIN])} - Valid {len(valid)} -"
                      f" Test {len(v1[Mask.TEST])}")
            print("#----------------------------------#")


# ---------------------- THE FUNCTION extract_masks BE INCLUDED IN THE STRATIFIER SAMPLER CLASS ---------------------- #
from json import load


def extract_masks(
        file_path: str,
        k: int = 20,
        l: int = 20
) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
    """
    Extract masks saved in json file.

    Parameters
    ----------
    file_path : str
        Path of json file containing the masks.
    k : int
        Number of outer loops to extract.
    l : int
        Number of inner loops to extract.

    Returns
    -------
    masks
    """
    # Opening JSON file
    f = open(file_path)

    # Extract complete masks
    all_masks = load(f)

    # Extraction of masks subset
    mask_keys = [Mask.TRAIN, Mask.VALID, Mask.TEST]
    masks = {}
    for i in map(str, range(k)):
        int_i = int(i)
        masks[int_i] = {}
        for t in mask_keys:
            masks[int_i][t] = all_masks[i][t]
        masks[int_i][Mask.INNER] = {}
        for j in map(str, range(l)):
            int_j = int(j)
            masks[int_i][Mask.INNER][int_j] = {}
            for t in mask_keys:
                masks[int_i][Mask.INNER][int_j][t] = all_masks[i][Mask.INNER][j][t]

    # Closing file
    f.close()

    return masks
