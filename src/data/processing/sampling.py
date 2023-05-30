"""
    @file:              sampling.py
    @Author:            Maxence Larose, Nicolas Raymond, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains the 'Sampler' class used to separate test sets and valid sets from train
                        sets. Also contains few functions used to extract specific datasets.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
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
            random_state: Optional[int] = None,
            alpha: int = 4
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
        random_state : Optional[int]
            Random state integer.
        alpha : int
            IQR multiplier used to check numerical variable range validity of the masks created
        """
        if n_out_split <= 0:
            raise ValueError('Number of outer split must be greater than 0')
        if n_in_split < 0:
            raise ValueError('Number of inner split must be greater or equal to 0')
        if not (0 <= valid_size < 1):
            raise ValueError('Validation size must be in the range [0, 1)')

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
        self.random_state = random_state
        self.valid_size = valid_size

        self.outer_skf = StratifiedKFold(n_splits=n_out_split, shuffle=True, random_state=random_state)
        self.inner_skf = StratifiedKFold(n_splits=n_in_split, shuffle=True, random_state=random_state)

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

        # We initialize the dict that will contain the results and the list of indexes to use
        masks, idx = {}, np.array(range(len(targets[0])))

        # We save a copy of targets in an array
        targets_c = deepcopy(targets)

        with tqdm(total=(self.n_out_split + self.n_out_split*self.n_in_split)) as bar:
            targets_str = targets_c[0].astype(str)
            for i in range(len(targets_c) - 1):
                targets_str = np.char.add(targets_str, targets_c[i + 1].astype(str))

            outer_splits = self.outer_skf.split(X=idx, y=targets_str[idx])
            for i, (outer_remaining_idx, outer_test_mask) in enumerate(outer_splits):

                outer_train_mask, outer_valid_mask = self._get_train_valid_masks(outer_remaining_idx, targets_str)

                # We create outer split masks
                masks[i] = {
                    **{Mask.TRAIN: outer_train_mask, Mask.VALID: outer_valid_mask, Mask.TEST: outer_test_mask},
                    Mask.INNER: {}
                }
                bar.update()

                inner_splits = self.inner_skf.split(X=outer_train_mask, y=targets_str[outer_train_mask])
                for j, (inner_remaining_idx, inner_test_mask) in enumerate(inner_splits):
                    inner_train_mask, inner_valid_mask = self._get_train_valid_masks(inner_remaining_idx, targets_str)

                    # We create the inner split masks
                    masks[i][Mask.INNER][j] = {
                        Mask.TRAIN: inner_train_mask, Mask.VALID: inner_valid_mask, Mask.TEST: inner_test_mask
                    }
                    bar.update()

        # We turn arrays of idx into lists of idx
        self.serialize_masks(masks)

        return masks

    def _get_train_valid_masks(
            self,
            remaining_idx: np.ndarray,
            targets_str: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[int]]]:
        """
        Returns lists of indexes to use as train and valid masks. The proportion of each class is conserved within
        each split.

        Parameters
        ----------
        remaining_idx : np.ndarray
            Array of indexes to split.
        targets_str : Union[np.ndarray, List[int]]
            Array of targets used for stratification.
        """
        if np.isclose(self.valid_size, 0.0):
            return remaining_idx, []
        else:
            return train_test_split(
                remaining_idx,
                stratify=targets_str[remaining_idx],
                test_size=self.valid_size,
                random_state=self.random_state
            )

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
