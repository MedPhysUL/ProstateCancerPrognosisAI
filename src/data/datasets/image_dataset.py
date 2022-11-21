"""
    @file:              image_dataset.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file. The foreground is cropped and a crop along Z
                        can be specified.
"""

import collections.abc
from typing import Callable, List, NamedTuple, Optional, Union, Sequence, Set

from delia.databases.patients_database import PatientsDatabase
from monai.transforms import apply_transform
import numpy as np
from torch.utils.data import Dataset, Subset

from src.utils.tasks import SegmentationTask


class ImageDataset(Dataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file. The rendered images are in shape (Z, X, Y).
    """

    def __init__(
            self,
            database: PatientsDatabase,
            tasks: List[SegmentationTask],
            modalities: Set[str],
            transforms: Optional[Callable] = None
    ) -> None:
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.
        Images and segmentation maps are rendered in shape (Z, X, Y).

        Parameters
        ----------
        database : PatientsDatabase
            A database that is used to interact with the HDF5 file that contains all the patients' folders.
        tasks : List[SegmentationTask]
            Task to perform.
        transforms : Optional[Callable]
            A single or a sequence of transforms to apply to images and segmentations (depending on transform keys).
        z_dim : ZDimension
            A tuple that specify the z-dimension crop.
        """
        self._db = database
        self._modalities = modalities
        self._tasks = tasks
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        data_i = self._get_single_patient(index)

        return self._transform(data_i)

    def _get_single_patient(self, index: int):
        patient_group = self._db[index]

        print(f"Loading {patient_group.name[1:]}.")
        img_dict, seg_dict = {}, {}
        for series_number in patient_group.keys():
            series = patient_group[series_number]
            for modality in self._modalities:
                if series.attrs[self._db.MODALITY] == modality:
                    img_dict[modality] = self._transpose(data=series[self._db.IMAGE]).astype(np.float16)

                    for task in self._tasks:
                        if modality == task.modality:
                            seg_dict[task.organ] = self._transpose(data=series["0"][task.organ]).astype(np.int8)

        return dict(img_dict, **seg_dict)

    @property
    def tasks(self) -> List[SegmentationTask]:
        return self._tasks

    def _transform(self, data_i):
        return apply_transform(self._transforms, data_i) if self._transforms is not None else data_i

    @staticmethod
    def _transpose(data) -> np.array:
        return np.transpose(np.array(data), (2, 0, 1))
