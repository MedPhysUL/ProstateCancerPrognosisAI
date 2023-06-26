"""
    @file:              image.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 05/2023

    @Description:       This file contains a class used to create a dataset containing multiple patient images and
                        segmentations from a given DELIA database.
"""

import collections.abc
from itertools import chain
from typing import Dict, List, NamedTuple, Optional, Union, Sequence, Set, Tuple

from delia.databases.patients_database import PatientsDatabase
from monai.data import MetaTensor
from monai.transforms import apply_transform, Compose, EnsureChannelFirstd, MapTransform, ToTensord
import numpy as np
from torch import cuda, float32
from torch import device as torch_device
from torch.utils.data import Dataset, Subset

from ...tasks import SegmentationTask, TaskList


class ImageDataModel(NamedTuple):
    """
    Image data element named tuple. This tuple is used to separate features (x) and targets (y) where
        - x : I-dimensional dictionary containing (N, Z, X, Y) tensor where I is the number of images used as features.
        - y : T-dimensional dictionary containing (N, Z, X, Y) tensor where T is the number of tasks.
    """
    x: Dict[str, MetaTensor]
    y: Dict[str, MetaTensor]


class ImageDataset(Dataset):
    """
    A class used to create a dataset containing multiple patient images and segmentations from a given DELIA database.
    The rendered images are in shape (Z, X, Y) by default.
    """

    def __init__(
            self,
            database: PatientsDatabase,
            modalities: Set[str],
            augmentations: Optional[Union[Compose, MapTransform]] = None,
            device: Optional[torch_device] = None,
            organs: Optional[Dict[str, Set[str]]] = None,
            seed: Optional[int] = None,
            tasks: Optional[Union[SegmentationTask, TaskList, List[SegmentationTask]]] = None,
            transposition: Tuple[int, int, int] = (2, 0, 1),
            **kwargs
    ):
        """
        Creates a dataset containing multiple patient images and segmentations from a given DELIA database.

        Parameters
        ----------
        database : PatientsDatabase
            A DELIA database that is used to interact with the HDF5 file that contains all the patients' folders.
        modalities : Set[str]
            Set of image modalities to include in the dataset. These images are added to the dataset as features.
                    Example : {"CT", "PT", "MR"}.
        augmentations : Optional[Union[Compose, MapTransform]]
            A single or a sequence of transforms to apply to images and segmentations (depending on transform keys).
            These transforms are applied to the dataset only if the method 'enable_augmentations' is called, usually
            before training.
        device : Optional[torch_device]
            Device to use for performing augmentations. If None, the device is set to 'cuda' if available, otherwise
            it is set to 'cpu'.
        organs : Dict[str, Set[str]]
            Dictionary of organs to include in the dataset. Keys are modality names and values are sets of organs. The
            keys of the dictionary, i.e. modality images (CT scan for example), will NOT be added to the dataset. This
            parameter is only used to add the organs. These label maps are added to the dataset as features.
                    Example : {
                        "CT": {"Prostate", "Bladder"},
                        "PT": {"Prostate"},
                        "MR": {"Brain"}
                    }.
        seed : Optional[int]
            Random state used for reproducibility.
        tasks : Optional[Union[SegmentationTask, TaskList, List[SegmentationTask]]]
            Segmentation tasks to perform. These label maps are added to the dataset as targets.
        transposition : Tuple[int, int, int]
            The transposition to apply to images before applying transforms. The rendered images are in shape (Z, X, Y)
            by default.
        **kwargs : dict
            Keywords arguments controlling images and segmentations format, and segmentations series to use.
        """
        self._tasks = TaskList(tasks)
        assert all(isinstance(task, SegmentationTask) for task in self._tasks), (
            f"All tasks must be instances of 'SegmentationTask'."
        )

        self._augmentations = augmentations
        self._augmentations_are_enabled = False
        self._database = database
        self._device = device if device else torch_device("cuda") if cuda.is_available() else torch_device("cpu")
        self._modalities = modalities
        self._organs = organs if organs else {}
        self._rng = np.random.RandomState(seed=seed)
        self._modalities_to_iterate_over = set(
            chain(
                modalities,
                self._organs.keys(),
                [t.modality for t in self._tasks]
            )
        )
        self._transposition = transposition

        self._organ_key_getter = kwargs.get("organ_key_getter", lambda modality, organ: f"{modality}_{organ}")
        self._seg_series = kwargs.get("seg_series", "0")

        self._task_key_to_task_name_mapping = {task.key: task.name for task in self._tasks}

    def __len__(self) -> int:
        """
        Gets dataset length.

        Returns
        -------
        length : int
            Length of the database.
        """
        return len(self._database)

    def __getitem__(
            self,
            index: Union[int, slice, Sequence[int]]
    ) -> Union[ImageDataModel, Subset]:
        """
        Gets specific items in the dataset.

        Parameters
        ----------
        index : Union[int, slice, Sequence[int]]
            The index of the items to get.

        Returns
        -------
        item : Union[ImageDataModel, Subset]
            A single patient data or a subset of the dataset.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        elif isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        elif isinstance(index, int):
            data = self._get_patient_data(index)
            transformed_data = self._transform(data)

            x_image, y_image = {}, {}
            for key, item in transformed_data.items():
                if key in [task.name for task in self.tasks]:
                    y_image[key] = item
                else:
                    x_image[key] = item

            return ImageDataModel(x=x_image, y=y_image)
        else:
            raise AssertionError(f"'index' must be of type 'int', 'slice' or 'Sequence[int]'. Found type {type(index)}")

    @property
    def tasks(self) -> TaskList:
        """
        Gets the tasks of the dataset.

        Returns
        -------
        tasks : TaskList
            The tasks of the dataset.
        """
        return self._tasks

    def _get_patient_data(self, index: int) -> Dict[str, np.ndarray]:
        """
        Gets a single patient images data from its index in the dataset.

        Parameters
        ----------
        index : int
            The index of the patient whose data needs to be retrieved

        Returns
        -------
        data : Dict[str, np.ndarray]
            A patient images data.
        """
        patient_group = self._database[index]

        # print(f"Loading {patient_group.name[1:]}.")  # TODO : Use logging instead of a print.
        img_dict, seg_dict = {}, {}
        for series_number in patient_group.keys():
            series = patient_group[series_number]
            for modality in self._modalities_to_iterate_over:
                if series.attrs[self._database.MODALITY] == modality:
                    if modality in self._modalities:
                        img_dict[modality] = self._transpose(series[self._database.IMAGE])

                    if modality in self._organs.keys():
                        for organ in self._organs[modality]:
                            seg_array = series[self._seg_series][organ]
                            seg_dict[self._organ_key_getter(modality, organ)] = self._transpose(seg_array)

                    for task in self.tasks.segmentation_tasks:
                        if modality == task.modality:
                            seg_array = series[self._seg_series][task.organ]
                            seg_dict[task.key] = self._transpose(seg_array)

        return dict(img_dict, **seg_dict)

    def enable_augmentations(self):
        """
        Enables augmentations on the dataset. This method should be called before training.
        """
        self._augmentations_are_enabled = True

    def disable_augmentations(self):
        """
        Disables augmentations on the dataset. This method should be called before validation and testing.
        """
        self._augmentations_are_enabled = False

    def _transform(self, data: Dict[str, np.ndarray]) -> Dict[str, MetaTensor]:
        """
        Transforms images and segmentations.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            A dictionary containing 3D images or segmentation maps of a single patient.

        Returns
        -------
        transformed_data : Dict[str, MetaTensor]
            The dictionary of transformed images and segmentation maps.
        """
        keys = list(data.keys())

        if self._augmentations is None or not self._augmentations_are_enabled:
            transforms = Compose([
                EnsureChannelFirstd(keys=keys),
                ToTensord(keys=keys, dtype=float32, device=self._device)
            ])
        else:
            if isinstance(self._augmentations, Compose):
                transforms = Compose([
                    EnsureChannelFirstd(keys=keys),
                    ToTensord(keys=keys, dtype=float32, device=self._device),
                    *self._augmentations.transforms
                ])
            elif isinstance(self._augmentations, MapTransform):
                transforms = Compose([
                    EnsureChannelFirstd(keys=keys),
                    ToTensord(keys=keys, dtype=float32, device=self._device),
                    self._augmentations,
                ])
            else:
                raise AssertionError(
                    f"'augmentations' must be of type 'Compose' or 'MapTransform'. Found type "
                    f"{type(self._augmentations)}"
                )

        random_seed = self._rng.randint(0, 2**16 - 1)
        transforms = transforms.set_random_state(random_seed)

        transformed_data = apply_transform(transforms, data) if transforms else data
        transformed_data = self._transform_task_key_to_task_name(transformed_data)

        return transformed_data

    def _transpose(self, array: np.ndarray) -> np.array:
        """
        Gets transposed data.

        Parameters
        ----------
        array : np.array
            A 3D image or segmentation map.

        Returns
        -------
        transposed_array : np.array
            The transposed array.
        """
        return np.transpose(np.array(array), self._transposition)

    def _transform_task_key_to_task_name(
            self,
            data: Dict[str, Union[np.array, MetaTensor]]
    ) -> Dict[str, Union[np.array, MetaTensor]]:
        """
        Transforms task keys to task names. This is useful when the task keys are different from the task names.

        Parameters
        ----------
        data : Dict[str, Union[np.array, MetaTensor]]
            A dictionary containing 3D images or segmentation maps of a single patient.

        Returns
        -------
        transformed_data : Dict[str, Union[np.array, MetaTensor]]
            The dictionary of transformed images and segmentation maps.
        """
        transformed_data = {}
        for key, value in data.items():
            if key in self._task_key_to_task_name_mapping.keys():
                transformed_data[self._task_key_to_task_name_mapping[key]] = value
            else:
                transformed_data[key] = value

        return transformed_data
