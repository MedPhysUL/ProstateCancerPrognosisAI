"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file.
"""

import numpy as np
from typing import Callable, Optional, Sequence

import h5py
from torch.utils.data import Dataset
from monai.data import ArrayDataset


class HDFDataset(ArrayDataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file.
    """
    def __init__(
            self,
            path: str,
            img_transform: Optional[Callable] = None,
            seg_transform: Optional[Callable] = None
    ):
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.

        Parameters
        ----------
        path : str
            The path to the HDF5 file that contains all the patients' folders.
        img_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the image.
        seg_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the segmentation.
        """
        file = h5py.File(path)
        img, seg = [], []
        for patient in file.keys():
            try:
                if file[patient]['0'].attrs['Modality'] == "CT":
                    img.append(np.array(file[patient]['0']['image']))
                    seg.append(np.array(file[patient]['0']['Prostate_label_map']))
                else:
                    img.append(np.array(file[patient]['1']['image']))
                    seg.append(np.array(file[patient]['1']['Prostate_label_map']))
            except KeyError:
                print(f"Patient {patient} ignored.")

        super().__init__(img=img, seg=seg, img_transform=img_transform, seg_transform=seg_transform)
