#
# Présentement ne fait que prendre la seg 0 mais ultimement on veut : SEG > RTSTRCT (soit 0 ou 1).
#
"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 06/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file.
"""

import numpy as np
from typing import Callable, Optional

import h5py
from monai.data import ArrayDataset


class HDFDataset(ArrayDataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file. The rendered images are in shape (Z x X x Y).
    """
    def __init__(
            self,
            path: str,
            img_transform: Optional[Callable] = None,
            seg_transform: Optional[Callable] = None
    ):
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.
        Images and segmentations are in shape (Z, X, Y).

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
            if file[patient]['0'].attrs['Modality'] == "CT":
                img.append(np.transpose(np.array(file[patient]['0']['image']), (2, 0, 1)))
                seg.append(np.transpose(np.array(file[patient]['0']['0']['Prostate_label_map']), (2, 0, 1)))
            else:
                img.append(np.transpose(np.array(file[patient]['1']['image']), (2, 0, 1)))
                seg.append(np.transpose(np.array(file[patient]['1']['0']['Prostate_label_map']), (2, 0, 1)))

        super().__init__(img=img, seg=seg, img_transform=img_transform, seg_transform=seg_transform)
