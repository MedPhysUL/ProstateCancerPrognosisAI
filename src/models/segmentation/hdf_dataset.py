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
from monai.transforms import CenterSpatialCrop, CropForeground, SpatialCrop
import torch


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
        img_list, seg_list = [], []
        for patient in file.keys():
            if file[patient]['0'].attrs['Modality'] == "CT":
                img = np.transpose(np.array(file[patient]['0']['image']), (2, 0, 1))
                # seg = np.transpose(np.array(file[patient]['0']['0']['Prostate_label_map']), (2, 0, 1))
                seg = np.transpose(np.array(file[patient]['0']['Prostate_label_map']), (2, 0, 1))

                img_cropped, seg_cropped = self._crop(img=img, seg=seg)
                img_list.append(img_cropped)
                seg_list.append(seg_cropped)

            else:
                img = np.transpose(np.array(file[patient]['1']['image']), (2, 0, 1))
                # seg = np.transpose(np.array(file[patient]['1']['0']['Prostate_label_map']), (2, 0, 1))
                seg = np.transpose(np.array(file[patient]['1']['Prostate_label_map']), (2, 0, 1))

                img_cropped, seg_cropped = self._crop(img=img, seg=seg)
                img_list.append(img_cropped)
                seg_list.append(seg_cropped)

        super().__init__(img=img_list, seg=seg_list, img_transform=img_transform, seg_transform=seg_transform)

    def _crop(
            self,
            img,
            seg,
            z_dim=None,
            xy_dim=32
    ):
        """

        """
        img_cropped, start, end = CropForeground(return_coords=True)(img)
        seg_cropped = SpatialCrop(roi_start=start, roi_end=end)(seg)

        # img_cropped = CenterSpatialCrop(roi_size=(10000, 20, 20))(img_cropped)
        # seg_cropped = CenterSpatialCrop(roi_size=(10000, 20, 20))(seg_cropped)

        if z_dim:
            img_cropped, seg_cropped = img_cropped[z_dim[0]:z_dim[1]], seg_cropped[z_dim[0]:z_dim[1]]
            return img_cropped, seg_cropped

        return img_cropped, seg_cropped
