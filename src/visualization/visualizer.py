"""
    @file:              visualizer.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       Description.

"""

from enum import IntEnum
from typing import Dict, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class AnatomicalPlane(IntEnum):
    ALL = -1
    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2


class Visualizer:
    def visualize(
            self,
            img: np.ndarray,
            seg: np.ndarray
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Create a 3D figure of a given image array in all anatomical plane views.

        Parameters
        ----------
        img : np.ndarray
            An image array.
        seg : np.ndarray
            A segmentation mask array.

        Returns
        -------
        fig, axes : Tuple[plt.figure, plt.axes]
            Figure and axes.
        """
        fig, axes = plt.subplots(1, 3)
        plt.subplots_adjust(bottom=0.20)
        image_max = np.max(img)
        anatomical_planes = AnatomicalPlane

        # Axial
        image_with_axial_plane_moved_to_first_axis = np.moveaxis(img, anatomical_planes.AXIAL, 0)
        max_axial_slice = img.shape[anatomical_planes.AXIAL] - 1
        initial_axial_image = image_max - image_with_axial_plane_moved_to_first_axis[int(max_axial_slice / 2)]
        seg_with_axial_plane_moved_to_first_axis = np.moveaxis(seg, anatomical_planes.AXIAL, 0)
