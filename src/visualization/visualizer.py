"""
    @file:              visualizer.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains the Visualizer class which is used to visualize a patient's image (whether
                        it is a PET or a CT) alongside its respective segmentation. Sliders allow the user to slice
                        through the coronal, sagittal and axial views.
"""

from enum import IntEnum
from functools import partial
from typing import Dict, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class AnatomicalPlane(IntEnum):
    ALL = -1,
    CORONAL = 0,
    SAGITTAL = 1,
    AXIAL = 2


class Visualizer:
    """
    A class which is used to visualize a patient's image (whether it is a PET or a CT) alongside its respective
    segmentation. Sliders allow the user to slice through the coronal, sagittal and axial views.
    """

    @staticmethod
    def _visualize_plane(
            plane: AnatomicalPlane,
            img: np.ndarray,
            seg: np.ndarray,
            axes: np.ndarray
    ) -> Dict[str, Union[np.ndarray, Slider, plt.axes]]:
        """
        Initializes the figures.

        Parameters
        ----------
        plane : AnatomicalPlane
            The anatomical plane to visualize. Coronal, sagittal or axial.
        img : np.ndarray
            An image array.
        seg : np.ndarray
            A segmentation map array.
        axes : plt.axes
            Axes.

        Returns
        -------
        returns : Dict[np.ndarray, np.ndarray, Slider, plt.axes, plt.axes]
            A dictionary pertaining to visualization.
        """
        max_slice = img.shape[plane.value] - 1
        img_max = np.max(img)
        img_min = np.min(img)

        img_plane_in_first_axis = np.moveaxis(img, plane.value, 0)
        initial_img = img_plane_in_first_axis[int(max_slice / 2)]
        seg_plane_in_first_axis = np.moveaxis(seg, plane.value, 0)
        initial_seg = seg_plane_in_first_axis[int(max_slice / 2)]

        img_3d = axes[plane.value].imshow(
            initial_img,
            cmap="Greys_r",
            vmax=img_max,
            vmin=img_min
        )
        seg_3d = axes[plane.value].imshow(
            initial_seg,
            vmax=1,
            vmin=0,
            alpha=0.1
        )

        ax_slice = plt.axes(
            [0.14 + plane.value * 0.27, 0.15, 0.2, 0.02],
            facecolor='lightgoldenrodyellow'
        )
        slice_index = Slider(
            ax_slice,
            'Slice',
            0,
            max_slice,
            valinit=int(max_slice / 2),
            valstep=1
        )

        returns = {
            "img_plane_in_first_axis": img_plane_in_first_axis,
            "seg_plane_in_first_axis": seg_plane_in_first_axis,
            "slice_index": slice_index,
            "img_3d": img_3d,
            "seg_3d": seg_3d
        }

        return returns

    @staticmethod
    def _update_plane(
            val,
            plane: AnatomicalPlane,
            set_up: Dict[str, Union[np.ndarray, Slider, plt.axes]]
    ):
        """
        Updates the figure associated to a slider.

        Parameters
        ----------
        plane : AnatomicalPlane
            The anatomical plane to update. Coronal, sagittal or axial.
        set_up : Dict[str, Union[np.ndarray, Slider, plt.axes]]
            A dictionary output by the _visualize_plane method.
        """
        slice_value = set_up["slice_index"].val
        new_img = set_up["img_plane_in_first_axis"][slice_value]
        new_seg = set_up["seg_plane_in_first_axis"][slice_value]
        set_up["img_3d"].set_data(new_img)
        set_up["seg_3d"].set_data(new_seg)
        plt.draw()

    def visualize(
            self,
            img,
            seg
    ):
        """
        Visualizes a patient's image (whether it is a PET or a CT) alongside its respective segmentation. Sliders allow
        the user to slice through the coronal, sagittal and axial views.

        Parameters
        ----------
        img : np.ndarray
            An image array.
        seg : np.ndarray
            A segmentation map array.
        """
        fig, axes = plt.subplots(1, 3)
        plt.subplots_adjust(bottom=0.20)

        set_up_coronal = self._visualize_plane(plane=AnatomicalPlane.CORONAL, img=img, seg=seg, axes=axes)
        set_up_sagittal = self._visualize_plane(plane=AnatomicalPlane.SAGITTAL, img=img, seg=seg, axes=axes)
        set_up_axial = self._visualize_plane(plane=AnatomicalPlane.AXIAL, img=img, seg=seg, axes=axes)

        set_up_coronal["slice_index"].on_changed(partial(
            self._update_plane,
            plane=AnatomicalPlane.CORONAL,
            set_up=set_up_coronal
        ))
        set_up_sagittal["slice_index"].on_changed(partial(
            self._update_plane,
            plane=AnatomicalPlane.SAGITTAL,
            set_up=set_up_sagittal
        ))
        set_up_axial["slice_index"].on_changed(partial(
            self._update_plane,
            plane=AnatomicalPlane.AXIAL,
            set_up=set_up_axial
        ))

        plt.show()





