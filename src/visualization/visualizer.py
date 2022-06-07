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
    """

    """

    def _visualize_axial():
        pass
    def _visualize_coronal():
        pass
    def _
    @staticmethod
    def visualize(
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
        img_max = np.max(img)
        img_min = np.min(img)


        # Axial
        img_with_axial_plane_moved_to_first_axis = np.moveaxis(img, AnatomicalPlane.AXIAL, 0)
        max_axial_slice = img.shape[AnatomicalPlane.AXIAL] - 1
        initial_axial_img = img_with_axial_plane_moved_to_first_axis[int(max_axial_slice / 2)]
        seg_with_axial_plane_moved_to_first_axis = np.moveaxis(seg, AnatomicalPlane.AXIAL, 0)
        initial_axial_seg = seg_with_axial_plane_moved_to_first_axis[int(max_axial_slice / 2)]

        axial_3d_img = axes[AnatomicalPlane.AXIAL].imshow(
            initial_axial_img,
            cmap="Greys_r",
            vmax=img_max,
            vmin=img_min
        )

        axial_3d_seg = axes[AnatomicalPlane.AXIAL].imshow(
            initial_axial_seg,
            vmax=1,
            vmin=0,
            alpha=0.1
        )

        ax_slice = plt.axes(
            [0.14 + AnatomicalPlane.AXIAL.value * 0.27, 0.15, 0.2, 0.02],
            facecolor='lightgoldenrodyellow'
        )

        axial_slice_index = Slider(
            ax_slice,
            'Slice',
            0,
            max_axial_slice,
            valinit=int(max_axial_slice / 2),
            valstep=1
        )

        def axial_image_update(val):
            slice_value = axial_slice_index.val
            new_img = img_with_axial_plane_moved_to_first_axis[slice_value]
            new_seg = seg_with_axial_plane_moved_to_first_axis[slice_value]
            axial_3d_img.set_data(new_img)
            axial_3d_seg.set_data(new_seg)
            plt.draw()

        axial_slice_index.on_changed(axial_image_update)

        # Coronal
        img_with_coronal_plane_moved_to_first_axis = np.moveaxis(img, AnatomicalPlane.CORONAL, 0)
        max_coronal_slice = img.shape[AnatomicalPlane.CORONAL] - 1
        initial_coronal_img = img_with_coronal_plane_moved_to_first_axis[int(max_coronal_slice / 2)]

        seg_with_coronal_plane_moved_to_first_axis = np.moveaxis(seg, AnatomicalPlane.CORONAL, 0)
        initial_coronal_seg = seg_with_coronal_plane_moved_to_first_axis[int(max_coronal_slice / 2)]

        coronal_3d_img = axes[AnatomicalPlane.CORONAL].imshow(
            initial_coronal_img,
            cmap="Greys_r",
            vmax=img_max,
            vmin=img_min
        )

        coronal_3d_seg = axes[AnatomicalPlane.CORONAL].imshow(
            initial_coronal_seg,
            vmax=1,
            vmin=0,
            alpha=0.1
        )

        ax_slice = plt.axes(
            [0.14 + AnatomicalPlane.CORONAL.value * 0.27, 0.15, 0.2, 0.02],
            facecolor='lightgoldenrodyellow'
        )

        coronal_slice_index = Slider(
            ax_slice,
            'Slice',
            0,
            max_coronal_slice,
            valinit=int(max_coronal_slice / 2),
            valstep=1
        )

        def coronal_image_update(val):
            slice_value = coronal_slice_index.val
            new_img = img_with_coronal_plane_moved_to_first_axis[slice_value]
            new_seg = seg_with_coronal_plane_moved_to_first_axis[slice_value]
            coronal_3d_img.set_data(new_img)
            coronal_3d_seg.set_data(new_seg)
            plt.draw()

        coronal_slice_index.on_changed(coronal_image_update)

        # Sagittal
        img_with_sagittal_plane_moved_to_first_axis = np.moveaxis(img, AnatomicalPlane.SAGITTAL, 0)

        max_sagittal_slice = img.shape[AnatomicalPlane.SAGITTAL] - 1

        initial_sagittal_img = img_with_sagittal_plane_moved_to_first_axis[int(max_sagittal_slice / 2)]

        seg_with_sagittal_plane_moved_to_first_axis = np.moveaxis(seg, AnatomicalPlane.SAGITTAL, 0)

        initial_sagittal_seg = seg_with_sagittal_plane_moved_to_first_axis[int(max_sagittal_slice / 2)]

        sagittal_3d_img = axes[AnatomicalPlane.SAGITTAL].imshow(
            initial_sagittal_img,
            cmap="Greys_r",
            vmax=img_max,
            vmin=img_min

        )

        sagittal_3d_seg = axes[AnatomicalPlane.SAGITTAL].imshow(
            initial_sagittal_seg,
            vmax=1,
            vmin=0,
            alpha=0.1
        )

        ax_slice = plt.axes(
            [0.14 + AnatomicalPlane.SAGITTAL.value * 0.27, 0.15, 0.2, 0.02],
            facecolor='lightgoldenrodyellow'
        )

        sagittal_slice_index = Slider(
            ax_slice,
            'Slice',
            0,
            max_sagittal_slice,
            valinit=int(max_sagittal_slice / 2),
            valstep=1
        )

        def sagittal_image_update(val):
            slice_value = sagittal_slice_index.val
            new_img = img_with_sagittal_plane_moved_to_first_axis[slice_value]
            new_seg = seg_with_sagittal_plane_moved_to_first_axis[slice_value]
            sagittal_3d_img.set_data(new_img)
            sagittal_3d_seg.set_data(new_seg)
            plt.draw()

        sagittal_slice_index.on_changed(sagittal_image_update)
        
        plt.show()
        return fig, axes






