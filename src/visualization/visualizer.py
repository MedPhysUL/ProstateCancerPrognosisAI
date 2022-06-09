"""
    @file:              visualizer.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     06/2022
    @Last modification: 06/2022

    @Description:       This file contains the Visualizer class which is used to visualize a patient's image (whether
                        it is a PET or a CT) alongside its respective segmentation and to compare an image alongside its
                        ground truth segmentation map and a predicted segmentation map in axial view. Sliders allow the
                        user to slice through the plotted views.
"""

from enum import IntEnum
from functools import partial
from typing import NamedTuple, Union

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


class AnatomicalPlane(IntEnum):
    ALL = -1,
    CORONAL = 0,
    SAGITTAL = 1,
    AXIAL = 2


class Plot(IntEnum):
    IMAGE = 0,
    GROUND_TRUTH = 1,
    PREDICTION = 2


class Visualizer:
    """
    A class which is used to visualize a patient's image (whether it is a PET or a CT) alongside its respective
    segmentation. Sliders allow the user to slice through the coronal, sagittal and axial views.
    """

    class Config(NamedTuple):
        img_plane_in_first_axis: np.ndarray
        seg_plane_in_first_axis: Union[np.ndarray, None]
        slice_index: Union[Slider, None]
        img_3d: plt.axes
        seg_3d: Union[plt.axes, None]

    @staticmethod
    def _get_config(
            plane: AnatomicalPlane,
            img: np.ndarray,
            seg: np.ndarray,
            axes: plt.axes,
            **kwargs
    ) -> Config:
        """
        Initializes the figures for visualization.

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
        **kwargs
            cmap : str
                A color map for the image.
            alpha : float
                The opacity of the segmentation map, between 0 (transparent) and 1 (opaque).

        Returns
        -------
        returns : Config
            A Config pertaining to the visualization.
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
            cmap=kwargs.get("cmap", "Greys_r"),
            vmax=img_max,
            vmin=img_min
        )
        seg_3d = axes[plane.value].imshow(
            initial_seg,
            vmax=1,
            vmin=0,
            alpha=kwargs.get("alpha", 0.1)
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

        returns = Visualizer.Config(
            img_plane_in_first_axis=img_plane_in_first_axis,
            seg_plane_in_first_axis=seg_plane_in_first_axis,
            slice_index=slice_index,
            img_3d=img_3d,
            seg_3d=seg_3d
        )

        return returns

    @staticmethod
    def _update_plane(
            val,
            config: Config
    ):
        """
        Updates the figure associated to a slider.

        Parameters
        ----------
        config : Config
            A Config as output by the _get_config method.
        """
        slice_value = config.slice_index.val
        new_img = config.img_plane_in_first_axis[slice_value]
        new_seg = config.seg_plane_in_first_axis[slice_value]
        config.img_3d.set_data(new_img)
        config.seg_3d.set_data(new_seg)

        plt.draw()

    def visualize(
            self,
            img: np.ndarray,
            seg: np.ndarray,
            **kwargs
    ):
        """
        Plots a patient's image (whether it is a PET or a CT) alongside its respective segmentation. Sliders allow
        the user to slice through the coronal, sagittal and axial views.

        Parameters
        ----------
        img : np.ndarray
            An image array.
        seg : np.ndarray
            A segmentation map array.
        **kwargs
            cmap : str
                A color map for the image.
            alpha : float
                The opacity of the segmentation map, between 0 (transparent) and 1 (opaque).
        """
        fig, axes = plt.subplots(1, 3)
        plt.subplots_adjust(bottom=0.20)

        config_coronal = self._get_config(plane=AnatomicalPlane.CORONAL, img=img, seg=seg, axes=axes, **kwargs)
        config_sagittal = self._get_config(plane=AnatomicalPlane.SAGITTAL, img=img, seg=seg, axes=axes, **kwargs)
        config_axial = self._get_config(plane=AnatomicalPlane.AXIAL, img=img, seg=seg, axes=axes, **kwargs)

        config_coronal.slice_index.on_changed(
            partial(
                self._update_plane,
                config=config_coronal
            ))
        config_sagittal.slice_index.on_changed(
            partial(
                self._update_plane,
                config=config_sagittal
            ))
        config_axial.slice_index.on_changed(
            partial(
                self._update_plane,
                config=config_axial
            ))

        plt.show()

    @staticmethod
    def _get_config_compare(
            plot: Plot,
            img: np.ndarray,
            seg_truth: np.ndarray,
            seg_pred: np.ndarray,
            axes: plt.axes,
            **kwargs
    ) -> Config:
        """
        Initializes the figures.

        Parameters
        ----------
        plot : Plot
            What is to be plotted. Image, ground truth segmentation map or prediction map.
        img : np.ndarray
            An image array.
        seg_truth : np.ndarray
            A ground truth segmentation map array.
        seg_pred : np.ndarray
            A predicted segmentation map array.
        axes : plt.axes
            Axes.
        **kwargs
            cmap : str
                A color map for the image.
            alpha : float
                The opacity of the segmentation map, between 0 (transparent) and 1 (opaque).

        Returns
        -------
        returns : Config
            A Config pertaining to the visualization.
        """
        max_slice = img.shape[AnatomicalPlane.AXIAL.value] - 1
        img_max = np.max(img)
        img_min = np.min(img)
        img_axial = np.moveaxis(img, AnatomicalPlane.AXIAL.value, 0)
        initial_img = img_axial[int(max_slice / 2)]

        axes[plot.value].set_title(plot.name)

        img_3d = axes[plot].imshow(
            initial_img,
            cmap=kwargs.get("cmap", "Greys_r"),
            vmax=img_max,
            vmin=img_min
        )

        if plot.value == 1:
            seg_axial = np.moveaxis(seg_truth, AnatomicalPlane.AXIAL.value, 0)
            initial_seg = seg_axial[int(max_slice / 2)]

            seg_3d = axes[1].imshow(
                initial_seg,
                vmax=1,
                vmin=0,
                alpha=kwargs.get("alpha", 0.1)
            )

            ax_slice = plt.axes(
                [0.14 + 1 * 0.27, 0.15, 0.2, 0.02],
                facecolor='lightgoldenrodyellow'
            )
            slice_index = Slider(
                ax_slice,
                'Z-Slice',
                0,
                max_slice,
                valinit=int(max_slice / 2),
                valstep=1
            )

            returns = Visualizer.Config(
                img_plane_in_first_axis=img_axial,
                seg_plane_in_first_axis=seg_axial,
                slice_index=slice_index,
                img_3d=img_3d,
                seg_3d=seg_3d
            )
            return returns

        if plot.value == 2:
            seg_axial = np.moveaxis(seg_pred, AnatomicalPlane.AXIAL.value, 0)
            initial_seg = seg_axial[int(max_slice / 2)]

            seg_3d = axes[2].imshow(
                initial_seg,
                vmax=1,
                vmin=0,
                alpha=kwargs.get("alpha", 0.1)
            )

            returns = Visualizer.Config(
                img_plane_in_first_axis=img_axial,
                seg_plane_in_first_axis=seg_axial,
                slice_index=None,
                img_3d=img_3d,
                seg_3d=seg_3d
            )
            return returns

        returns = Visualizer.Config(
            img_plane_in_first_axis=img_axial,
            seg_plane_in_first_axis=None,
            slice_index=None,
            img_3d=img_3d,
            seg_3d=None
        )

        return returns

    @staticmethod
    def _update_compare(
            val,
            image_config: Config,
            truth_config: Config,
            pred_config: Config
    ):
        """
        Updates the figures when slicing with the slider.

        Parameters
        ----------
        image_config : Config
            An image Config as output by the _get_config_compare method.
        truth_config : Config
            A ground truth segmentation map Config as output by the _get_config_compare method.
        pred_config : Config
            A predicted segmentation map Config as output by the _get_config_compare method.
        """
        slice_value = truth_config.slice_index.val

        new_img = image_config.img_plane_in_first_axis[slice_value]
        new_img_truth = truth_config.img_plane_in_first_axis[slice_value]
        new_img_pred = pred_config.img_plane_in_first_axis[slice_value]
        new_seg_truth = truth_config.seg_plane_in_first_axis[slice_value]
        new_seg_pred = pred_config.seg_plane_in_first_axis[slice_value]

        image_config.img_3d.set_data(new_img)
        truth_config.img_3d.set_data(new_img_truth)
        pred_config.img_3d.set_data(new_img_pred)
        truth_config.seg_3d.set_data(new_seg_truth)
        pred_config.seg_3d.set_data(new_seg_pred)

        plt.draw()

    def compare(
            self,
            img: np.ndarray,
            seg_truth: np.ndarray,
            seg_pred: np.ndarray,
            **kwargs
    ):
        """
        Plots in axial view a patient's image (whether it is a PET or a CT) alongside its ground truth segmentation and
        its predicted segmentation. A slider allows the user to slice through the body.

        Parameters
        ----------
        img : np.ndarray
            An image array.
        seg_truth : np.ndarray
            A ground truth segmentation map array.
        seg_pred : np.ndarray
            A predicted segmentation map array.
        **kwargs
            cmap : str
                A color map for the image.
            alpha : float
                The opacity of the segmentation map, between 0 (transparent) and 1 (opaque).
        """
        fig, axes = plt.subplots(1, 3)

        image_config = self._get_config_compare(
            plot=Plot.IMAGE,
            img=img,
            seg_truth=seg_truth,
            seg_pred=seg_pred,
            axes=axes,
            **kwargs
        )
        truth_config = self._get_config_compare(
            plot=Plot.GROUND_TRUTH,
            img=img,
            seg_truth=seg_truth,
            seg_pred=seg_pred,
            axes=axes,
            **kwargs
        )
        pred_config = self._get_config_compare(
            plot=Plot.PREDICTION,
            img=img,
            seg_truth=seg_truth,
            seg_pred=seg_pred,
            axes=axes,
            **kwargs
        )

        truth_config.slice_index.on_changed(
            partial(
                self._update_compare,
                image_config=image_config,
                truth_config=truth_config,
                pred_config=pred_config
            ))

        plt.show()
