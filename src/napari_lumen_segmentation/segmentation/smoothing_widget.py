import copy
import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage
from skimage.io import imread

from ..layer_selection.layer_manager import LayerManager


class SmoothingWidget(QWidget):
    """Smooth and slightly grow labels by combining them with the result of a median filter"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.layer_manager = label_manager
        self.outputdir = None

        smoothbox = QGroupBox("Smooth objects")
        smooth_boxlayout = QVBoxLayout()
        smooth_layout = QHBoxLayout()

        radius_layout = QVBoxLayout()
        radius_layout.addWidget(QLabel("Median filter radius"))
        self.median_radius_field = QSpinBox()
        self.median_radius_field.setMaximum(100)
        radius_layout.addWidget(self.median_radius_field)
        smooth_layout.addLayout(radius_layout)

        iteration_layout = QVBoxLayout()
        iteration_layout.addWidget(QLabel("n iterations"))
        self.n_iterations = QSpinBox()
        self.n_iterations.setMaximum(1)
        self.n_iterations.setMaximum(100)
        iteration_layout.addWidget(self.n_iterations)
        smooth_layout.addLayout(iteration_layout)

        self.smooth_btn = QPushButton("Run")
        self.smooth_btn.clicked.connect(self._smooth_objects)
        self.smooth_btn.setEnabled(True)
        smooth_layout.addWidget(self.smooth_btn, alignment=Qt.AlignBottom)

        smooth_boxlayout.addLayout(smooth_layout)
        smoothbox.setLayout(smooth_boxlayout)
        layout = QVBoxLayout()
        layout.addWidget(smoothbox)
        self.setLayout(layout)

    def _smooth_objects(self) -> None:
        """Smooth objects by using a median filter."""

        if isinstance(self.layer_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(
                    self, "Select Output Folder"
                )

                outputdir = os.path.join(
                    self.outputdir,
                    (self.layer_manager.selected_layer.name + "_smoothed"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

            else:
                outputdir = os.path.join(
                    self.outputdir,
                    (self.layer_manager.selected_layer.name + "_smoothed"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

            for i in range(
                self.layer_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.layer_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                # Apply smoothing using median filter
                smoothed = ndimage.median_filter(
                    current_stack,
                    size=self.median_radius_field.value(),
                )

                # combine smoothed result with original result to selectively grow the mask
                mask = (smoothed != 0) & (current_stack == 0)
                result = current_stack.copy()
                result[mask] = smoothed[mask]

                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.layer_manager.selected_layer.name
                            + "_smoothed_slice"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(result, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.layer_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.layer_manager.selected_layer.name + "_smoothed",
            )

        else:
            if len(self.layer_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(self.layer_manager.selected_layer.data.shape[0]):
                    # Apply smoothing using median filter
                    smoothed = ndimage.median_filter(
                        self.layer_manager.selected_layer.data[i],
                        size=self.median_radius_field.value(),
                    )

                    # combine smoothed result with original result to selectively grow the mask
                    mask = (smoothed != 0) & (
                        self.layer_manager.selected_layer.data[i] == 0
                    )
                    result = copy.deepcopy(self.layer_manager.selected_layer.data[i])
                    result[mask] = smoothed[mask]

                    stack.append(result)
                self.layer_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.layer_manager.selected_layer.name + "_smoothed",
                )
                self.layer_manager._update_labels(
                    self.layer_manager.selected_layer.name
                )

            elif len(self.layer_manager.selected_layer.data.shape) == 3:
                input_data = copy.deepcopy(self.layer_manager.selected_layer.data)
                for _ in range(self.n_iterations.value()):
                    # Apply smoothing using median filter
                    smoothed = ndimage.median_filter(
                        input_data,
                        size=self.median_radius_field.value(),
                    )

                    # combine smoothed result with original result to selectively grow the mask
                    mask = (input_data == 0) & (smoothed != 0)
                    input_data[mask] = smoothed[mask]

                self.layer_manager.selected_layer = self.viewer.add_labels(
                    input_data,
                    name=self.layer_manager.selected_layer.name + "_smoothed",
                )
                self.layer_manager._update_labels(
                    self.layer_manager.selected_layer.name
                )

            else:
                print("input should be a 3D or 4D array")
