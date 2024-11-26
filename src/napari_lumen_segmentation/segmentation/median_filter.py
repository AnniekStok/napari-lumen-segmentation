import os
import shutil

import dask.array as da
import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from scipy import ndimage
from skimage.io import imread

from ..layer_selection.layer_manager import LayerManager


class MedianFilter(QWidget):
    """Apply median filter for smoothing labels or masks"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        smoothbox = QGroupBox("Median filter")
        smooth_boxlayout = QVBoxLayout()

        smooth_layout = QHBoxLayout()
        self.median_radius_field = QSpinBox()
        self.median_radius_field.setMaximum(100)
        self.smooth_btn = QPushButton("Apply")
        smooth_layout.addWidget(self.median_radius_field)
        smooth_layout.addWidget(self.smooth_btn)

        smooth_boxlayout.addWidget(QLabel("Median filter radius"))
        smooth_boxlayout.addLayout(smooth_layout)

        self.smooth_btn.clicked.connect(self._smooth_objects)
        self.smooth_btn.setEnabled(True)

        smoothbox.setLayout(smooth_boxlayout)
        layout = QVBoxLayout()
        layout.addWidget(smoothbox)
        self.setLayout(layout)

    def _smooth_objects(self) -> None:
        """Smooth objects by using a median filter."""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

                outputdir = os.path.join(
                    self.outputdir,
                    (self.label_manager.selected_layer.name + "_median_filter"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

            else:
                outputdir = os.path.join(
                    self.outputdir,
                    (self.label_manager.selected_layer.name + "_median_filter"),
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

            for i in range(
                self.label_manager.selected_layer.data.shape[0]
            ):  # Loop over the first dimension
                current_stack = self.label_manager.selected_layer.data[
                    i
                ].compute()  # Compute the current stack
                smoothed = ndimage.median_filter(
                    current_stack, size=self.median_radius_field.value()
                )
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_median_filter_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(smoothed, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name + "_median_filter",
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    smoothed = ndimage.median_filter(
                        self.label_manager.selected_layer.data[i],
                        size=self.median_radius_field.value(),
                    )
                    stack.append(smoothed)
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.label_manager.selected_layer.name + "_median_filter",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            elif len(self.label_manager.selected_layer.data.shape) == 3:
                self.label_manager.selected_layer = self.viewer.add_labels(
                    ndimage.median_filter(
                        self.label_manager.selected_layer.data,
                        size=self.median_radius_field.value(),
                    ),
                    name=self.label_manager.selected_layer.name + "_median_filter",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            else:
                print("input should be a 3D or 4D array")
