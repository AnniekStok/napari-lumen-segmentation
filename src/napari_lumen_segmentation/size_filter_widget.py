import functools
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
from skimage import measure
from skimage.io import imread

from .layer_manager import LayerManager


class SizeFilterWidget(QWidget):
    """Widget to delete objects that do not fulfill a size criterion"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.label_manager = label_manager
        self.outputdir = None

        filterbox = QGroupBox("Filter objects by size")
        
        min_size_layout = QVBoxLayout()
        min_size_layout.addWidget(QLabel('min size (voxels)'))
        self.min_size_field = QSpinBox()
        self.min_size_field.setMaximum(1000000)
        min_size_layout.addWidget(self.min_size_field)

        max_size_layout = QVBoxLayout()
        max_size_layout.addWidget(QLabel('max size (voxels)'))
        self.max_size_field = QSpinBox()
        self.max_size_field.setMaximum(1000000)
        max_size_layout.addWidget(self.max_size_field)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_objects)
        self.delete_btn.setEnabled(True)

        threshold_size_layout = QHBoxLayout()
        threshold_size_layout.addLayout(min_size_layout)
        threshold_size_layout.addLayout(max_size_layout)
        threshold_size_layout.addWidget(self.delete_btn, alignment=Qt.AlignBottom)

        filterbox.setLayout(threshold_size_layout)

        layout = QVBoxLayout()
        layout.addWidget(filterbox)

        self.setLayout(layout)

    def _delete_objects(self) -> None:
        """Delete objects in the selected layer that are too small or too big"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            if self.outputdir is None:
                self.outputdir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

            outputdir = os.path.join(
                self.outputdir,
                (self.label_manager.selected_layer.name + "_sizefiltered"),
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

                # measure the sizes in pixels of the labels in slice using skimage.regionprops
                props = measure.regionprops(current_stack)
                filtered_labels = [
                    p.label
                    for p in props
                    if (p.area >= self.min_size_field.value() and p.area <= self.max_size_field.value())
                ]
                mask = functools.reduce(
                    np.logical_or,
                    (current_stack == val for val in filtered_labels),
                )
                filtered = np.where(mask, current_stack, 0)
                tifffile.imwrite(
                    os.path.join(
                        outputdir,
                        (
                            self.label_manager.selected_layer.name
                            + "_sizefiltered_TP"
                            + str(i).zfill(4)
                            + ".tif"
                        ),
                    ),
                    np.array(filtered, dtype="uint16"),
                )

            file_list = [
                os.path.join(outputdir, fname)
                for fname in os.listdir(outputdir)
                if fname.endswith(".tif")
            ]
            self.label_manager.selected_layer = self.viewer.add_labels(
                da.stack([imread(fname) for fname in sorted(file_list)]),
                name=self.label_manager.selected_layer.name
                + "_sizefiltered",
            )
            self.label_manager._update_labels(
                self.label_manager.selected_layer.name
            )

        else:
            # Image data is a normal array and can be directly edited.
            if len(self.label_manager.selected_layer.data.shape) == 4:
                stack = []
                for i in range(
                    self.label_manager.selected_layer.data.shape[0]
                ):
                    props = measure.regionprops(
                        self.label_manager.selected_layer.data[i]
                    )
                    filtered_labels = [
                        p.label
                        for p in props
                        if (p.area >= self.min_size_field.value() and p.area <= self.max_size_field.value())
                    ]
                    mask = functools.reduce(
                        np.logical_or,
                        (
                            self.label_manager.selected_layer.data[i] == val
                            for val in filtered_labels
                        ),
                    )
                    filtered = np.where(
                        mask, self.label_manager.selected_layer.data[i], 0
                    )
                    stack.append(filtered)
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.label_manager.selected_layer.name
                    + "_sizefiltered",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            elif len(self.label_manager.selected_layer.data.shape) == 3:
                props = measure.regionprops(
                    self.label_manager.selected_layer.data
                )
                filtered_labels = [
                    p.label
                    for p in props
                    if (p.area >= self.min_size_field.value() and p.area <= self.max_size_field.value())
                ]
                mask = functools.reduce(
                    np.logical_or,
                    (
                        self.label_manager.selected_layer.data == val
                        for val in filtered_labels
                    ),
                )
                self.label_manager.selected_layer = self.viewer.add_labels(
                    np.where(mask, self.label_manager.selected_layer.data, 0),
                    name=self.label_manager.selected_layer.name
                    + "_sizefiltered",
                )
                self.label_manager._update_labels(
                    self.label_manager.selected_layer.name
                )

            else:
                print("input should be 3D or 4D array")
