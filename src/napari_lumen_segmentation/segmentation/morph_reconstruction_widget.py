import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.morphology import reconstruction

from ..layer_selection.layer_dropdown import LayerDropdown


class MorphReconstructionWidget(QWidget):
    """Widget for implementation for morphological reconstruction by dilation with an additional intensity threshold."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        box = QGroupBox("Morphological reconstruction")
        box_layout = QVBoxLayout()

        int_layout = QHBoxLayout()
        input_image_label = QLabel("Input image")
        input_image_label.setToolTip(
            "Intensity image to apply threshold on before reconstruction"
        )
        int_layout.addWidget(input_image_label)
        self.int_input_dropdown = LayerDropdown(self.viewer, (Image))
        self.int_input_dropdown.layer_changed.connect(self._update_int_input)
        int_layout.addWidget(self.int_input_dropdown)

        seeds_layout = QHBoxLayout()
        seeds_label = QLabel("Region to expand")
        seeds_label.setToolTip("(Lumen) segmentation mask to be expanded")
        seeds_layout.addWidget(seeds_label)
        self.seeds_dropdown = LayerDropdown(self.viewer, (Labels))
        self.seeds_dropdown.layer_changed.connect(self._update_seeds)
        seeds_layout.addWidget(self.seeds_dropdown)

        mask_layout = QHBoxLayout()
        mask_label = QLabel("Mask region to expand into")
        mask_label.setToolTip(
            "Mask region to expand into. Use different label values to provide multiple local masks for reconstruction."
        )
        mask_layout.addWidget(mask_label)
        self.mask_dropdown = LayerDropdown(self.viewer, (Labels))
        self.mask_dropdown.layer_changed.connect(self._update_mask)
        mask_layout.addWidget(self.mask_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_label = QLabel("Min threshold")
        min_threshold_label.setToolTip("Min internal (lumen) value")
        min_threshold_layout.addWidget(min_threshold_label)
        self.min_threshold = QDoubleSpinBox()
        self.min_threshold.setMinimum(0)
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_label = QLabel("Max threshold")
        max_threshold_label.setToolTip("Max internal (lumen) value")
        max_threshold_layout.addWidget(max_threshold_label)
        self.max_threshold = QDoubleSpinBox()
        self.max_threshold.setMinimum(0)
        self.max_threshold.setMaximum(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        run_region_growing_btn = QPushButton("Run")
        run_region_growing_btn.clicked.connect(self._morphological_reconstruction)

        box_layout.addLayout(int_layout)
        box_layout.addLayout(seeds_layout)
        box_layout.addLayout(mask_layout)
        box_layout.addLayout(min_threshold_layout)
        box_layout.addLayout(max_threshold_layout)
        box_layout.addWidget(run_region_growing_btn)
        box.setLayout(box_layout)

        layout = QVBoxLayout()
        layout.addWidget(box)
        self.setLayout(layout)

    def _update_int_input(self, selected_layer: str) -> None:
        """Update the input Image layer"""

        if selected_layer == "":
            self.region_growing_int_layer = None
        else:
            self.region_growing_int_layer = self.viewer.layers[selected_layer]
            self.int_input_dropdown.setCurrentText(selected_layer)

    def _update_seeds(self, selected_layer: str) -> None:
        """Update the seeds label layer"""

        if selected_layer == "":
            self.seeds_layer = None
        else:
            self.seeds_layer = self.viewer.layers[selected_layer]
            self.seeds_dropdown.setCurrentText(selected_layer)

    def _update_mask(self, selected_layer: str) -> None:
        """Update the mask label layer (region to expand into)"""

        if selected_layer == "":
            self.mask = None
        else:
            self.mask = self.viewer.layers[selected_layer]
            self.mask_dropdown.setCurrentText(selected_layer)

    def _morphological_reconstruction(self) -> None:
        """Run custom region growing algorithm"""

        # do not execute if the two layers are the same
        if self.seeds_layer == self.mask:
            msg = QMessageBox()
            msg.setWindowTitle(
                "Seeds and mask layers are the same. Please select different layers."
            )
            msg.setText(
                "Seeds and mask layers are the same. Please select different layers. The seeds layer is the lumen that you are trying to correct. The mask should be a labels layer with one or more regions to expand into."
            )
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        updated_seeds = self.seeds_layer.data.copy() > 0

        region_labels = np.unique(self.mask.data)
        region_labels = [label for label in region_labels if label > 0]
        for label in region_labels:
            # define a mask of pixels that fulfill both the threshold criteria and are in the 'mask' layer
            mask = (
                (self.region_growing_int_layer.data >= self.min_threshold.value())
                & (self.region_growing_int_layer.data <= self.max_threshold.value())
                & (self.mask.data == label)
            )

            # Find the bounding box of the non-zero pixels in the mask
            non_zero_coords = np.argwhere(mask)
            if non_zero_coords.size == 0:
                print("No valid region found in the mask.")
                return

            min_coords = non_zero_coords.min(axis=0)
            max_coords = (
                non_zero_coords.max(axis=0) + 1
            )  # Add 1 to include the max boundary

            # Clip the mask and seeds to the bounding box
            clipped_mask = mask[
                min_coords[0] : max_coords[0], min_coords[1] : max_coords[1]
            ]
            seeds = (self.seeds_layer.data > 0) & (mask > 0)
            clipped_seeds = seeds[
                min_coords[0] : max_coords[0], min_coords[1] : max_coords[1]
            ]

            # Run the reconstruction function on the clipped data
            reconst = reconstruction(
                clipped_seeds.astype(int), clipped_mask.astype(int)
            )
            result = np.logical_or(clipped_seeds, reconst > 0).astype(bool)

            # Put the updated pixels in the updated_seeds array
            updated_seeds[
                min_coords[0] : max_coords[0], min_coords[1] : max_coords[1]
            ] |= result

        # Add the updated seeds layer to the viewer
        self.seeds_layer = self.viewer.add_labels(
            updated_seeds, name="morphological reconstruction"
        )
        self.seeds_dropdown.setCurrentText("morphological reconstruction")
