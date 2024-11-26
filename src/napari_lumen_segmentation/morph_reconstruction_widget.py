import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.morphology import reconstruction

from .layer_dropdown import LayerDropdown


class MorphReconstructionWidget(QWidget):
    """Widget for implementation for morphological reconstruction by dilation with an additional intensity threshold."""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        box = QGroupBox("Morphological reconstruction")
        box_layout = QVBoxLayout()

        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Input image"))
        self.int_input_dropdown = LayerDropdown(self.viewer, (Image))
        self.int_input_dropdown.layer_changed.connect(self._update_int_input)
        int_layout.addWidget(self.int_input_dropdown)

        seeds_layout = QHBoxLayout()
        seeds_layout.addWidget(QLabel("Seed labels"))
        self.seeds_dropdown = LayerDropdown(self.viewer, (Labels))
        self.seeds_dropdown.layer_changed.connect(self._update_seeds)
        seeds_layout.addWidget(self.seeds_dropdown)

        exclude_layout = QHBoxLayout()
        exclude_layout.addWidget(QLabel("Mask"))
        self.exclude_dropdown = LayerDropdown(self.viewer, (Labels))
        self.exclude_dropdown.layer_changed.connect(self._update_exclude)
        exclude_layout.addWidget(self.exclude_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel("Threshold (min internal value)"))
        self.min_threshold = QDoubleSpinBox()
        self.min_threshold.setMinimum(0)
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel("Threshold (max internal value)"))
        self.max_threshold = QDoubleSpinBox()
        self.max_threshold.setMinimum(0)
        self.max_threshold.setMaximum(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        run_region_growing_btn = QPushButton("Run")
        run_region_growing_btn.clicked.connect(
            self._morphological_reconstruction
        )

        box_layout.addLayout(int_layout)
        box_layout.addLayout(seeds_layout)
        box_layout.addLayout(exclude_layout)
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

    def _update_exclude(self, selected_layer: str) -> None:
        """Update the exclude label layer (forbidden regions)"""

        if selected_layer == "":
            self.mask = None
        else:
            self.mask = self.viewer.layers[selected_layer]
            self.exclude_dropdown.setCurrentText(selected_layer)

    def _morphological_reconstruction(self) -> None:
        """Run custom region growing algorithm"""

        # define a mask of pixels that fulfill both the threshold criteria and are in the 'mask' layer
        mask = (self.region_growing_int_layer.data >= self.min_threshold.value()) & (self.region_growing_int_layer.data <= self.max_threshold.value()) & (self.mask.data > 0)
        seeds = (self.seeds_layer.data > 0) & (mask > 0)

        reconst = reconstruction(seeds * 1, mask * 2)
        result = np.logical_or(self.seeds_layer.data > 0, reconst > 0).astype(int)

        self.seeds_layer = self.viewer.add_labels(result, name = "morphological reconstruction")
        self.seeds_dropdown.setCurrentText("morphological reconstruction")
