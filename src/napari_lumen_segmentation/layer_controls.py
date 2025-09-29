import napari
from napari_plane_sliders import PlaneSliderWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from napari_lumen_segmentation.layer_selection.layer_manager import LayerManager

from .copy_label_widget import CopyLabelWidget
from .save_labels_widget import SaveLabelsWidget


class LayerControlsWidget(QWidget):
    """Widget showing region props as a table and plot widget"""

    def __init__(
        self, viewer: "napari.viewer.Viewer", label_manager: LayerManager
    ) -> None:
        super().__init__()

        self.viewer = viewer
        self.layer_manager = label_manager

        layout = QVBoxLayout()

        ### create the dropdown for selecting label images
        layout.addWidget(self.layer_manager)

        ### plane sliders
        plane_slider_box = QGroupBox("Plane Sliders")
        plane_slider_layout = QVBoxLayout()
        self.plane_sliders = PlaneSliderWidget(self.viewer)
        plane_slider_layout.addWidget(self.plane_sliders)
        plane_slider_box.setLayout(plane_slider_layout)
        layout.addWidget(plane_slider_box)

        ### Add widget for copy-pasting labels from one layer to another
        self.copy_label_widget = CopyLabelWidget(self.viewer)
        layout.addWidget(self.copy_label_widget)

        ### Add widget to save labels
        save_labels = SaveLabelsWidget(self.viewer, self.layer_manager)
        layout.addWidget(save_labels)

        layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.setLayout(layout)
