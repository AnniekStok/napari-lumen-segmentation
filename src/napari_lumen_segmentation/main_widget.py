"""
Napari plugin widget for editing N-dimensional label data
"""


import napari
from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from .distance.distance_widget import DistanceWidget
from .layer_selection.layer_manager import LayerManager
from .segmentation.widget import SegmentationWidgets
from .skeleton.skeleton_widget import SkeletonWidget
from .view3D import View3D

from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from napari.layers.labels import _labels_mouse_bindings, Labels

from napari.layers.labels import Labels, _labels_mouse_bindings
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from qtpy.QtWidgets import QMessageBox

def patch_draw_behavior():
    # Skip if already patched
    if getattr(_labels_mouse_bindings, "_custom_draw_patched", False):
        return

    # Capture original draw BEFORE patching
    original_draw_fn = _labels_mouse_bindings.draw

    def custom_draw(layer, event):
        if layer.mode == "fill":
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            coord = tuple(int(c) for c in coordinates)
            old_label = layer.data[coord]

            if old_label == 0:
                msg = QMessageBox()
                msg.setWindowTitle("Fill background label?")
                msg.setText("Are you sure you want to fill the background?")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                ok_btn = msg.button(QMessageBox.Ok)
                ok_btn.setText("Yes!")
                cancel_btn = msg.button(QMessageBox.Cancel)
                cancel_btn.setText("Oops, no!")
                result = msg.exec_()

                if result != QMessageBox.Ok:
                    return

        # Call original draw if allowed
        return original_draw_fn(layer, event)

    # Monkey-patch and mark
    _labels_mouse_bindings.draw = custom_draw
    _labels_mouse_bindings._custom_draw_patched = True

    # Update mode mappings for existing/future layers
    for mode, func in Labels._drag_modes.items():
        if func is original_draw_fn:
            Labels._drag_modes[mode] = custom_draw
class LumenSegmentationWidget(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        tab_widget = QTabWidget(self)
        self.layer_manager = LayerManager(self.viewer)

        ## add 3D viewing widget
        view_3D_widget = View3D(self.viewer)
        tab_widget.addTab(view_3D_widget, "3D Viewing")

        ## add combined segmentation widgets and layer manager
        segmentation_layout = QVBoxLayout()
        segmentation_layout.addWidget(self.layer_manager)
        segmentation_widgets = SegmentationWidgets(self.viewer, self.layer_manager)
        segmentation_layout.addWidget(segmentation_widgets)
        layer_manager_segmentation = QWidget()
        layer_manager_segmentation.setLayout(segmentation_layout)
        tab_widget.addTab(layer_manager_segmentation, "Segmentation")

        ## add skeleton analysis widgets
        self.skeleton_widget = SkeletonWidget(
            viewer=self.viewer, label_manager=self.layer_manager
        )
        tab_widget.addTab(self.skeleton_widget, "Skeleton Analysis")

        self.distance_widget = DistanceWidget(
            viewer=self.viewer, label_manager=self.layer_manager
        )
        tab_widget.addTab(self.distance_widget, "Distance Analysis")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(tab_widget)
        self.setLayout(self.main_layout)

        patch_draw_behavior()