"""
Napari plugin widget for editing N-dimensional label data
"""


import napari
from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .distance.distance_widget import DistanceWidget
from .layer_selection.layer_manager import LayerManager
from .segmentation.widget import SegmentationWidgets
from .skeleton.skeleton_widget import SkeletonWidget
from .view3D import View3D


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
