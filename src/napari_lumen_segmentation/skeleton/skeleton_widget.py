import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import skan
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from skimage import morphology

from ..layer_selection.layer_manager import LayerManager
from ..plots.plot_widget import PlotWidget
from ..tables.custom_table_widget import CustomTableWidget


class SkeletonWidget(QScrollArea):

    def __init__(self, viewer: napari.Viewer, label_manager: LayerManager):
        super().__init__()
        self.viewer = viewer
        self.label_manager = label_manager

        ### Skeleton analysis widget
        self.analysis_layout = QVBoxLayout()

        ## Add box for skeleton analysis
        skeleton_box = QGroupBox("Skeleton Analysis")
        self.skeleton_box_layout = QVBoxLayout()

        self.skeleton_btn = QPushButton("Create skeleton")
        self.skeleton_btn.clicked.connect(self._skeletonize)
        self.skeleton_box_layout.addWidget(self.skeleton_btn)

        self.skeleton_visualization_dropdown = QComboBox()
        self.skeleton_visualization_dropdown.addItem("Skeleton")
        self.skeleton_visualization_dropdown.addItem("Path")
        self.skeleton_visualization_dropdown.addItem("Branch Length")
        self.skeleton_visualization_dropdown.currentIndexChanged.connect(
            self._update_skeleton_visualization
        )

        self.skeleton_box_layout.addWidget(
            self.skeleton_visualization_dropdown
        )
        skeleton_box.setLayout(self.skeleton_box_layout)

        self.analysis_layout.addWidget(skeleton_box)

        self.table_widget = CustomTableWidget(props=pd.DataFrame())
        self.plot_widget = PlotWidget(props=pd.DataFrame())

        self.analysis_layout.addWidget(self.table_widget)
        self.analysis_layout.addWidget(self.plot_widget)

        self.analysis_widgets = QWidget()
        self.analysis_widgets.setLayout(self.analysis_layout)

        self.setWidget(self.analysis_widgets)
        self.setWidgetResizable(True)

    def _skeletonize(self) -> None:
        """Create skeleton from label image"""

        skel = morphology.skeletonize(self.label_manager.selected_layer.data)
        degree_image = skan.csr.make_degree_image(skel)

        self.viewer.add_labels(degree_image, name="Connectivity")

        skeleton = skan.Skeleton(skel)
        all_paths = [
            skeleton.path_coordinates(i) for i in range(skeleton.n_paths)
        ]

        self.paths_table = skan.summarize(skeleton)
        self.paths_table["path-id"] = np.arange(skeleton.n_paths)

        # Create a randomized colormap
        colormaps = ["tab20", "tab20b", "tab20c"]
        color_cycle = []
        for cmap_name in colormaps:
            cmap = plt.get_cmap(cmap_name)
            colors = [mcolors.to_hex(cmap(i)) for i in range(20)]
            color_cycle.extend(colors)
        np.random.shuffle(color_cycle)
        n_colors = len(np.unique(self.paths_table["path-id"]))
        repetitions = (n_colors + len(color_cycle) - 1) // len(color_cycle)
        extended_color_cycle = color_cycle * repetitions
        self.random_cmap = mcolors.ListedColormap(
            extended_color_cycle[:n_colors]
        )

        # define shapes layer
        self.skeleton = self.viewer.add_shapes(
            all_paths,
            shape_type="path",
            properties=self.paths_table,
            edge_width=0.5,
            edge_color="skeleton-id",
            face_color="skeleton-id",
            edge_colormap="viridis",
            face_colormap="viridis",
            edge_color_cycle=self.random_cmap.colors,
            face_color_cycle=self.random_cmap.colors,
        )

        # update table widget
        self.table_widget.set_content(self.paths_table.to_dict(orient="list"))
        self.table_widget._recolor(by="skeleton-id", cmap=self.random_cmap)

        # update plot widget
        self.plot_widget.props = self.paths_table
        self.plot_widget.categorical_cmap = self.random_cmap
        self.plot_widget.continuous_cmap = "viridis"
        self.plot_widget._update_dropdowns()

    def _update_skeleton_visualization(self) -> None:
        """Update the coloring of the skeleton layer"""

        if self.skeleton_visualization_dropdown.currentText() == "Path":

            ids = self.skeleton.properties["path-id"]
            colors = [self.random_cmap.colors[c] for c in ids]
            self.skeleton.edge_color = colors
            self.skeleton.face_color = colors
            self.table_widget._recolor(by="path-id", cmap=self.random_cmap)
        if self.skeleton_visualization_dropdown.currentText() == "Skeleton":
            ids = self.skeleton.properties["skeleton-id"]
            colors = [self.random_cmap.colors[c] for c in ids]
            self.skeleton.edge_color = colors
            self.skeleton.face_color = colors
            self.table_widget._recolor(by="skeleton-id", cmap=self.random_cmap)
        if (
            self.skeleton_visualization_dropdown.currentText()
            == "Branch Length"
        ):
            self.skeleton.edge_color = "branch-distance"
            self.skeleton.face_color = "branch-distance"
            self.table_widget._recolor(by=None, cmap=self.random_cmap)

        self.skeleton.refresh()
