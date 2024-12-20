from itertools import combinations

import dask.array as da
import diplib as dip
import localthickness as lt
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from napari.layers import Labels, Points
from PyQt5.QtGui import QColor
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)
from scipy.spatial import KDTree

from ..layer_selection.layer_dropdown import LayerDropdown
from ..layer_selection.layer_manager import LayerManager
from ..tables.custom_table_widget import CustomTableWidget
from .histogram_widget import HistWidget


class DistanceWidget(QScrollArea):

    def __init__(self, viewer: napari.Viewer, label_manager: LayerManager):
        super().__init__()
        self.viewer = viewer
        self.label_manager = label_manager

        distance_analysis_layout = QVBoxLayout()

        ### Add button to calculate local thickness map
        local_thickness_box = QGroupBox("Compute local thickness map")
        self.thickness_btn = QPushButton("Calculate local thickness")
        self.thickness_btn.clicked.connect(self._calculate_local_thickness)
        thickness_layout = QVBoxLayout()
        thickness_layout.addWidget(self.thickness_btn)
        local_thickness_box.setLayout(thickness_layout)
        distance_analysis_layout.addWidget(local_thickness_box)

        ### Add widget for histogram
        self.hist_widget_box = HistWidget("Histogram", viewer)
        self.hist_widget_box.setMaximumHeight(400)
        distance_analysis_layout.addWidget(self.hist_widget_box)

        ### geodesic distance from points
        geodesic_distmap_box = QGroupBox("Compute geodesic distance map")
        geodesic_distmap_box_layout = QVBoxLayout()

        geodesic_distmap_mask_layout = QHBoxLayout()
        geodesic_distmap_mask_layout.addWidget(QLabel("Mask image"))
        self.geodesic_distmap_mask_dropdown = LayerDropdown(
            self.viewer, (Labels)
        )
        self.geodesic_distmap_mask_dropdown.layer_changed.connect(
            self._update_geodesic_distmap_mask
        )
        geodesic_distmap_mask_layout.addWidget(
            self.geodesic_distmap_mask_dropdown
        )

        geodesic_distmap_marker_layer_layout = QHBoxLayout()
        geodesic_distmap_marker_layer_layout.addWidget(QLabel("Marker points"))
        self.geodesic_distmap_marker_layer_dropdown = LayerDropdown(
            self.viewer, (Labels, Points)
        )
        self.geodesic_distmap_marker_layer_dropdown.layer_changed.connect(
            self._update_geodesic_distmap_marker_layer
        )
        geodesic_distmap_marker_layer_layout.addWidget(
            self.geodesic_distmap_marker_layer_dropdown
        )

        geodesic_distmap_btn = QPushButton("Run")
        geodesic_distmap_btn.clicked.connect(self._calculate_geodesic_distance)

        geodesic_distmap_box_layout.addLayout(geodesic_distmap_mask_layout)
        geodesic_distmap_box_layout.addLayout(
            geodesic_distmap_marker_layer_layout
        )
        geodesic_distmap_box_layout.addWidget(geodesic_distmap_btn)

        geodesic_distmap_box.setLayout(geodesic_distmap_box_layout)
        distance_analysis_layout.addWidget(geodesic_distmap_box)

        ### euclidean and geodesic distance measurements
        distance_measurement_box = QGroupBox("Distance measurements")
        self.table_widget = CustomTableWidget(props=pd.DataFrame())
        distance_measurement_layout = QHBoxLayout()
        distance_measurement_layout.addWidget(self.table_widget)
        distance_measurement_box.setLayout(distance_measurement_layout)
        distance_analysis_layout.addWidget(distance_measurement_box)

        ### Map points to closest point on mask image
        points_to_mask_box = QGroupBox("Map points to closest point on mask image")
        points_to_mask_box_layout = QVBoxLayout()

        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Points"))
        self.points_dropdown = LayerDropdown(
            self.viewer, (Points)
        )
        self.points_dropdown.layer_changed.connect(
            self._update_points
        )
        points_layout.addWidget(
            self.points_dropdown
        )

        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Mask"))
        self.mask_dropdown = LayerDropdown(
            self.viewer, (Labels)
        )
        self.mask_dropdown.layer_changed.connect(
            self._update_mask
        )
        mask_layout.addWidget(
            self.mask_dropdown
        )

        map_points_to_mask_btn = QPushButton("Run")
        map_points_to_mask_btn.clicked.connect(self._map_points_to_mask)

        points_to_mask_box_layout.addLayout(points_layout)
        points_to_mask_box_layout.addLayout(mask_layout)
        points_to_mask_box_layout.addWidget(map_points_to_mask_btn)

        points_to_mask_box.setLayout(points_to_mask_box_layout)
        distance_analysis_layout.addWidget(points_to_mask_box)

        # Set main layout
        self.setLayout(distance_analysis_layout)
        self.setWidgetResizable(True)

    def _update_points(self, selected_layer: str) -> None:
        """Set the points layer for mapping to mask image"""

        if selected_layer == "":
            self.points_layer = None
        else:
            self.points_layer = self.viewer.layers[selected_layer]
            self.points_dropdown.setCurrentText(selected_layer)

    def _update_mask(self, selected_layer: str) -> None:
        """Set the mask layer for mapping points to mask image"""

        if selected_layer == "":
            self.mask_layer = None
        else:
            self.mask_layer = self.viewer.layers[selected_layer]
            self.mask_dropdown.setCurrentText(selected_layer)

    def _map_points_to_mask(self):
        """Map points to closest point on mask image"""

        mask_coords = np.array(np.nonzero(self.mask_layer.data)).T
        mask_kdtree = KDTree(mask_coords)
        _, indices = mask_kdtree.query(self.points_layer.data)
        nearest_points = mask_coords[indices]
        self.viewer.add_points(nearest_points, name="Nearest Points on Mask", face_color='green')

    def _update_geodesic_distmap_mask(self, selected_layer: str) -> None:
        """Set the mask layer for geodesic distance map calculation"""

        if selected_layer == "":
            self.geodesic_distmap_mask_layer = None
        else:
            self.geodesic_distmap_mask_layer = self.viewer.layers[
                selected_layer
            ]
            self.geodesic_distmap_mask_dropdown.setCurrentText(selected_layer)

    def _update_geodesic_distmap_marker_layer(
        self, selected_layer: str
    ) -> None:
        """Set the marker layer for geodesic distance map calculation"""

        if selected_layer == "":
            self.geodesic_distmap_marker_layer = None
        else:
            self.geodesic_distmap_marker_layer = self.viewer.layers[
                selected_layer
            ]
            self.geodesic_distmap_marker_layer_dropdown.setCurrentText(
                selected_layer
            )

    def _calculate_geodesic_distance(self):
        """Run geodesic distance map computation"""

        if isinstance(self.geodesic_distmap_mask_layer, da.core.Array):
            msg = QMessageBox()
            msg.setWindowTitle("Please convert to an in memory array")
            msg.setText("Please convert to an in memory array")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if isinstance(self.geodesic_distmap_marker_layer, Labels):

            marker = self.geodesic_distmap_marker_layer.data.copy() > 0
            self.viewer.add_image(
                np.array(
                    dip.GeodesicDistanceTransform(
                        ~marker, self.geodesic_distmap_mask_layer.data > 0
                    ),
                    dtype=np.float32,
                ),
                colormap="magma",
            )

        elif isinstance(self.geodesic_distmap_marker_layer, Points):

            if len(self.geodesic_distmap_marker_layer.data) == 1:
                mask2 = self.geodesic_distmap_mask_layer.data.copy() > 0
                mask2[
                    tuple(
                        self.geodesic_distmap_marker_layer.data[0].astype(int)
                    )
                ] = False
                self.viewer.add_image(
                    np.array(
                        dip.GeodesicDistanceTransform(
                            mask2, self.geodesic_distmap_mask_layer.data > 0
                        ),
                        dtype=np.float32,
                    ),
                    colormap="magma",
                )

            elif len(self.geodesic_distmap_marker_layer.data) > 1:

                measurements = pd.DataFrame()
                point_ids = {}
                unique_id_counter = -1
                colormap = plt.get_cmap("tab10")

                for point1, point2 in combinations(
                    self.geodesic_distmap_marker_layer.data, 2
                ):

                    # Calculate unique IDs for point1 and point2
                    if tuple(point1) not in point_ids:
                        unique_id_counter += 1
                        point_ids[tuple(point1)] = unique_id_counter
                    if tuple(point2) not in point_ids:
                        unique_id_counter += 1
                        point_ids[tuple(point2)] = unique_id_counter

                    # calculate euclidean distance between the two points
                    euclidean_dist = np.linalg.norm(point1 - point2)

                    # calculate the geodesic distance
                    mask2 = self.geodesic_distmap_mask_layer.data.copy() > 0
                    mask2[tuple(point2.astype(int))] = False
                    dist_map = np.array(
                        dip.GeodesicDistanceTransform(
                            mask2, self.geodesic_distmap_mask_layer.data > 0
                        ),
                        dtype=np.float32,
                    )
                    geodesic_dist = dist_map[tuple(point1.astype(int))]

                    # Get unique colors for point1 and point2 from tab10 colormap
                    point1_color = colormap(point_ids[tuple(point1)] % 10)
                    point2_color = colormap(point_ids[tuple(point2)] % 10)

                    # Create a dictionary to store the measurements for this pair of points
                    measurement_dict = {
                        "point1.ID": point_ids[tuple(point1)],
                        "point2.ID": point_ids[tuple(point2)],
                        "point1.x": point1[0],
                        "point1.y": point1[1],
                        "point1.z": point1[2],
                        "point2.x": point2[0],
                        "point2.y": point2[1],
                        "point2.z": point2[2],
                        "point1.color": point1_color[
                            :3
                        ],  # Use only RGB components
                        "point2.color": point2_color[
                            :3
                        ],  # Use only RGB components
                        "euclidean_dist": euclidean_dist,
                        "geodesic_dist": geodesic_dist,
                    }

                    measurements = pd.concat(
                        [measurements, pd.DataFrame([measurement_dict])]
                    )

                self.table_widget.set_content(
                    measurements.to_dict(orient="list")
                )

                # Iterate over all rows in the QTableWidget
                i = 0
                for index, row in measurements.iterrows():
                    point1_color = row["point1.color"]
                    point2_color = row["point2.color"]

                    point1_cols = [0, 2, 3, 4, 8]
                    point2_cols = [1, 5, 6, 7, 9]

                    # Set background color for point1 cells (columns 0, 1, and 2)
                    for j in point1_cols:
                        self.table_widget._view.item(i, j).setBackground(
                            QColor(
                                int(point1_color[0] * 255),
                                int(point1_color[1] * 255),
                                int(point1_color[2] * 255),
                            )
                        )

                    # Set background color for point2 cells (columns 3, 4, and 5)
                    for j in point2_cols:
                        self.table_widget._view.item(i, j).setBackground(
                            QColor(
                                int(point2_color[0] * 255),
                                int(point2_color[1] * 255),
                                int(point2_color[2] * 255),
                            )
                        )

                    i += 1

                # also set colormap to the points
                colors = [
                    colormap(i)
                    for i in range(
                        len(self.geodesic_distmap_marker_layer.data)
                    )
                ]

                self.geodesic_distmap_marker_layer.edge_color = colors
                self.geodesic_distmap_marker_layer.face_color = colors

    def _calculate_local_thickness(self) -> None:
        """Calculates local thickness of label image and adds the image to the viewer"""

        self.viewer.add_image(
            lt.local_thickness(self.label_manager.selected_layer.data), colormap="magma"
        )
