"""
Napari plugin widget for editing N-dimensional label data
"""

import functools
import os
import shutil

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap, to_rgb
from napari.layers import Image, Labels
from napari_plane_sliders._plane_slider_widget import PlaneSliderWidget
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage import measure
from skimage.io import imread
from skimage.segmentation import (
    expand_labels,
)
from napari.utils.notifications import show_info
import morphsnakes as ms

from ._custom_table_widget import ColoredTableWidget, TableWidget
from ._distance_widget import DistanceWidget
from ._layer_dropdown import LayerDropdown
from ._plot_widget import PlotWidget
from ._skeleton_widget import SkeletonWidget


class AnnotateLabelsND(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

        self.labels = None
        self.skeleton = None
        self.label_table = None
        self.skeleton_table = None
        self.distance_table = None
        self.csv_table = None

        self.points = None
        self.outputdir = None
        self.segmentation_layout = QVBoxLayout()
        self.tab_widget = QTabWidget(self)
        self.option_labels = None
        self.csv_path = None
        self.label_properties = None
        self.plot_widget = PlotWidget(props=pd.DataFrame())
        self.table_widget = TableWidget(props=pd.DataFrame())
        self.label_table_widget = ColoredTableWidget(
            napari.layers.Labels(np.zeros((10, 10), dtype=np.uint8)),
            self.viewer,
        )
        self.label_plot_widget = PlotWidget(props=pd.DataFrame())

        ### specify output directory
        outputbox_layout = QHBoxLayout()
        self.outputdirbtn = QPushButton("Select output directory")
        self.output_path = QLineEdit()
        outputbox_layout.addWidget(self.outputdirbtn)
        outputbox_layout.addWidget(self.output_path)
        self.outputdirbtn.clicked.connect(self._on_get_output_dir)
        self.segmentation_layout.addLayout(outputbox_layout)

        ### create the dropdown for selecting label images
        self.label_dropdown = LayerDropdown(self.viewer, (Labels))
        self.label_dropdown.layer_changed.connect(self._update_labels)
        self.segmentation_layout.addWidget(self.label_dropdown)

        ### Add option to convert dask array to in-memory array
        self.convert_to_array_btn = QPushButton("Convert to in-memory array")
        self.convert_to_array_btn.setEnabled(
            self.labels is not None
            and isinstance(self.labels.data, da.core.Array)
        )
        self.convert_to_array_btn.clicked.connect(self._convert_to_array)
        self.segmentation_layout.addWidget(self.convert_to_array_btn)

        ### Add widget for adding overview table
        self.table_btn = QPushButton("Show table")
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.clicked.connect(
            lambda: self.tab_widget.setCurrentIndex(2)
        )
        if self.labels is not None:
            self.table_btn.setEnabled(True)
        self.segmentation_layout.addWidget(self.table_btn)

        ## Add save labels widget
        self.save_btn = QPushButton("Save labels")
        self.save_btn.clicked.connect(self._save_labels)
        self.segmentation_layout.addWidget(self.save_btn)

        ## Add button to clear all layers
        self.clear_btn = QPushButton("Clear all layers")
        self.clear_btn.clicked.connect(self._clear_layers)
        self.segmentation_layout.addWidget(self.clear_btn)

        ## Add button to run connected component analysis
        self.convert_to_labels_btn = QPushButton(
            "Run connected components labeling"
        )
        self.convert_to_labels_btn.clicked.connect(self._convert_to_labels)
        self.segmentation_layout.addWidget(self.convert_to_labels_btn)

        ### Add widget for size filtering
        filterbox = QGroupBox("Filter objects by size")
        filter_layout = QVBoxLayout()

        label_size = QLabel("Min size threshold (voxels)")
        threshold_size_layout = QHBoxLayout()
        self.min_size_field = QSpinBox()
        self.min_size_field.setMaximum(1000000)
        self.delete_btn = QPushButton("Delete")
        threshold_size_layout.addWidget(self.min_size_field)
        threshold_size_layout.addWidget(self.delete_btn)

        filter_layout.addWidget(label_size)
        filter_layout.addLayout(threshold_size_layout)
        self.delete_btn.clicked.connect(self._delete_small_objects)
        self.delete_btn.setEnabled(True)

        filterbox.setLayout(filter_layout)
        self.segmentation_layout.addWidget(filterbox)

        self.setLayout(self.segmentation_layout)

        ### Add widget for eroding/dilating labels
        dil_erode_box = QGroupBox("Erode/dilate labels")
        dil_erode_box_layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        str_element_diameter_label = QLabel("Structuring element diameter")
        str_element_diameter_label.setFixedWidth(200)
        self.structuring_element_diameter = QSpinBox()
        self.structuring_element_diameter.setMaximum(100)
        self.structuring_element_diameter.setValue(1)
        radius_layout.addWidget(str_element_diameter_label)
        radius_layout.addWidget(self.structuring_element_diameter)

        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations")
        iterations_label.setFixedWidth(200)
        self.iterations = QSpinBox()
        self.iterations.setMaximum(100)
        self.iterations.setValue(1)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations)

        shrink_dilate_buttons_layout = QHBoxLayout()
        self.erode_btn = QPushButton("Erode")
        self.dilate_btn = QPushButton("Dilate")
        self.erode_btn.clicked.connect(self._erode_labels)
        self.dilate_btn.clicked.connect(self._dilate_labels)
        shrink_dilate_buttons_layout.addWidget(self.erode_btn)
        shrink_dilate_buttons_layout.addWidget(self.dilate_btn)

        if self.labels is not None:
            self.erode_btn.setEnabled(True)
            self.dilate_btn.setEnabled(True)

        dil_erode_box_layout.addLayout(radius_layout)
        dil_erode_box_layout.addLayout(iterations_layout)
        dil_erode_box_layout.addLayout(shrink_dilate_buttons_layout)

        dil_erode_box.setLayout(dil_erode_box_layout)
        self.segmentation_layout.addWidget(dil_erode_box)

        ### Threshold image
        threshold_box = QGroupBox("Threshold")
        threshold_box_layout = QVBoxLayout()

        self.threshold_layer_dropdown = LayerDropdown(
            self.viewer, (Image, Labels)
        )
        self.threshold_layer_dropdown.layer_changed.connect(
            self._update_threshold_layer
        )
        threshold_box_layout.addWidget(self.threshold_layer_dropdown)

        min_threshold_layout = QHBoxLayout()
        min_threshold_layout.addWidget(QLabel("Min value"))
        self.min_threshold = QSpinBox()
        self.min_threshold.setMaximum(65535)
        min_threshold_layout.addWidget(self.min_threshold)

        max_threshold_layout = QHBoxLayout()
        max_threshold_layout.addWidget(QLabel("Max value"))
        self.max_threshold = QSpinBox()
        self.max_threshold.setMaximum(65535)
        self.max_threshold.setValue(65535)
        max_threshold_layout.addWidget(self.max_threshold)

        threshold_box_layout.addLayout(min_threshold_layout)
        threshold_box_layout.addLayout(max_threshold_layout)
        threshold_btn = QPushButton("Run")
        threshold_btn.clicked.connect(self._threshold)
        threshold_box_layout.addWidget(threshold_btn)

        threshold_box.setLayout(threshold_box_layout)
        self.segmentation_layout.addWidget(threshold_box)

        ### Add one image to another
        image_calc_box = QGroupBox("Image calculator")
        image_calc_box_layout = QVBoxLayout()

        image1_layout = QHBoxLayout()
        image1_layout.addWidget(QLabel("Label image 1"))
        self.image1_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image1_dropdown.layer_changed.connect(self._update_image1)
        image1_layout.addWidget(self.image1_dropdown)

        image2_layout = QHBoxLayout()
        image2_layout.addWidget(QLabel("Label image 2"))
        self.image2_dropdown = LayerDropdown(self.viewer, (Image, Labels))
        self.image2_dropdown.layer_changed.connect(self._update_image2)
        image2_layout.addWidget(self.image2_dropdown)

        image_calc_box_layout.addLayout(image1_layout)
        image_calc_box_layout.addLayout(image2_layout)

        operation_layout = QHBoxLayout()
        self.operation = QComboBox()
        self.operation.addItem("Add")
        self.operation.addItem("Subtract")
        self.operation.addItem("Multiply")
        self.operation.addItem("Divide")
        self.operation.addItem("AND")
        self.operation.addItem("OR")
        operation_layout.addWidget(QLabel("Operation"))
        operation_layout.addWidget(self.operation)
        image_calc_box_layout.addLayout(operation_layout)

        add_images_btn = QPushButton("Run")
        add_images_btn.clicked.connect(self._calculate_images)
        image_calc_box_layout.addWidget(add_images_btn)

        image_calc_box.setLayout(image_calc_box_layout)
        self.segmentation_layout.addWidget(image_calc_box)

        ### Morphological snakes
        active_contour_box = QGroupBox("Region growing (Morphological Snakes)")
        active_contour_box_layout = QVBoxLayout()

        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Input image"))
        self.int_input_dropdown = LayerDropdown(self.viewer, (Image))
        self.int_input_dropdown.layer_changed.connect(self._update_int_input)
        int_layout.addWidget(self.int_input_dropdown)

        seeds_layout = QHBoxLayout()
        seeds_layout.addWidget(QLabel("Label seeds"))
        self.seeds_dropdown = LayerDropdown(self.viewer, (Labels))
        self.seeds_dropdown.layer_changed.connect(self._update_seeds)
        seeds_layout.addWidget(self.seeds_dropdown)

        num_iter_layout = QHBoxLayout()
        num_iter_layout.addWidget(QLabel("Number of iterations"))
        self.num_iter_spin = QSpinBox()
        self.num_iter_spin.setMinimum(1)
        self.num_iter_spin.setMaximum(5000)
        num_iter_layout.addWidget(self.num_iter_spin)

        lambda1_layout = QHBoxLayout()
        lambda1_layout.addWidget(QLabel("Lambda1"))
        self.lambda1_spin = QSpinBox()
        self.lambda1_spin.setMinimum(1)
        self.lambda1_spin.setMaximum(5000)
        lambda1_layout.addWidget(self.lambda1_spin)

        lambda2_layout = QHBoxLayout()
        lambda2_layout.addWidget(QLabel("Lambda2"))
        self.lambda2_spin = QSpinBox()
        self.lambda2_spin.setMinimum(1)
        self.lambda2_spin.setMaximum(5000)
        lambda2_layout.addWidget(self.lambda2_spin)

        calc_active_contour_btn = QPushButton("Run")
        calc_active_contour_btn.clicked.connect(
            self._morphological_active_contour
        )

        active_contour_box_layout.addLayout(int_layout)
        active_contour_box_layout.addLayout(seeds_layout)
        active_contour_box_layout.addLayout(num_iter_layout)
        active_contour_box_layout.addLayout(lambda1_layout)
        active_contour_box_layout.addLayout(lambda2_layout)
        active_contour_box_layout.addWidget(calc_active_contour_btn)

        active_contour_box.setLayout(active_contour_box_layout)
        self.segmentation_layout.addWidget(active_contour_box)

        ### combine into tab widget

        ## add plane viewing widget
        plane_widget = PlaneSliderWidget(self.viewer)
        self.tab_widget.addTab(plane_widget, "Plane Viewing")

        ## add combined segmentation widgets
        self.segmentation_widgets = QWidget()
        self.segmentation_widgets.setLayout(self.segmentation_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.segmentation_widgets)
        scroll_area.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_area, "Segmentation")

        ## add tab with label plots
        self.label_plotting_widgets = QWidget()
        self.label_plotting_widgets_layout = QVBoxLayout()
        self.label_plotting_widgets_layout.addWidget(self.label_table_widget)
        self.label_plotting_widgets_layout.addWidget(self.label_plot_widget)
        self.label_plotting_widgets.setLayout(
            self.label_plotting_widgets_layout
        )
        self.tab_widget.addTab(self.label_plotting_widgets, "Label Plots")

        ## add skeleton analysis widgets

        self.skeleton_widget = SkeletonWidget(
            viewer=self.viewer, labels=self.labels
        )
        self.tab_widget.addTab(self.skeleton_widget, "Skeleton Analysis")

        self.distance_widget = DistanceWidget(
            viewer=self.viewer, labels=self.labels
        )
        self.tab_widget.addTab(self.distance_widget, "Distance Analysis")

        # Add the tab widget to the main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)

    def _switch_table_content(self) -> None:
        """Set the content of the table widget depending on the choice in the table dropdown"""

        if self.table_dropdown.currentText() == "CSV":
            # switch to skeleton table
            self.table_widget.set_content(
                self.csv_table.to_dict(orient="list")
            )
            self.plot_widget.props = self.csv_table
            self.plot_widget._update_dropdowns()
        if self.table_dropdown.currentText() == "Skeleton":
            # switch to skeleton table
            self.table_widget.set_content(
                self.skeleton_table.to_dict(orient="list")
            )
            self.plot_widget.props = self.skeleton_table
            self.plot_widget._update_dropdowns()
        if self.table_dropdown.currentText() == "Distances":
            # switch to distance measurements table
            self.table_widget.set_content(
                self.distance_table.to_dict(orient="list")
            )
            self.plot_widget.props = self.distance_table
            self.plot_widget._update_dropdowns()

    def _update_table_dropdown(self) -> None:
        """Update options in the table dropdown for plotting"""

        for label, table_option in zip(
            ["CSV", "Skeleton", "Distances"],
            [self.csv_table, self.skeleton_table, self.distance_table],
        ):
            if table_option is not None:

                label_exists = False
                for index in range(self.table_dropdown.count()):
                    if self.table_dropdown.itemText(index) == label:
                        label_exists = True
                        break
                if not label_exists:
                    self.table_dropdown.addItem(label)

    def _choose_csv_path(self) -> None:
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .csv file",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if path:
            self.csv_path_edit.setText(path)
            self.csv_path = str(path)

    def _update_label_props_path(self) -> None:
        self.csv_path = str(self.csv_path_edit.text())

    def _set_csv_table(self) -> None:
        if self.csv_path is not None and os.path.exists(self.csv_path):
            self.csv_table = pd.read_csv(self.csv_path)
            self.table_widget.set_content(
                self.csv_table.to_dict(orient="list")
            )
            self.plot_widget.props = self.csv_table
            self.plot_widget.label_colormap = None
            self.plot_widget._update_dropdowns()
            self._update_table_dropdown()
            self.table_dropdown.setCurrentText("CSV")
        else:
            print("no csv file selected")

    def _on_get_output_dir(self) -> None:
        """Show a dialog window to let the user pick the output directory."""

        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)
            self.outputdir = str(self.output_path.text())

    def _convert_to_labels(self) -> None:
        """Convert to labels image"""

        if self.labels is not None:
            self.labels = self.viewer.add_labels(
                measure.label(self.labels.data)
            )
            self._update_labels(self.labels.name)

    def _update_labels(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'labels' layer that is being edited."""

        if selected_layer == "":
            self.labels = None
        else:
            self.labels = self.viewer.layers[selected_layer]
            self.label_dropdown.setCurrentText(selected_layer)
            self.convert_to_array_btn.setEnabled(
                isinstance(self.labels.data, da.core.Array)
            )
            self.skeleton_widget.labels = self.labels
            self.distance_widget.labels = self.labels

    def _update_image1(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image1_layer = None
        else:
            self.image1_layer = self.viewer.layers[selected_layer]
            self.image1_dropdown.setCurrentText(selected_layer)

    def _update_image2(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.image2_layer = None
        else:
            self.image2_layer = self.viewer.layers[selected_layer]
            self.image2_dropdown.setCurrentText(selected_layer)

    def _update_threshold_layer(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.threshold_layer = None
        else:
            self.threshold_layer = self.viewer.layers[selected_layer]
            self.threshold_layer_dropdown.setCurrentText(selected_layer)

    def _update_int_input(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.int_input_layer = None
        else:
            self.int_input_layer = self.viewer.layers[selected_layer]
            self.int_input_dropdown.setCurrentText(selected_layer)

    def _update_seeds(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'source labels' layer for copying labels from."""

        if selected_layer == "":
            self.seeds_layer = None
        else:
            self.seeds_layer = self.viewer.layers[selected_layer]
            self.seeds_dropdown.setCurrentText(selected_layer)

    def _convert_to_array(self) -> None:
        """Convert from dask array to in-memory array. This is necessary for manual editing using the label tools (brush, eraser, fill bucket)."""

        if isinstance(self.labels.data, da.core.Array):
            stack = []
            for i in range(self.labels.data.shape[0]):
                current_stack = self.labels.data[i].compute()
                stack.append(current_stack)
            self.labels.data = np.stack(stack, axis=0)

    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if isinstance(self.labels.data, da.core.Array):
            tp = self.viewer.dims.current_step[0]
            current_stack = self.labels.data[
                tp
            ].compute()  # Compute the current stack
            self.label_table = measure.regionprops_table(
                current_stack, properties=["label", "area", "centroid"]
            )
            if hasattr(self.labels, "properties"):
                self.labels.properties = self.label_table
            if hasattr(self.labels, "features"):
                self.labels.features = self.label_table

        else:
            if len(self.labels.data.shape) == 4:
                tp = self.viewer.dims.current_step[0]
                self.label_table = measure.regionprops_table(
                    self.labels.data[tp],
                    properties=["label", "area", "centroid"],
                )
                if hasattr(self.labels, "properties"):
                    self.labels.properties = self.label_table
                if hasattr(self.labels, "features"):
                    self.labels.features = self.label_table

            elif len(self.labels.data.shape) == 3:
                self.label_table = measure.regionprops_table(
                    self.labels.data, properties=["label", "area", "centroid"]
                )
                if hasattr(self.labels, "properties"):
                    self.labels.properties = self.label_table
                if hasattr(self.labels, "features"):
                    self.labels.features = self.label_table
            else:
                print("input should be a 3D or 4D array")
                self.label_table = None

        if self.label_table_widget is not None:
            self.label_table_widget.hide()

        if self.viewer is not None:
            self.label_table_widget = ColoredTableWidget(
                self.labels, self.viewer
            )
            self.label_table_widget._set_label_colors_to_rows()
            self.label_table_widget.setMinimumWidth(500)
            self.label_plotting_widgets_layout.addWidget(
                self.label_table_widget
            )

            # update the plot widget and set label colors
            self.label_plot_widget.props = pd.DataFrame.from_dict(
                self.label_table
            )
            unique_labels = self.label_plot_widget.props["label"].unique()
            label_colors = [
                to_rgb(self.labels.get_color(label)) for label in unique_labels
            ]
            self.label_plot_widget.label_colormap = ListedColormap(
                label_colors
            )
            self.label_plot_widget._update_dropdowns()

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if isinstance(self.labels.data, da.core.Array):

            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(
                    self.outputdir, (self.labels.name + "_finalresult")
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.labels.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.labels.data[
                        i
                    ].compute()  # Compute the current stack
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.labels.name
                                + "_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(current_stack, dtype="uint16"),
                    )
                return True

        elif len(self.labels.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )
            for i in range(self.labels.data.shape[0]):
                labels_data = self.labels.data[i].astype(np.uint16)
                tifffile.imwrite(
                    (
                        filename.split(".tif")[0]
                        + "_TP"
                        + str(i).zfill(4)
                        + ".tif"
                    ),
                    labels_data,
                )

        elif len(self.labels.data.shape) == 3:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )

            if filename:
                labels_data = self.labels.data.astype(np.uint16)
                tifffile.imwrite(filename, labels_data)

        else:
            print("labels should be a 3D or 4D array")

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        self.cross.setChecked(False)
        self.cross.layer = None
        self.viewer.layers.clear()

    def _keep_objects(self) -> None:
        """Keep only the labels that are selected by the points layer."""

        if isinstance(self.labels.data, da.core.Array):
            tps = np.unique([int(p[0]) for p in self.points.data])
            for tp in tps:
                labels_to_keep = []
                points = [p for p in self.points.data if p[0] == tp]
                current_stack = self.labels.data[
                    tp
                ].compute()  # Compute the current stack
                for p in points:
                    labels_to_keep.append(
                        current_stack[int(p[1]), int(p[2]), int(p[3])]
                    )
                mask = functools.reduce(
                    np.logical_or,
                    (current_stack == val for val in labels_to_keep),
                )
                filtered = np.where(mask, current_stack, 0)
                self.labels.data[tp] = filtered
            self.labels.data = self.labels.data  # to trigger viewer update

        else:
            if len(self.points.data[0]) == 4:
                tps = np.unique([int(p[0]) for p in self.points.data])
                for tp in tps:
                    labels_to_keep = []
                    points = [p for p in self.points.data if p[0] == tp]
                    for p in points:
                        labels_to_keep.append(
                            self.labels.data[
                                tp, int(p[1]), int(p[2]), int(p[3])
                            ]
                        )
                    mask = functools.reduce(
                        np.logical_or,
                        (
                            self.labels.data[tp] == val
                            for val in labels_to_keep
                        ),
                    )
                    filtered = np.where(mask, self.labels.data[tp], 0)
                    self.labels.data[tp] = filtered
                self.labels.data = self.labels.data  # to trigger viewer update

            else:
                labels_to_keep = []
                for p in self.points.data:
                    if len(p) == 2:
                        labels_to_keep.append(
                            self.labels.data[int(p[0]), int(p[1])]
                        )
                    elif len(p) == 3:
                        labels_to_keep.append(
                            self.labels.data[int(p[0]), int(p[1]), int(p[2])]
                        )

                mask = functools.reduce(
                    np.logical_or,
                    (self.labels.data == val for val in labels_to_keep),
                )
                filtered = np.where(mask, self.labels.data, 0)

                self.labels = self.viewer.add_labels(
                    filtered, name=self.labels.name + "_points_kept"
                )
                self._update_labels(self.labels.name)

    def _add_option_layer(self):
        """Add a new labels layer that contains different alternative segmentations as channels, and add a function to select and copy these cells through shift-clicking"""

        path = QFileDialog.getExistingDirectory(
            self, "Select Label Image Parent Folder"
        )
        if path:
            label_dirs = sorted(
                [
                    d
                    for d in os.listdir(path)
                    if os.path.isdir(os.path.join(path, d))
                ]
            )
            label_stacks = []
            for d in label_dirs:
                # n dirs indicates number of channels
                label_files = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(path, d))
                        if ".tif" in f
                    ]
                )
                label_imgs = []
                for f in label_files:
                    # n label_files indicates n time points
                    img = imread(os.path.join(path, d, f))
                    label_imgs.append(img)

                if len(label_imgs) > 1:
                    label_stack = np.stack(label_imgs, axis=0)
                    label_stacks.append(label_stack)
                else:
                    label_stacks.append(img)

            if len(label_stacks) > 1:
                self.option_labels = np.stack(label_stacks, axis=0)
            elif len(label_stacks) == 1:
                self.option_labels = label_stacks[0]

            n_channels = len(label_dirs)
            n_timepoints = len(label_files)
            if len(img.shape) == 3:
                n_slices = img.shape[0]
            elif len(img.shape) == 2:
                n_slices = 1

            self.option_labels = self.option_labels.reshape(
                n_channels,
                n_timepoints,
                n_slices,
                img.shape[-2],
                img.shape[-1],
            )
            self.option_labels = self.viewer.add_labels(
                self.option_labels, name="label options"
            )

        viewer = self.viewer

        @viewer.mouse_drag_callbacks.append
        def cell_copied(viewer, event):
            if (
                event.type == "mouse_press"
                and "Shift" in event.modifiers
                and viewer.layers.selection.active == self.option_labels
            ):
                coords = self.option_labels.world_to_data(event.position)
                coords = [int(c) for c in coords]
                selected_label = self.option_labels.get_value(coords)
                mask = (
                    self.option_labels.data[coords[0], coords[1], :, :, :]
                    == selected_label
                )

                if isinstance(self.labels.data, da.core.Array):
                    target_stack = self.labels.data[coords[-4]].compute()
                    orig_label = target_stack[
                        coords[-3], coords[-2], coords[-1]
                    ]
                    if orig_label != 0:
                        target_stack[target_stack == orig_label] = 0
                    target_stack[mask] = np.max(target_stack) + 1
                    self.labels.data[coords[-4]] = target_stack
                    self.labels.data = self.labels.data

                else:
                    if len(self.labels.data.shape) == 3:
                        orig_label = self.labels.data[
                            coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.labels.data[
                                self.labels.data == orig_label
                            ] = 0  # set the original label to zero
                        self.labels.data[mask] = np.max(self.labels.data) + 1
                        self.labels.data = self.labels.data

                    elif len(self.labels.data.shape) == 4:
                        orig_label = self.labels.data[
                            coords[-4], coords[-3], coords[-2], coords[-1]
                        ]

                        if orig_label != 0:
                            self.labels.data[coords[-4]][
                                self.labels.data[coords[-4]] == orig_label
                            ] = 0  # set the original label to zero
                        self.labels.data[coords[-4]][mask] = (
                            np.max(self.labels.data) + 1
                        )
                        self.labels.data = self.labels.data

                    elif len(self.labels.data.shape) == 5:
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setText(
                            "Copy-pasting in 5 dimensions is not implemented, do you want to convert the labels layer to 5 dimensions (tzyx)?"
                        )
                        msg_box.setWindowTitle("Convert to 4 dimensions?")

                        yes_button = msg_box.addButton(QMessageBox.Yes)
                        no_button = msg_box.addButton(QMessageBox.No)

                        msg_box.exec_()

                        if msg_box.clickedButton() == yes_button:
                            self.labels.data = self.labels.data[0]
                        elif msg_box.clickedButton() == no_button:
                            return False
                    else:
                        print(
                            "copy-pasting in more than 5 dimensions is not supported"
                        )

    def _delete_small_objects(self) -> None:
        """Delete small objects in the selected layer"""

        if isinstance(self.labels.data, da.core.Array):
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(
                    self.outputdir, (self.labels.name + "_sizefiltered")
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.labels.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.labels.data[
                        i
                    ].compute()  # Compute the current stack

                    # measure the sizes in pixels of the labels in slice using skimage.regionprops
                    props = measure.regionprops(current_stack)
                    filtered_labels = [
                        p.label
                        for p in props
                        if p.area > self.min_size_field.value()
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
                                self.labels.name
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
                self.labels = self.viewer.add_labels(
                    da.stack([imread(fname) for fname in sorted(file_list)]),
                    name=self.labels.name + "_sizefiltered",
                )
                self._update_labels(self.labels.name)

        else:
            # Image data is a normal array and can be directly edited.
            if len(self.labels.data.shape) == 4:
                stack = []
                for i in range(self.labels.data.shape[0]):
                    props = measure.regionprops(self.labels.data[i])
                    filtered_labels = [
                        p.label
                        for p in props
                        if p.area > self.min_size_field.value()
                    ]
                    mask = functools.reduce(
                        np.logical_or,
                        (
                            self.labels.data[i] == val
                            for val in filtered_labels
                        ),
                    )
                    filtered = np.where(mask, self.labels.data[i], 0)
                    stack.append(filtered)
                self.labels = self.viewer.add_labels(
                    np.stack(stack, axis=0),
                    name=self.labels.name + "_sizefiltered",
                )
                self._update_labels(self.labels.name)

            elif len(self.labels.data.shape) == 3:
                props = measure.regionprops(self.labels.data)
                filtered_labels = [
                    p.label
                    for p in props
                    if p.area > self.min_size_field.value()
                ]
                mask = functools.reduce(
                    np.logical_or,
                    (self.labels.data == val for val in filtered_labels),
                )
                self.labels = self.viewer.add_labels(
                    np.where(mask, self.labels.data, 0),
                    name=self.labels.name + "_sizefiltered",
                )
                self._update_labels(self.labels.name)

            else:
                print("input should be 3D or 4D array")

    def _erode_labels(self):
        """Shrink oversized labels through erosion"""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()
        structuring_element = np.ones(
            (diam, diam, diam), dtype=bool
        )  # Define a 3x3x3 structuring element for 3D erosion

        if isinstance(self.labels.data, da.core.Array):
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(
                    self.outputdir, (self.labels.name + "_eroded")
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.labels.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.labels.data[
                        i
                    ].compute()  # Compute the current stack
                    mask = current_stack > 0
                    filled_mask = ndimage.binary_fill_holes(mask)
                    eroded_mask = binary_erosion(
                        filled_mask,
                        structure=structuring_element,
                        iterations=iterations,
                    )
                    eroded = np.where(eroded_mask, current_stack, 0)
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.labels.name
                                + "_eroded_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(eroded, dtype="uint16"),
                    )

                file_list = [
                    os.path.join(outputdir, fname)
                    for fname in os.listdir(outputdir)
                    if fname.endswith(".tif")
                ]
                self.labels = self.viewer.add_labels(
                    da.stack([imread(fname) for fname in sorted(file_list)]),
                    name=self.labels.name + "_eroded",
                )
                self._update_labels(self.labels.name)
                return True

        else:
            if len(self.labels.data.shape) == 4:
                stack = []
                for i in range(self.labels.data.shape[0]):
                    mask = self.labels.data[i] > 0
                    filled_mask = ndimage.binary_fill_holes(mask)
                    eroded_mask = binary_erosion(
                        filled_mask,
                        structure=structuring_element,
                        iterations=iterations,
                    )
                    stack.append(np.where(eroded_mask, self.labels.data[i], 0))
                self.labels = self.viewer.add_labels(
                    np.stack(stack, axis=0), name=self.labels.name + "_eroded"
                )
                self._update_labels(self.labels.name)
            elif len(self.labels.data.shape) == 3:
                mask = self.labels.data > 0
                filled_mask = ndimage.binary_fill_holes(mask)
                eroded_mask = binary_erosion(
                    filled_mask,
                    structure=structuring_element,
                    iterations=iterations,
                )
                self.labels = self.viewer.add_labels(
                    np.where(eroded_mask, self.labels.data, 0),
                    name=self.labels.name + "_eroded",
                )
                self._update_labels(self.labels.name)
            else:
                print("4D or 3D array required!")

    def _dilate_labels(self):
        """Dilate labels in the selected layer."""

        diam = self.structuring_element_diameter.value()
        iterations = self.iterations.value()

        if isinstance(self.labels.data, da.core.Array):
            if self.outputdir is None:
                msg = QMessageBox()
                msg.setWindowTitle("No output directory selected")
                msg.setText("Please specify an output directory first!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return False

            else:
                outputdir = os.path.join(
                    self.outputdir, (self.labels.name + "_dilated")
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.labels.data.shape[0]
                ):  # Loop over the first dimension
                    expanded_labels = self.labels.data[
                        i
                    ].compute()  # Compute the current stack
                    for _j in range(iterations):
                        expanded_labels = expand_labels(
                            expanded_labels, distance=diam
                        )
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.labels.name
                                + "_dilated_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(expanded_labels, dtype="uint16"),
                    )

                file_list = [
                    os.path.join(outputdir, fname)
                    for fname in os.listdir(outputdir)
                    if fname.endswith(".tif")
                ]
                self.labels = self.viewer.add_labels(
                    da.stack([imread(fname) for fname in sorted(file_list)]),
                    name=self.labels.name + "_dilated",
                )
                self._update_labels(self.labels.name)
                return True

        else:
            if len(self.labels.data.shape) == 4:
                stack = []
                for i in range(self.labels.data.shape[0]):
                    expanded_labels = self.labels.data[i]
                    for _i in range(iterations):
                        expanded_labels = expand_labels(
                            expanded_labels, distance=diam
                        )
                    stack.append(expanded_labels)
                self.labels = self.viewer.add_labels(
                    np.stack(stack, axis=0), name=self.labels.name + "_dilated"
                )
                self._update_labels(self.labels.name)

            elif len(self.labels.data.shape) == 3:
                expanded_labels = self.labels.data
                for _i in range(iterations):
                    expanded_labels = expand_labels(
                        expanded_labels, distance=diam
                    )

                self.labels = self.viewer.add_labels(
                    expanded_labels, name=self.labels.name + "_dilated"
                )
                self._update_labels(self.labels.name)
            else:
                print("input should be a 3D or 4D stack")

    def _threshold(self) -> None:
        """Threshold the selected label or intensity image"""

        if isinstance(self.threshold_layer.data, da.core.Array):
            msg = QMessageBox()
            msg.setWindowTitle(
                "Thresholding not yet implemented for dask arrays"
            )
            msg.setText("Thresholding not yet implemented for dask arrays")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        thresholded = (
            self.threshold_layer.data >= int(self.min_threshold.value())
        ) & (self.threshold_layer.data <= int(self.max_threshold.value()))
        self.viewer.add_labels(
            thresholded, name=self.threshold_layer.name + "_thresholded"
        )

    def _calculate_images(self) -> None:
        """Add label image 2 to label image 1"""

        if isinstance(self.image1_layer, da.core.Array) or isinstance(
            self.image2_layer, da.core.Array
        ):
            msg = QMessageBox()
            msg.setWindowTitle(
                "Cannot yet run image calculator on dask arrays"
            )
            msg.setText("Cannot yet run image calculator on dask arrays")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        if self.image1_layer.data.shape != self.image2_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        if self.operation.currentText() == "Add":
            self.viewer.add_image(
                np.add(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Subtract":
            self.viewer.add_image(
                np.subtract(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Multiply":
            self.viewer.add_image(
                np.multiply(self.image1_layer.data, self.image2_layer.data)
            )
        if self.operation.currentText() == "Divide":
            self.viewer.add_image(
                np.divide(
                    self.image1_layer.data,
                    self.image2_layer.data,
                    out=np.zeros_like(self.image1_layer.data, dtype=float),
                    where=self.image2_layer.data != 0,
                )
            )
        if self.operation.currentText() == "AND":
            self.viewer.add_labels(
                np.logical_and(
                    self.image1_layer.data != 0, self.image2_layer.data != 0
                ).astype(int)
            )
        if self.operation.currentText() == "OR":
            self.viewer.add_labels(
                np.logical_or(
                    self.image1_layer.data != 0, self.image2_layer.data != 0
                ).astype(int)
            )

    def status_callback(self):
        """Count the number of iterations processed"""
        
        def callback(levelset):      
            print("Iteration", self.counter, 'out of', self.num_iter_spin.value())
            show_info("Iteration " + str(self.counter) + ' out of ' + str(self.num_iter_spin.value()))
            self.counter += 1
        return callback

    def _morphological_active_contour(self) -> None:
        """Run morphological active contour algorithm"""

        if isinstance(self.int_input_layer, da.core.Array) or isinstance(
            self.seeds_layer, da.core.Array
        ):
            msg = QMessageBox()
            msg.setWindowTitle("Please convert to an in memory array")
            msg.setText("Please convert to an in memory array")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False
        if self.int_input_layer.data.shape != self.seeds_layer.data.shape:
            msg = QMessageBox()
            msg.setWindowTitle("Images must have the same shape")
            msg.setText("Images must have the same shape")
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return False

        self.counter = 0
        callback = self.status_callback()
        self.viewer.add_labels(
            ms.morphological_chan_vese(self.int_input_layer.data, iterations=self.num_iter_spin.value(),
                               init_level_set=self.seeds_layer.data,
                               smoothing=0, lambda1=self.lambda1_spin.value(), lambda2=self.lambda2_spin.value(), iter_callback=callback)
        )
