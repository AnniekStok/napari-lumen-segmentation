"""
Napari plugin widget for editing N-dimensional label data
"""

import os
import shutil

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap, to_rgb
from napari_plane_sliders._plane_slider_widget import PlaneSliderWidget
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage import measure

from .distance_widget import DistanceWidget
from .morph_reconstruction_widget import MorphReconstructionWidget
from .plot_widget import PlotWidget
from .skeleton_widget import SkeletonWidget
from .connected_components import ConnectedComponents
from .custom_table_widget import ColoredTableWidget, CustomTableWidget
from .erosion_dilation_widget import ErosionDilationWidget
from .image_calculator import ImageCalculator
from .layer_manager import LayerManager
from .size_filter_widget import SizeFilterWidget
from .smoothing_widget import SmoothingWidget
from .median_filter import MedianFilter
from .threshold_widget import ThresholdWidget


class LumenSegmentationWidget(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self.viewer = viewer

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
        self.table_widget = CustomTableWidget(props=pd.DataFrame())
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
        self.label_manager = LayerManager(self.viewer)
        self.segmentation_layout.addWidget(self.label_manager)

        ### Add widget for adding overview table
        self.table_btn = QPushButton("Show table")
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.clicked.connect(
            lambda: self.tab_widget.setCurrentIndex(2)
        )
        if self.label_manager.selected_layer is not None:
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

        ### Add widget for connected component labeling
        conn_comp_widget = ConnectedComponents(self.viewer, self.label_manager)
        self.segmentation_layout.addWidget(conn_comp_widget)

        ### Add widget for size filtering
        size_filter_widget = SizeFilterWidget(self.viewer, self.label_manager)
        self.segmentation_layout.addWidget(size_filter_widget)

        ### Add widget for eroding/dilating labels
        erode_dilate_widget = ErosionDilationWidget(
            self.viewer, self.label_manager
        )
        self.segmentation_layout.addWidget(erode_dilate_widget)

        ### Threshold image
        threshold_widget = ThresholdWidget(self.viewer)
        self.segmentation_layout.addWidget(threshold_widget)

        ### Add one image to another
        image_calc = ImageCalculator(self.viewer)
        self.segmentation_layout.addWidget(image_calc)

        ### Add widget for smoothing labels
        median_widget = MedianFilter(self.viewer, self.label_manager)
        self.segmentation_layout.addWidget(median_widget)

        ### Add widget for smoothing labels
        smooth_close_widget = SmoothingWidget(self.viewer, self.label_manager)
        self.segmentation_layout.addWidget(smooth_close_widget)

        # widget for custom region growing
        self.segmentation_layout.addWidget(MorphReconstructionWidget(self.viewer))

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
            viewer=self.viewer, label_manager=self.label_manager
        )
        self.tab_widget.addTab(self.skeleton_widget, "Skeleton Analysis")

        self.distance_widget = DistanceWidget(
            viewer=self.viewer, label_manager=self.label_manager
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

        if self.label_manager.selected_layer is not None:
            self.label_manager.selected_layer = self.viewer.add_labels(
                measure.label(self.label_manager.selected_layer.data)
            )
            self._update_labels(self.label_manager.selected_layer.name)

    def _update_labels(self, selected_layer: str) -> None:
        """Update the layer that is set to be the 'labels' layer that is being edited."""

        if selected_layer == "":
            self.label_manager.selected_layer = None
        else:
            self.label_manager.selected_layer = self.viewer.layers[selected_layer]
            self.label_dropdown.setCurrentText(selected_layer)
            self.convert_to_array_btn.setEnabled(
                isinstance(self.label_manager.selected_layer.data, da.core.Array)
            )
            self.skeleton_widget.labels = self.label_manager.selected_layer
            self.distance_widget.labels = self.label_manager.selected_layer

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

    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):
            tp = self.viewer.dims.current_step[0]
            current_stack = self.label_manager.selected_layer.data[
                tp
            ].compute()  # Compute the current stack
            self.label_table = measure.regionprops_table(
                current_stack, properties=["label", "area", "centroid"]
            )
            if hasattr(self.label_manager.selected_layer, "properties"):
                self.label_manager.selected_layer.properties = self.label_table
            if hasattr(self.label_manager.selected_layer, "features"):
                self.label_manager.selected_layer.features = self.label_table

        else:
            if len(self.label_manager.selected_layer.data.shape) == 4:
                tp = self.viewer.dims.current_step[0]
                self.label_table = measure.regionprops_table(
                    self.label_manager.selected_layer.data[tp],
                    properties=["label", "area", "centroid"],
                )
                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = self.label_table
                if hasattr(self.label_manager.selected_layer, "features"):
                    self.label_manager.selected_layer.features = self.label_table

            elif len(self.label_manager.selected_layer.data.shape) == 3:
                self.label_table = measure.regionprops_table(
                    self.label_manager.selected_layer.data, properties=["label", "area", "centroid"]
                )
                if hasattr(self.label_manager.selected_layer, "properties"):
                    self.label_manager.selected_layer.properties = self.label_table
                if hasattr(self.label_manager.selected_layer, "features"):
                    self.label_manager.selected_layer.features = self.label_table
            else:
                print("input should be a 3D or 4D array")
                self.label_table = None

        if self.label_table_widget is not None:
            self.label_table_widget.hide()

        if self.viewer is not None:
            self.label_table_widget = ColoredTableWidget(
                self.label_manager.selected_layer, self.viewer
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
                to_rgb(self.label_manager.selected_layer.get_color(label)) for label in unique_labels
            ]
            self.label_plot_widget.label_colormap = ListedColormap(
                label_colors
            )
            self.label_plot_widget._update_dropdowns()

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if isinstance(self.label_manager.selected_layer.data, da.core.Array):

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
                    self.outputdir, (self.label_manager.selected_layer.name + "_finalresult")
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
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.label_manager.selected_layer.name
                                + "_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(current_stack, dtype="uint16"),
                    )
                return True

        elif len(self.label_manager.selected_layer.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )
            for i in range(self.label_manager.selected_layer.data.shape[0]):
                labels_data = self.label_manager.selected_layer.data[i].astype(np.uint16)
                tifffile.imwrite(
                    (
                        filename.split(".tif")[0]
                        + "_TP"
                        + str(i).zfill(4)
                        + ".tif"
                    ),
                    labels_data,
                )

        elif len(self.label_manager.selected_layer.data.shape) == 3:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )

            if filename:
                labels_data = self.label_manager.selected_layer.data.astype(np.uint16)
                tifffile.imwrite(filename, labels_data)

        else:
            print("labels should be a 3D or 4D array")

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        self.cross.setChecked(False)
        self.cross.layer = None
        self.viewer.layers.clear()
