"""
Collection of segmentation related widgets
"""

import os
import shutil

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap, to_rgb
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

from ..layer_selection.layer_manager import LayerManager
from ..plots.plot_widget import PlotWidget
from ..tables.custom_table_widget import ColoredTableWidget
from .connected_components import ConnectedComponents
from .erosion_dilation_widget import ErosionDilationWidget
from .image_calculator import ImageCalculator
from .median_filter import MedianFilter
from .morph_reconstruction_widget import MorphReconstructionWidget
from .size_filter_widget import SizeFilterWidget
from .smoothing_widget import SmoothingWidget
from .threshold_widget import ThresholdWidget


class SegmentationWidgets(QWidget):
    """Widget for manual correction of label data, for example to prepare ground truth data for training a segmentation model"""

    def __init__(self, viewer: "napari.viewer.Viewer", layer_manager: LayerManager) -> None:
        super().__init__()
        self.viewer = viewer
        self.layer_manager = layer_manager
        self.tab_widget = QTabWidget(self)

        ## Define segmentation widgets
        segmentation_layout = QVBoxLayout()
        self.outputdir = None

        ### specify output directory
        outputbox_layout = QHBoxLayout()
        self.outputdirbtn = QPushButton("Select output directory")
        self.output_path = QLineEdit()
        outputbox_layout.addWidget(self.outputdirbtn)
        outputbox_layout.addWidget(self.output_path)
        self.outputdirbtn.clicked.connect(self._on_get_output_dir)
        segmentation_layout.addLayout(outputbox_layout)

        ### Add widget for adding overview table
        self.table_btn = QPushButton("Show table")
        self.table_btn.clicked.connect(self._create_summary_table)
        self.table_btn.clicked.connect(
            lambda: self.tab_widget.setCurrentIndex(1)
        )
        if self.layer_manager.selected_layer is not None:
            self.table_btn.setEnabled(True)
        segmentation_layout.addWidget(self.table_btn)

        ## Add save labels widget
        self.save_btn = QPushButton("Save labels")
        self.save_btn.clicked.connect(self._save_labels)
        segmentation_layout.addWidget(self.save_btn)

        ## Add button to clear all layers
        self.clear_btn = QPushButton("Clear all layers")
        self.clear_btn.clicked.connect(self._clear_layers)
        segmentation_layout.addWidget(self.clear_btn)

        ### Add widget for connected component labeling
        conn_comp_widget = ConnectedComponents(self.viewer, self.layer_manager)
        segmentation_layout.addWidget(conn_comp_widget)

        ### Add widget for size filtering
        size_filter_widget = SizeFilterWidget(self.viewer, self.layer_manager)
        segmentation_layout.addWidget(size_filter_widget)

        ### Add widget for eroding/dilating labels
        erode_dilate_widget = ErosionDilationWidget(
            self.viewer, self.layer_manager
        )
        segmentation_layout.addWidget(erode_dilate_widget)

        ### Threshold image
        threshold_widget = ThresholdWidget(self.viewer)
        segmentation_layout.addWidget(threshold_widget)

        ### Add one image to another
        image_calc = ImageCalculator(self.viewer)
        segmentation_layout.addWidget(image_calc)

        ### Add widget for median filter
        median_widget = MedianFilter(self.viewer, self.layer_manager)
        segmentation_layout.addWidget(median_widget)

        ### Add widget for smoothing labels
        smooth_close_widget = SmoothingWidget(self.viewer, self.layer_manager)
        segmentation_layout.addWidget(smooth_close_widget)

        ### Widget for custom region growing
        segmentation_layout.addWidget(MorphReconstructionWidget(self.viewer))

        ### Combine in QScrollArea
        segmentation_widgets = QWidget()
        segmentation_widgets.setLayout(segmentation_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(segmentation_widgets)
        scroll_area.setWidgetResizable(True)

        ## Define label plot widgets
        self.label_table_widget = ColoredTableWidget(
            napari.layers.Labels(np.zeros((10, 10), dtype=np.uint8)),
            self.viewer,
        )

        self.label_plot_widget = PlotWidget(props=pd.DataFrame())
        self.label_plotting_widgets = QWidget()
        self.label_plotting_widgets_layout = QVBoxLayout()
        self.label_plotting_widgets_layout.addWidget(self.label_table_widget)
        self.label_plotting_widgets_layout.addWidget(self.label_plot_widget)
        self.label_plotting_widgets.setLayout(
            self.label_plotting_widgets_layout
        )

        ## Combine tabs
        self.tab_widget.addTab(scroll_area, "Segmentation Tools")
        self.tab_widget.addTab(self.label_plotting_widgets, "View Labels")

        ### Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def _on_get_output_dir(self) -> None:
        """Show a dialog window to let the user pick the output directory."""

        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)
            self.outputdir = str(self.output_path.text())

    def _create_summary_table(self) -> None:
        """Create table displaying the sizes of the different labels in the current stack"""

        if isinstance(self.layer_manager.selected_layer.data, da.core.Array):
            tp = self.viewer.dims.current_step[0]
            current_stack = self.layer_manager.selected_layer.data[
                tp
            ].compute()  # Compute the current stack
            self.label_table = measure.regionprops_table(
                current_stack, properties=["label", "area", "centroid"]
            )
            if hasattr(self.layer_manager.selected_layer, "properties"):
                self.layer_manager.selected_layer.properties = self.label_table
            if hasattr(self.layer_manager.selected_layer, "features"):
                self.layer_manager.selected_layer.features = self.label_table

        else:
            if len(self.layer_manager.selected_layer.data.shape) == 4:
                tp = self.viewer.dims.current_step[0]
                self.label_table = measure.regionprops_table(
                    self.layer_manager.selected_layer.data[tp],
                    properties=["label", "area", "centroid"],
                )
                if hasattr(self.layer_manager.selected_layer, "properties"):
                    self.layer_manager.selected_layer.properties = self.label_table
                if hasattr(self.layer_manager.selected_layer, "features"):
                    self.layer_manager.selected_layer.features = self.label_table

            elif len(self.layer_manager.selected_layer.data.shape) == 3:
                self.label_table = measure.regionprops_table(
                    self.layer_manager.selected_layer.data, properties=["label", "area", "centroid"]
                )
                if hasattr(self.layer_manager.selected_layer, "properties"):
                    self.layer_manager.selected_layer.properties = self.label_table
                if hasattr(self.layer_manager.selected_layer, "features"):
                    self.layer_manager.selected_layer.features = self.label_table
            else:
                print("input should be a 3D or 4D array")
                self.label_table = None

        if self.label_table_widget is not None:
            self.label_table_widget.hide()

        if self.viewer is not None:
            self.label_table_widget = ColoredTableWidget(
                self.layer_manager.selected_layer, self.viewer
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
                to_rgb(self.layer_manager.selected_layer.get_color(label)) for label in unique_labels
            ]
            self.label_plot_widget.label_colormap = ListedColormap(
                label_colors
            )
            self.label_plot_widget._update_dropdowns()

    def _save_labels(self) -> None:
        """Save the currently active labels layer. If it consists of multiple timepoints, they are written to multiple 3D stacks."""

        if isinstance(self.layer_manager.selected_layer.data, da.core.Array):

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
                    self.outputdir, (self.layer_manager.selected_layer.name + "_finalresult")
                )
                if os.path.exists(outputdir):
                    shutil.rmtree(outputdir)
                os.mkdir(outputdir)

                for i in range(
                    self.layer_manager.selected_layer.data.shape[0]
                ):  # Loop over the first dimension
                    current_stack = self.layer_manager.selected_layer.data[
                        i
                    ].compute()  # Compute the current stack
                    tifffile.imwrite(
                        os.path.join(
                            outputdir,
                            (
                                self.layer_manager.selected_layer.name
                                + "_TP"
                                + str(i).zfill(4)
                                + ".tif"
                            ),
                        ),
                        np.array(current_stack, dtype="uint16"),
                    )
                return True

        elif len(self.layer_manager.selected_layer.data.shape) == 4:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )
            for i in range(self.layer_manager.selected_layer.data.shape[0]):
                labels_data = self.layer_manager.selected_layer.data[i].astype(np.uint16)
                tifffile.imwrite(
                    (
                        filename.split(".tif")[0]
                        + "_TP"
                        + str(i).zfill(4)
                        + ".tif"
                    ),
                    labels_data,
                )

        elif len(self.layer_manager.selected_layer.data.shape) == 3:
            filename, _ = QFileDialog.getSaveFileName(
                caption="Save Labels",
                directory="",
                filter="TIFF files (*.tif *.tiff)",
            )

            if filename:
                labels_data = self.layer_manager.selected_layer.data.astype(np.uint16)
                tifffile.imwrite(filename, labels_data)

        else:
            print("labels should be a 3D or 4D array")

    def _clear_layers(self) -> None:
        """Clear all the layers in the viewer"""

        self.viewer.layers.clear()
