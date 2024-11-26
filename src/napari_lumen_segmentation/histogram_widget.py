import os
from pathlib import Path

import matplotlib.pyplot as plt
import napari.layers
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT,
)
from napari.layers import Image
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .layer_dropdown import LayerDropdown

ICON_ROOT = Path(__file__).parent / "icons"


class HistWidget(QGroupBox):
    """Customized plotting widget class.

    Intended for interactive plotting of features in a pandas dataframe (props).
    """

    def __init__(self, title, viewer: napari.Viewer):
        super().__init__(title)

        self.viewer = viewer

        # Main plot.
        self.fig = plt.figure(constrained_layout=True)
        self.plot_canvas = FigureCanvas(self.fig)
        self.ax = self.plot_canvas.figure.subplots()
        self.toolbar = NavigationToolbar2QT(self.plot_canvas)

        # Specify plot customizations.
        self.fig.patch.set_facecolor("#262930")
        self.ax.tick_params(colors="white")
        self.ax.set_facecolor("#262930")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        for action_name in self.toolbar._actions:
            action = self.toolbar._actions[action_name]
            icon_path = os.path.join(ICON_ROOT, action_name + ".png")
            action.setIcon(QIcon(icon_path))

        # Create a dropdown window for selecting what to plot on the axes.
        x_axis_layout = QHBoxLayout()
        self.image_dropdown = LayerDropdown(self.viewer, (Image))
        x_axis_layout.addWidget(self.image_dropdown)
        apply_btn = QPushButton("Show Histogram")
        x_axis_layout.addWidget(apply_btn)
        apply_btn.clicked.connect(self._update_plot)

        dropdown_layout = QVBoxLayout()
        dropdown_layout.addLayout(x_axis_layout)
        dropdown_widget = QWidget()
        dropdown_widget.setLayout(dropdown_layout)

        # Create and apply a horizontal layout for the dropdown widget, toolbar and canvas.
        plotting_layout = QVBoxLayout()
        plotting_layout.addWidget(dropdown_widget)
        plotting_layout.addWidget(self.toolbar)
        plotting_layout.addWidget(self.plot_canvas)
        self.setLayout(plotting_layout)

    def _update_plot(self) -> None:
        """Update the histogram"""

        image_layer = self.viewer.layers[self.image_dropdown.currentText()]
        intensity_values = image_layer.data.flatten()
        intensity_values = intensity_values[
            np.isfinite(intensity_values) & (intensity_values != 0)
        ]

        # Clear data points, and reset the axis scaling
        for artist in self.ax.lines + self.ax.collections:
            artist.remove()
        self.ax.clear()
        self.ax.set_xlabel("Intensity")
        self.ax.set_ylabel("Count")
        self.ax.relim()  # Recalculate limits for the current data
        self.ax.autoscale_view()  # Update the view to include the new limits
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")

        self.ax.hist(intensity_values, bins=255, color="turquoise", alpha=0.7)

        self.plot_canvas.draw()
