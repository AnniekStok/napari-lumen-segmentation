import os
from pathlib                            import Path 
import napari.layers

import matplotlib.pyplot                as plt
import pandas                           as pd
import numpy                            as np

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from qtpy.QtWidgets                     import QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QLabel
from qtpy.QtGui                         import QIcon

ICON_ROOT = Path(__file__).parent / "icons"

class PlotWidget(QWidget):
    """Customized plotting widget class.
    """

    def __init__(self, props: pd.DataFrame):
        super().__init__()

        self.props = props
        self.label_colormap = None
        self.categorical_cmap = 'tab10' # may be overwritten by parent 
        self.continuous_cmap = 'summer' # may be overwritten by parent 

        # Main plot.
        self.fig = plt.figure()
        self.plot_canvas = FigureCanvas(self.fig)
        self.ax = self.plot_canvas.figure.subplots()
        self.toolbar = NavigationToolbar2QT(self.plot_canvas)

        # Specify plot customizations.
        self.fig.patch.set_facecolor("#262930")
        self.ax.tick_params(colors='white')
        self.ax.set_facecolor("#262930")
        self.ax.xaxis.label.set_color('white') 
        self.ax.yaxis.label.set_color('white') 
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        for action_name in self.toolbar._actions:
            action=self.toolbar._actions[action_name]
            icon_path = os.path.join(ICON_ROOT, action_name + ".png")
            action.setIcon(QIcon(icon_path))

        # Create a dropdown window for selecting what to plot on the axes.
        x_axis_layout = QHBoxLayout()
        self.x_combo = QComboBox()
        self.x_combo.addItems([item for item in self.props.columns if item != 'index'])
        x_axis_layout.addWidget(QLabel('x-axis'))
        x_axis_layout.addWidget(self.x_combo)

        y_axis_layout = QHBoxLayout()
        self.y_combo = QComboBox()
        self.y_combo.addItems([item for item in self.props.columns if item != 'index'])
        y_axis_layout.addWidget(QLabel('y-axis'))
        y_axis_layout.addWidget(self.y_combo)

        self.x_combo.currentIndexChanged.connect(self._update_plot)
        self.y_combo.currentIndexChanged.connect(self._update_plot)

        color_group_layout = QHBoxLayout()
        self.group_combo = QComboBox()
        self.group_combo.addItems([item for item in self.props.columns if item != 'index'])
        self.group_combo.currentIndexChanged.connect(self._update_plot)
        color_group_layout.addWidget(QLabel('Group color'))
        color_group_layout.addWidget(self.group_combo)

        dropdown_layout = QVBoxLayout()
        dropdown_layout.addLayout(x_axis_layout)
        dropdown_layout.addLayout(y_axis_layout)
        dropdown_layout.addLayout(color_group_layout)
        dropdown_widget = QWidget()
        dropdown_widget.setLayout(dropdown_layout)

        # Create and apply a horizontal layout for the dropdown widget, toolbar and canvas.
        plotting_layout = QVBoxLayout()
        plotting_layout.addWidget(dropdown_widget)
        plotting_layout.addWidget(self.toolbar)
        plotting_layout.addWidget(self.plot_canvas)
        self.setLayout(plotting_layout)     
  
    def _update_properties(self, filtered_measurements): 
        """Update the properties and regenerate the plot"""
        self.props = filtered_measurements[filtered_measurements['selected']]
        self._update_plot()

    def _update_dropdowns(self) -> None: 
        """Update the options in the dropdown menus"""

        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.group_combo.blockSignals(True)
        
        self.x_combo.clear()
        self.y_combo.clear()
        self.group_combo.clear()
        self.x_combo.addItems([item for item in self.props.columns if item != 'index'])
        self.x_combo.setCurrentIndex(0)
        self.y_combo.addItems([item for item in self.props.columns if item != 'index'])
        self.y_combo.setCurrentIndex(1)
        self.group_combo.addItems([item for item in self.props.columns if item != 'index'])
        self.group_combo.setCurrentIndex(0)

        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self.group_combo.blockSignals(False)

        self._update_plot()

    def _update_plot(self) -> None:
        """Update the plot by plotting the features selected by the user. 

        """

        if not self.props.empty:
            x_axis_property = self.x_combo.currentText()
            y_axis_property = self.y_combo.currentText()
            group = self.group_combo.currentText()
            
            # Clear data points, and reset the axis scaling
            for artist in self.ax.lines + self.ax.collections:
                artist.remove()
            self.ax.set_xlabel(x_axis_property)
            self.ax.set_ylabel(y_axis_property)
            self.ax.relim()  # Recalculate limits for the current data
            self.ax.autoscale_view()  # Update the view to include the new limits
        
            if group == "label":
                # plot using label colors
                if self.label_colormap is not None:
                    self.ax.scatter(self.props[x_axis_property], self.props[y_axis_property], c=self.props[group], cmap=self.label_colormap, s = 10)
                else: 
                    self.ax.scatter(self.props[x_axis_property], self.props[y_axis_property], c=self.props[group], cmap=self.categorical_cmap, s = 10)
            else:
                # Plot data points on a custom categorical or continuous colormap.
                if self.props[group].dtype == 'object' or np.issubdtype(self.props[group].dtype, np.integer):

                    unique_categories = np.unique(self.props[group])
                    cmap_colors = self.categorical_cmap.colors
                    if len(unique_categories) <= len(cmap_colors):
                        category_to_color = {category: cmap_colors[i] for i, category in enumerate(unique_categories)}
                        colors = [category_to_color[category] for category in self.props[group]]
                        self.ax.scatter(self.props[x_axis_property], self.props[y_axis_property], c=colors, cmap=self.categorical_cmap, s = 10)
                    else: 
                        self.ax.scatter(self.props[x_axis_property], self.props[y_axis_property], c=self.props[group], cmap=self.categorical_cmap, s = 10)
                else:
                    self.ax.scatter(self.props[x_axis_property], self.props[y_axis_property], c=self.props[group], cmap=self.continuous_cmap, s = 10)

            self.plot_canvas.draw()