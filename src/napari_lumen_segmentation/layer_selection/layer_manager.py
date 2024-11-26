import dask.array as da
import napari
import numpy as np
from qtpy.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

from .layer_dropdown import LayerDropdown


class LayerManager(QWidget):
    """QComboBox widget with functions for updating the selected layer and to update the list of options when the list of layers is modified."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self._selected_layer = None

        self.label_dropdown = LayerDropdown(
            self.viewer, (napari.layers.Labels)
        )
        self.label_dropdown.layer_changed.connect(self._update_labels)

        ### Add option to convert dask array to in-memory array
        self.convert_to_array_btn = QPushButton("Convert to in-memory array")
        self.convert_to_array_btn.setEnabled(
            self.selected_layer is not None
            and isinstance(self.selected_layer.data, da.core.Array)
        )
        self.convert_to_array_btn.clicked.connect(self._convert_to_array)

        box = QGroupBox("Selected Labels Layer")
        widget_layout = QVBoxLayout()
        widget_layout.addWidget(self.label_dropdown)
        widget_layout.addWidget(self.convert_to_array_btn)
        box.setLayout(widget_layout)

        layout = QVBoxLayout()
        layout.addWidget(box)

        self.setLayout(layout)

    @property
    def selected_layer(self):
        return self._selected_layer

    @selected_layer.setter
    def selected_layer(self, layer):
        if layer != self._selected_layer:
            self._selected_layer = layer

    def _update_labels(self, selected_layer) -> None:
        """Update the layer that is set to be the 'labels' layer that is being edited."""

        if selected_layer == "":
            self._selected_layer = None
        else:
            self.selected_layer = self.viewer.layers[selected_layer]
            self.label_dropdown.setCurrentText(selected_layer)
            self.convert_to_array_btn.setEnabled(
                isinstance(self._selected_layer.data, da.core.Array)
            )

    def _convert_to_array(self) -> None:
        """Convert from dask array to in-memory array. This is necessary for manual editing using the label tools (brush, eraser, fill bucket)."""

        if isinstance(self._selected_layer.data, da.core.Array):
            stack = []
            for i in range(self._selected_layer.data.shape[0]):
                current_stack = self._selected_layer.data[i].compute()
                stack.append(current_stack)
            self._selected_layer.data = np.stack(stack, axis=0)
