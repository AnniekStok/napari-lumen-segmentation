import napari
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgb
from napari_skimage_regionprops import TableWidget
from pandas import DataFrame
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class ColoredTableWidget(TableWidget):
    """Customized table widget derived from the napari_skimage_regionprops TableWidget"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ascending = (
            False  # for choosing whether to sort ascending or descending
        )

        # Reconnect the clicked signal to your custom method.
        self._view.clicked.connect(self._clicked_table)

        # Connect to single click in the header to sort the table.
        self._view.horizontalHeader().sectionClicked.connect(self._sort_table)

    def _set_label_colors_to_rows(self) -> None:
        """Apply the colors of the napari label image to the table"""

        for i in range(self._view.rowCount()):
            label = self._table["label"][i]
            label_color = to_rgb(self._layer.get_color(label))
            scaled_color = (
                int(label_color[0] * 255),
                int(label_color[1] * 255),
                int(label_color[2] * 255),
            )
            for j in range(self._view.columnCount()):
                self._view.item(i, j).setBackground(QColor(*scaled_color))

    def _clicked_table(self):
        """Also set show_selected_label to True and jump to the corresponding stack position"""

        super()._clicked_table()
        self._layer.show_selected_label = True

        row = self._view.currentRow()
        z = int(self._table["centroid-0"][row])
        current_step = self._viewer.dims.current_step
        if len(current_step) == 4:
            new_step = (current_step[0], z, current_step[2], current_step[3])
        elif len(current_step) == 3:
            new_step = (z, current_step[1], current_step[2])
        else:
            new_step = current_step
        self._viewer.dims.current_step = new_step

    def _sort_table(self):
        """Sorts the table in ascending or descending order"""

        selected_column = list(self._table.keys())[self._view.currentColumn()]
        df = pd.DataFrame(self._table).sort_values(
            by=selected_column, ascending=self.ascending
        )
        self.ascending = not self.ascending

        self.set_content(df.to_dict(orient="list"))
        self._set_label_colors_to_rows()


class CustomTableWidget(QWidget):
    """
    Custom table widget based on the napari_skimage_regionprops TableWdiget
    """

    def __init__(
        self, props: pd.DataFrame | None, viewer: "napari.Viewer" = None
    ):
        super().__init__()

        self._viewer = viewer
        self.sort_by = None
        self.colormap = None

        self._view = QTableWidget()
        self._view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ascending = False
        self._view.horizontalHeader().sectionClicked.connect(self._sort_table)

        if props is None:
            self.props = pd.DataFrame().to_dict(orient="list")
        else:
            self.props = props.to_dict(orient="list")
        self.set_content(self.props)

        # copy from napari_skimage_regionprops
        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_clicked)

        save_button = QPushButton("Save as csv...")
        save_button.clicked.connect(self._save_clicked)

        self.setLayout(QGridLayout())
        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(copy_button)
        action_widget.layout().addWidget(save_button)
        self.layout().addWidget(action_widget)
        self.layout().addWidget(self._view)
        action_widget.layout().setSpacing(3)
        action_widget.layout().setContentsMargins(0, 0, 0, 0)

    def _save_clicked(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save as csv...", ".", "*.csv"
            )
        DataFrame(self._table).to_csv(filename)

    def _copy_clicked(self):
        DataFrame(self._table).to_clipboard()

    def set_content(self, table: dict):
        """
        Overwrites the content of the table with the content of a given dictionary.
        """
        if table is None:
            table = {}

        self._table = table

        self._view.clear()
        try:
            self._view.setRowCount(len(next(iter(table.values()))))
            self._view.setColumnCount(len(table))
        except StopIteration:
            pass

        for i, column in enumerate(table.keys()):

            self._view.setHorizontalHeaderItem(i, QTableWidgetItem(column))
            for j, value in enumerate(table.get(column)):
                self._view.setItem(j, i, QTableWidgetItem(str(value)))

    def get_content(self) -> dict:
        """
        Returns the current content of the table
        """
        return self._table

    def _sort_table(self):
        """Sorts the table in ascending or descending order"""

        selected_column = list(self._table.keys())[self._view.currentColumn()]
        df = pd.DataFrame(self._table).sort_values(
            by=selected_column, ascending=self.ascending
        )
        self.ascending = not self.ascending
        self.set_content(df.to_dict(orient="list"))
        if self.sort_by is not None:
            self._recolor(self.sort_by, self.colormap)

    def _recolor(self, by: str, cmap: ListedColormap):
        """Assign colors to the table based on given column and colormap"""

        default_color = self.palette().color(self.backgroundRole())
        if by is None:
            for i in range(self._view.rowCount()):
                for j in range(self._view.columnCount()):
                    self._view.item(i, j).setBackground(default_color)

        else:
            for i in range(self._view.rowCount()):
                label = self._table[by][i]
                color = to_rgb(cmap.colors[label])
                scaled_color = (
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255),
                )
                for j in range(self._view.columnCount()):
                    self._view.item(i, j).setBackground(QColor(*scaled_color))

        self.sort_by = by
        self.colormap = cmap
