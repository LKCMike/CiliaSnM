"""
:Date        : 2025-12-08 19:59:34
:LastEditTime: 2025-12-10 00:34:39
:Description : Minimal file tree component: only shows My Computer (disk root directory)
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QTreeView, QFileSystemModel, QVBoxLayout, QWidget, QLabel
)
from PyQt5.QtCore import pyqtSignal, QSortFilterProxyModel


class FileFilterProxyModel(QSortFilterProxyModel):
    """
    Custom proxy model to filter displayed file types
    """

    # filter for .ome.* and .tiff 
    SUPPORTED_EXTENSIONS = {
        '.ome.tif', '.ome.tiff', '.ome.tf2', 
        '.ome.tf8', '.ome.tfb', '.tif', '.tiff'
    }

    def filterAcceptsRow(self, source_row, source_parent):
        """
        Filter rows
        """
        source_model = self.sourceModel()
        index = source_model.index(source_row, 0, source_parent)

        # Do not filter directories
        if source_model.isDir(index):
            return True

        # Check suffix
        file_path = source_model.filePath(index)
        file_ext = Path(file_path).suffix.lower()
        full_ext = ''.join(Path(file_path).suffixes[-2:]).lower()

        # Only display filtered files
        return file_ext in self.SUPPORTED_EXTENSIONS or full_ext in self.SUPPORTED_EXTENSIONS


class FileTree(QWidget):
    """
    Alternative: simplest My Computer view
    """

    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """
        Initialize Tree UI for file explorer
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("File Explorer")
        layout.addWidget(title_label)
        self.tree_view = QTreeView()

        # File System
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")  # Partitions for Windows and / for *nix

        # Apply .tiff filter
        self.proxy_model = FileFilterProxyModel()
        self.proxy_model.setSourceModel(self.file_model)
        self.tree_view.setModel(self.proxy_model)

        # Hide unnessesary columns
        for i in range(2, 4):
            self.tree_view.hideColumn(i)
        self.tree_view.setColumnWidth(0, 800)

        # Set empty root index to display all drives
        root_index = self.proxy_model.mapFromSource(self.file_model.index(""))
        self.tree_view.setRootIndex(root_index)

        self.tree_view.clicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree_view)

    def _on_item_clicked(self, index):
        """
        Initialize user interface
        """
        if index.isValid():
            source_index = self.proxy_model.mapToSource(index)
            file_path = self.file_model.filePath(source_index)

            # emit signals when click on files
            if not self.file_model.isDir(source_index):
                self.file_selected.emit(file_path)
