"""
Main window module - CiliaSnM
Three-column layout
left file tree
middle preview and adjustment
right log monitoring
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from multiprocessing import Process
import subprocess

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QGroupBox, QDoubleSpinBox, QGridLayout, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

from ui.file_tree import FileTree
from core.data_manager import DataManager
from core.image_processor import stretch_contrast, merge_channels
from core.cilia_statistic_measurement import cilia_analysis

class ProcessingThread(QThread):
    """
    Sub Process to analyse OME-TIFF file
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, image_path, image_snapshot, channel_params, output_folder):
        super().__init__()
        self.image_path = image_path
        self.image_snapshot = image_snapshot
        self.channel_params = channel_params
        self.output_folder = output_folder

    def run(self):
        """
        Process AI analysis in sub-process
        """
        self.log_signal.emit("Initialize AI analysis...")
        analysis_process = Process(
            target=cilia_analysis,
            args=(
                self.image_path,
                self.image_snapshot,
                self.channel_params,
                self.output_folder
            )
        )
        analysis_process.start()
        try:
            analysis_process.join(timeout=180)

            if analysis_process.is_alive():
                analysis_process.terminate()
                raise TimeoutError("AI analysis timeout")
            return_code = analysis_process.exitcode
            if return_code == 0:
                self.finished_signal.emit(f"Finished! Open {self.output_folder} for results.")
            else:
                self.finished_signal.emit(f"Sub-process Failed! exit-code: {return_code}")

        except Exception as e: # pylint: disable=broad-exception-caught
            self.finished_signal.emit(f"Exception Occured: {repr(e)}")

class MainWindow(QMainWindow):
    """
    Main Window
    """

    def __init__(self):
        super().__init__()

        # Initialize datamanager for parsing OME File data
        self.data_manager = DataManager()
        self.channel_params = {}
        self.rgb_image = None
        self.processing_thread = None

        # Initialize UI
        self._init_ui()
        self._connect_signals()

        # Window Properties
        self.setWindowTitle("Cilia Statistic and Measurement")
        self.resize(1400, 800)

        # Debouncing timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)
        self.update_timer.timeout.connect(self._update_preview)

    def _init_ui(self):
        """
        Full UI Initialization
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Three columns layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left - Tree View for file explorer
        self.file_tree = FileTree()
        self.file_tree.setMaximumWidth(350)
        main_layout.addWidget(self.file_tree, 1)

        # Middle - Preview and Adjustment panel
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)

        # Preview
        preview_group = QGroupBox("Image preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Please select an image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
        preview_layout.addWidget(self.preview_label)
        center_layout.addWidget(preview_group, 12)

        # Adjustment panel
        self.channel_group = QGroupBox("Channel adjustment")
        self.channel_layout = QGridLayout(self.channel_group)
        center_layout.addWidget(self.channel_group, 2)

        # Output options
        control_panel = self._create_control_panel()
        center_layout.addWidget(control_panel, 1)

        main_layout.addWidget(center_container, 3)

        # Right - Log and Fire button
        right_panel = self._create_monitor_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_control_panel(self):
        """
        Create output control panel at the bottom of the middle column
        """
        panel = QGroupBox("Output")
        layout = QHBoxLayout(panel)

        # Label
        label = QLabel("Output to:")
        layout.addWidget(label)

        # Folder path input box
        self.output_folder_input = QLineEdit()
        self.output_folder_input.setPlaceholderText("Please select a directory for output")
        layout.addWidget(self.output_folder_input, 4)

        # Select folder button
        self.browse_output_btn = QPushButton("Select")
        self.browse_output_btn.clicked.connect(self._browse_output_folder)
        layout.addWidget(self.browse_output_btn)

        # Open folder button
        self.open_folder_btn = QPushButton("Open")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        self.open_folder_btn.setEnabled(False)
        layout.addWidget(self.open_folder_btn)

        return panel

    def _create_monitor_panel(self):
        """
        Create right-side monitoring panel
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Log output box
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group, 4)

        # Start processing button
        self.start_process_btn = QPushButton("▶ Start")
        self.start_process_btn.clicked.connect(self._start_process)
        self.start_process_btn.setEnabled(False)
        self.start_process_btn.setMinimumHeight(40)
        layout.addWidget(self.start_process_btn, 1)

        return panel

    def _connect_signals(self):
        """
        Connect signals and slots
        """
        self.file_tree.file_selected.connect(self._on_file_selected)

    def _on_file_selected(self, file_path):
        """
        Handle file selection
        """
        self._log(f"Loading image: {Path(file_path).name}")

        # Load image
        success = self.data_manager.load_image(file_path)

        if success:
            # Initialize channel parameter controls
            self._init_channel_controls()

            # Update preview
            self._update_preview()

            self._log(f"Image loaded: {Path(file_path).name}")
        else:
            self._log(f"Loading image failed: {file_path}")

    def _init_channel_controls(self):
        """
        Initialize channel control widgets
        """
        # Clear existing widgets
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.channel_params = {}
        channel_count = self.data_manager.get_channel_count()

        if channel_count == 0:
            self.channel_group.setTitle("No channel")
            return

        self.channel_group.setTitle(f"Channel adjustment ({channel_count} channels)")

        # Create controls for each channel
        for i in range(channel_count):
            # Channel name
            if i < len(self.data_manager.channel_names):
                name = str(self.data_manager.channel_names[i])
            else:
                name = f"Channel {i}"

            name_label = QLabel(name)
            self.channel_layout.addWidget(name_label, i, 0)

            # Lower Threshold
            lower_label = QLabel("Lower Threshold:")
            self.channel_layout.addWidget(lower_label, i, 1)

            lower_spin = QDoubleSpinBox()
            lower_spin.setRange(0.0, 100.0)
            lower_spin.setValue(1.0)
            lower_spin.setDecimals(2)
            lower_spin.setSuffix("%")
            lower_spin.valueChanged.connect(
                lambda value, idx=i: self._on_param_changed(idx, 'lower', value)
            )
            self.channel_layout.addWidget(lower_spin, i, 2)

            # Upper Threshold
            upper_label = QLabel("Upper Threshold:")
            self.channel_layout.addWidget(upper_label, i, 3)

            upper_spin = QDoubleSpinBox()
            upper_spin.setRange(0.0, 100.0)
            upper_spin.setValue(99.0)
            upper_spin.setDecimals(2)
            upper_spin.setSuffix("%")
            upper_spin.valueChanged.connect(
                lambda value, idx=i: self._on_param_changed(idx, 'upper', value)
            )
            self.channel_layout.addWidget(upper_spin, i, 4)

            type_combo = QComboBox()
            type_combo.addItems(["Unknown", "Nuclei", "Cilia"])
            type_combo.setCurrentIndex(0) # Unknown by default
            type_combo.currentIndexChanged.connect(
                lambda idx, ch=i: self._on_channel_type_changed(ch, idx)
            )
            self.channel_layout.addWidget(type_combo, i, 5)

            # 保存参数
            self.channel_params[i] = {
                'name': name,
                'lower': 1.0,
                'upper': 99.0,
                'lower_spin': lower_spin,
                'upper_spin': upper_spin,
                'type': 0
            }


    def _on_channel_type_changed(self, channel_index: int, type_index: int):
        """
        Handle channel type selection change
        """
        if channel_index in self.channel_params:
            self.channel_params[channel_index]['type'] = type_index

            type_names = ["未指定", "Nuclei", "Cilia"]

            type_name = type_names[type_index] if type_index < len(type_names) else "未指定"
            self._log(f"Channel {channel_index} ({self.data_manager.channel_names[channel_index]}) marked as {type_name}")


    def _on_param_changed(self, channel_idx, param_type, value):
        """
        Handle parameter changes
        """
        if channel_idx in self.channel_params:
            params = self.channel_params[channel_idx]

            if param_type == 'lower':
                params['lower'] = value
                params['lower_spin'].setValue(value)
            elif param_type == 'upper':
                params['upper'] = value
                params['upper_spin'].setValue(value)

            # 使用防抖定时器
            self.update_timer.stop()
            self.update_timer.start()

    def _update_preview(self):
        """
        Update preview
        """
        if self.data_manager.max_projection is None:
            return

        channel_count = self.data_manager.get_channel_count()

        # Apply contrast stretching to each channel
        stretched_channels = []
        channel_colors = self.data_manager.get_channel_colors()

        for i in range(channel_count):
            if i in self.channel_params:
                params = self.channel_params[i]
                channel_data = self.data_manager.max_projection[i]

                stretched = stretch_contrast(
                    channel_data,
                    params['lower'],
                    params['upper']
                )
                stretched_channels.append(stretched)

        # Merge channels
        rgb_image = merge_channels(stretched_channels, channel_colors)

        if rgb_image is not None:
            # Convert to QPixmap for display
            height, width = rgb_image.shape[:2]
            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
            self.rgb_image = rgb_uint8

            bytes_per_line = 3 * width
            qimage = QImage(
                rgb_uint8.data, width, height, bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)

    def _browse_output_folder(self):
        """
        Select output folder
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Output to",
            str(Path.home())
        )
        if folder:
            self.output_folder_input.setText(folder)
            self.open_folder_btn.setEnabled(True)
            self.start_process_btn.setEnabled(True)
            self._log(f"Output to: {folder}")

    def _open_output_folder(self):
        """
        Open Output folder
        """
        folder_path = self.output_folder_input.text()
        if folder_path and Path(folder_path).exists():
            try:
                if os.name == 'nt':
                    os.startfile(folder_path)
                elif os.name == 'posix':
                    subprocess.run(['open', folder_path] if sys.platform == 'darwin'
                                  else ['xdg-open', folder_path], check=False)
            except Exception as e: # pylint: disable=broad-exception-caught
                self._log(f"Failed to open directory: {str(e)}")
        else:
            self._log(f"Error: No such directory: {folder_path}")

    def _start_process(self):
        """
        Analyse the image with parameters
        """
        if not self.data_manager.current_path:
            self._log("Error: Please load an OME-TIFF image.")
            return

        output_folder = self.output_folder_input.text()
        if not output_folder or not Path(output_folder).exists():
            self._log("Error: Please select a valid output directory.")
            return

        # Check current parameters
        current_file = self.data_manager.current_path

        check_list = []
        channel_params = {}
        for channel_index, channel_param in self.channel_params.items():
            channel_params[channel_index] = {}
            channel_param: dict
            for key, value in channel_param.items():
                if key not in ("lower_spin", "upper_spin"):
                    channel_params[channel_index][key] = value
                if key == "type":
                    check_list.append(value)
        if 1 not in check_list or 2 not in check_list:
            self._log("Error: Please mark channel types.")
            return

        # Disable start button
        self._set_processing_controls(False)

        # Create subprocess for analysing
        self.processing_thread = ProcessingThread(
            image_path=current_file,
            image_snapshot=self.rgb_image,
            channel_params=channel_params,
            output_folder=output_folder
        )

        # Subprocess can output log now
        self.processing_thread.log_signal.connect(self._log)
        self.processing_thread.finished_signal.connect(self._on_processing_finished)

        # Let's do it
        self.processing_thread.start()

    def _set_processing_controls(self, enabled):
        """
        Start Button enable & disable
        """
        self.start_process_btn.setEnabled(enabled)
        self.browse_output_btn.setEnabled(enabled)
        if enabled:
            self.start_process_btn.setText("▶ Start")
        else:
            self.start_process_btn.setText("Processing")

    def _on_processing_finished(self, success):
        """
        Call back when subprocess finished.
        """
        self._set_processing_controls(True)
        self._log(success)
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.wait()

        self.processing_thread = None

    def _log(self, message):
        """Log output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Automaticly scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
