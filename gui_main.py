import sys
import os
import json
import subprocess
import psutil
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget, QPushButton, 
    QProgressBar, QTextEdit, QLabel, QTabWidget, QSplitter, QTreeView, QFileDialog,
    QMenuBar, QToolBar, QStatusBar, QDockWidget, QGraphicsView, QGraphicsScene, 
    QCheckBox, QRadioButton, QSpinBox, QProgressDialog, QHBoxLayout, QListWidgetItem, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QAction

CONFIG_FILE = "config.json"

class ProcessingThread(QThread):
    """ Runs the processing pipeline in a separate thread. """
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, files, steps):
        super().__init__()
        self.files = files
        self.steps = steps

    def run(self):
        total_steps = len(self.files) * len(self.steps)
        progress = 0

        for file in self.files:
            for step in self.steps:
                command = step["command"].format(input=file, output="output/")
                self.log_signal.emit(f"Running: {command}")

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    self.log_signal.emit(line.strip())
                process.wait()

                progress += 1
                self.progress_signal.emit(int(progress / total_steps * 100))

        self.finished_signal.emit()

class AI_VFX_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load Config
        with open(CONFIG_FILE, 'r') as f:
            self.config = json.load(f)

        self.setWindowTitle("AI VFX Pipeline")
        self.setGeometry(100, 100, 1200, 800)

        self.init_ui()

    def init_ui(self):
        """ Set up UI components. """
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create Tabs
        self.setup_file_selection_tab()
        self.setup_processing_and_logs_tab()  # Changed to combined tab
        self.setup_cleaning_tab()

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Menu Bar
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Open Files", self)
        open_action.triggered.connect(self.load_files)
        file_menu.addAction(open_action)

        # Tool Bar
        self.tool_bar = QToolBar("Main Toolbar")
        self.addToolBar(self.tool_bar)
        self.tool_bar.addAction(open_action)

    def setup_file_selection_tab(self):
        """ Create File Selection UI without QFileSystemModel """
        tab = QWidget()
        layout = QVBoxLayout()

        # File List
        self.file_list = QListWidget()
        layout.addWidget(QLabel("Selected Files:"))
        layout.addWidget(self.file_list)

        # Load Files Button
        self.load_button = QPushButton("Load Files")
        self.load_button.clicked.connect(self.load_files)
        layout.addWidget(self.load_button)

        # Folder Browser (Using QFileDialog Instead of QFileSystemModel)
        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.clicked.connect(self.open_folder_dialog)
        layout.addWidget(QLabel("Select a Folder:"))
        layout.addWidget(self.select_folder_button)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "File Selection")

    def setup_processing_and_logs_tab(self):
        """Create Processing and Logs UI side by side"""
        tab = QWidget()
        layout = QHBoxLayout()  # Horizontal layout for side-by-side

        # Left side - Processing
        processing_widget = QWidget()
        processing_layout = QVBoxLayout()

        # Pipeline Steps Selection
        processing_layout.addWidget(QLabel("Pipeline Steps:"))
        self.step_selector = QListWidget()
        self.step_selector.addItems([step["name"] for step in self.config["steps"]])
        self.step_selector.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        processing_layout.addWidget(self.step_selector)

        # Control Buttons
        button_layout = QHBoxLayout()
        self.run_step_button = QPushButton("Run Selected Step")
        self.run_step_button.clicked.connect(self.run_selected_step)
        self.run_all_button = QPushButton("Run Full Pipeline")
        self.run_all_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.run_all_button)
        
        processing_layout.addLayout(button_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        processing_layout.addWidget(self.progress_bar)
        
        processing_widget.setLayout(processing_layout)

        # Right side - Logs
        logs_widget = QWidget()
        logs_layout = QVBoxLayout()

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        logs_layout.addWidget(QLabel("Processing Logs:"))
        logs_layout.addWidget(self.log_output)

        # System Monitor
        self.system_monitor = QLabel("CPU: 0% | RAM: 0%")
        logs_layout.addWidget(self.system_monitor)
        
        logs_widget.setLayout(logs_layout)

        # Add splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(processing_widget)
        splitter.addWidget(logs_widget)
        splitter.setSizes([400, 400])  # Equal initial widths

        layout.addWidget(splitter)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Processing & Logs")

    def setup_cleaning_tab(self):
        """Create Cleaning UI with folder buttons"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add scroll area for folder buttons
        self.folder_buttons_layout = QVBoxLayout()
        folder_scroll = QWidget()
        folder_scroll.setLayout(self.folder_buttons_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(folder_scroll)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.refresh_dirs_button = QPushButton("Refresh Directories")
        self.refresh_dirs_button.clicked.connect(self.refresh_directories)
        controls_layout.addWidget(self.refresh_dirs_button)

        self.clean_all_button = QPushButton("Clean All Selected")
        self.clean_all_button.clicked.connect(self.clean_selected_directories)
        controls_layout.addWidget(self.clean_all_button)
        
        layout.addLayout(controls_layout)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Cleanup")

        # Initial directory load
        self.refresh_directories()

    def refresh_directories(self):
        """Refresh the list of available directories with buttons"""
        # Clear existing buttons
        for i in reversed(range(self.folder_buttons_layout.count())): 
            self.folder_buttons_layout.itemAt(i).widget().setParent(None)

        # Add new buttons for each directory
        for root, dirs, files in os.walk("output"):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if any(os.path.isfile(os.path.join(dir_path, f)) for f in os.listdir(dir_path)):
                    button = QPushButton(dir_path)
                    button.setCheckable(True)
                    self.folder_buttons_layout.addWidget(button)

    def clean_selected_directories(self):
        """Clean files from selected directory buttons"""
        selected_dirs = []
        for i in range(self.folder_buttons_layout.count()):
            button = self.folder_buttons_layout.itemAt(i).widget()
            if button.isChecked():
                selected_dirs.append(button.text())

        if not selected_dirs:
            QMessageBox.warning(self, "No Selection", "Please select directories to clean.")
            return

        # Confirm deletion
        dirs_text = "\n".join(selected_dirs)
        reply = QMessageBox.question(self, 'Confirm Cleanup',
                                   f'Are you sure you want to delete all files in:\n\n{dirs_text}',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            for dir_path in selected_dirs:
                try:
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    self.log_output.append(f"Cleaned: {dir_path}")
                except Exception as e:
                    self.log_output.append(f"Error cleaning {dir_path}: {str(e)}")
            
            QMessageBox.information(self, "Cleanup Complete", "Selected directories have been cleaned!")
            self.refresh_directories()

    def run_selected_step(self):
        """Run only the selected pipeline step"""
        if not self.file_list.count():
            self.log_output.append("Error: No input files selected")
            return

        selected_items = self.step_selector.selectedItems()
        if not selected_items:
            self.log_output.append("Error: No pipeline step selected")
            return

        step_name = selected_items[0].text()
        step = next((s for s in self.config["steps"] if s["name"] == step_name), None)
        
        if step:
            files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
            self.thread = ProcessingThread([files[0]], [step])  # Process first file
            self.thread.progress_signal.connect(self.progress_bar.setValue)
            self.thread.log_signal.connect(self.log_output.append)
            self.thread.finished_signal.connect(
                lambda: self.status_bar.showMessage(f"Step '{step_name}' Completed!", 5000)
            )
            self.thread.start()
            self.log_output.append(f"Running step: {step_name}")

    def load_files(self):
        """ Load files via QFileDialog """
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "Videos (*.mp4 *.mov);;Images (*.png *.jpg *.tif)")
        if files:
            self.file_list.addItems(files)

    def open_folder_dialog(self):
        """ Opens a folder selection dialog """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            selected_folder = file_dialog.selectedFiles()[0]
            print("Selected Folder:", selected_folder)

    def start_processing(self):
        """ Start processing pipeline """
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        steps = self.config["steps"]

        self.thread = ProcessingThread(files, steps)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(self.log_output.append)
        self.thread.finished_signal.connect(lambda: self.status_bar.showMessage("Processing Completed!", 5000))

        self.thread.start()

    def update_system_monitor(self):
        """ Update system resource usage display """
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        self.system_monitor.setText(f"CPU: {cpu_usage}% | RAM: {ram_usage}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AI_VFX_GUI()
    window.show()
    sys.exit(app.exec())
