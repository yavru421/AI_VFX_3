import sys
import subprocess
import os
import json
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QProgressBar, QVBoxLayout, QWidget, QTextEdit, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class PathManager:
    def __init__(self, base_dir, resolution="1920x1080"):
        self.base_dir = Path(base_dir).absolute()
        self.resolution = resolution  # Store resolution

    def set_resolution(self, resolution):
        self.resolution = resolution  # Update resolution dynamically

    def get_step_dir(self, step_name):
        safe_name = step_name.lower().replace(' ', '_')
        return self.base_dir / safe_name

    def format_command(self, command, input_file=None):
        output_path = str(self.base_dir)
        if input_file:
            input_path = str(Path(input_file).absolute())
        else:
            input_path = ''

        return command.format(input=input_path, output=output_path)

class PipelineThread(QThread):
    log_signal = pyqtSignal(str)
    step_complete_signal = pyqtSignal()

    def __init__(self, step, path_manager, input_file):
        super().__init__()
        self.step = step
        self.path_manager = path_manager
        self.input_file = input_file

    def run(self):
        command = self.path_manager.format_command(self.step["command"], self.input_file)
        self.log_signal.emit(f"Running command: {command}")

        process = subprocess.Popen(command, shell=True, cwd=os.path.abspath("."), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        process.wait()

        self.log_signal.emit(f"{self.step['name']} completed!")
        self.step_complete_signal.emit()

class AI_VFX_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        with open('config.json', 'r') as f:
            self.config = json.load(f)

        self.path_manager = PathManager(self.config['output_dir'])
        self.video_path = None  # No video selected initially

        self.setWindowTitle("AI VFX Pipeline")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        # Resolution Selection
        self.resolution_label = QLabel("Select Output Resolution:")
        self.layout.addWidget(self.resolution_label)

        self.resolution_dropdown = QComboBox()
        self.resolution_dropdown.addItems(["1920x1080", "1080x1920", "1280x720", "640x480"])
        self.resolution_dropdown.setCurrentText("1920x1080")
        self.layout.addWidget(self.resolution_dropdown)

        # Video Selection
        self.video_label = QLabel("Select a Video File:")
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        # Buttons for Pipeline Steps
        self.buttons = []
        self.threads = []

        for i, step in enumerate(self.config["steps"]):
            button = QPushButton(step["name"])
            button.setEnabled(False)
            button.clicked.connect(lambda _, s=step, idx=i: self.run_step(s, idx))
            self.layout.addWidget(button)
            self.buttons.append(button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_label.setText(f"Selected Video: {file_path}")
            self.video_path = file_path

            # Enable first button only
            self.buttons[0].setEnabled(True)

    def run_step(self, step, index):
        self.log_output.append(f"Starting {step['name']}...")

        command = self.path_manager.format_command(step["command"], self.video_path)

        self.log_output.append(f"Running command: {command}")
        thread = PipelineThread(step, self.path_manager, self.video_path)
        thread.log_signal.connect(self.log_output.append)
        thread.step_complete_signal.connect(lambda: self.enable_next_step(index))
        thread.start()
        self.threads.append(thread)

    def enable_next_step(self, current_index):
        """ Enables the next step only when the current step is completed """
        if current_index + 1 < len(self.buttons):
            self.buttons[current_index + 1].setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = AI_VFX_GUI()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
