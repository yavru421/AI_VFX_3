import sys
import subprocess
import os
import json
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QProgressBar, QVBoxLayout, QWidget, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal

class PathManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).absolute()
    
    def get_step_dir(self, step_name):
        safe_name = step_name.lower().replace(' ', '_')
        return self.base_dir / safe_name
    
    def format_command(self, command, input_file=None):
        output_path = str(self.base_dir)
        if input_file:
            input_path = str(Path(input_file).absolute())
        else:
            input_path = ''
            
        # Get the current script directory
        script_dir = str(Path(__file__).parent.absolute())
        
        # If it's a Python script, make sure we use the full path
        if command.startswith('python'):
            command = command.replace('python ', f'python "{script_dir}/')
            command = command.replace('.py', '.py"')
            
        return command.format(
            input=input_path,
            output=output_path
        )

class PipelineThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, step, path_manager, input_file):
        super().__init__()
        self.step = step
        self.path_manager = path_manager
        self.input_file = input_file

    def run(self):
        command = self.path_manager.format_command(self.step["command"], self.input_file)
        self.log_signal.emit(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        process.wait()
        self.log_signal.emit(f"{self.step['name']} completed!")

class AI_VFX_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        self.path_manager = PathManager(self.config['output_dir'])
        self.setup_directories()
        
        self.setWindowTitle("AI VFX Pipeline")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        self.video_label = QLabel("Select a Video File:")
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        self.buttons = []
        self.threads = []

        for step in self.config["steps"]:
            button = QPushButton(step["name"])
            button.setEnabled(False)
            button.clicked.connect(lambda _, s=step: self.run_step(s))
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

    def setup_directories(self):
        os.makedirs(self.path_manager.base_dir, exist_ok=True)
        for step in self.config["steps"]:
            subdir = self.path_manager.get_step_dir(step["name"])
            os.makedirs(subdir, exist_ok=True)
            if "motion_vectors" in step["command"]:
                os.makedirs(self.path_manager.base_dir / "motion_vectors", exist_ok=True)
            if "masks" in step["command"]:
                os.makedirs(self.path_manager.base_dir / "masks", exist_ok=True)

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_label.setText(f"Selected Video: {file_path}")
            self.video_path = file_path
            for button in self.buttons:
                button.setEnabled(True)

    def run_step(self, step):
        self.log_output.append(f"Starting {step['name']}...")
        thread = PipelineThread(step, self.path_manager, self.video_path)
        thread.log_signal.connect(self.log_output.append)
        thread.start()
        self.threads.append(thread)

def main():
    app = QApplication(sys.argv)
    window = AI_VFX_GUI()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
