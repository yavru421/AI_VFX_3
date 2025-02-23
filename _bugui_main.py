import sys
import subprocess
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QProgressBar, QVBoxLayout, QWidget, QTextEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Create output directory at startup
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MotionVectorThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        output_folder = os.path.join("output", "motion_vectors")
        os.makedirs(output_folder, exist_ok=True)
        output_sequence = os.path.join(output_folder, "frame_%04d.png")
        ffmpeg_command = [
            "ffmpeg", "-flags2", "+export_mvs", "-i", self.video_path,
            "-vf", "codecview=mv=pf+bf+bb", "-q:v", "2", output_sequence
        ]

        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        process.wait()
        self.log_signal.emit("Motion vector extraction completed! Frames saved as image sequence.")

class AIProcessingThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, image_folder):
        super().__init__()
        self.image_folder = os.path.join("output", "motion_vectors")  # Changed this line

    def run(self):
        ai_script = os.path.join(os.path.dirname(__file__), "ai_processing.py")
        process = subprocess.Popen(["python", ai_script, self.image_folder], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        process.wait()
        self.log_signal.emit("AI Processing completed! Masks and tracking data saved.")

class MaskRefinementThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, motion_vector_dir, mask_dir):
        super().__init__()
        self.motion_vector_dir = os.path.join("output", "motion_vectors")
        self.mask_dir = os.path.join("output", "masks")

    def run(self):
        refine_script = os.path.join(os.path.dirname(__file__), "refine_masks.py")
        process = subprocess.Popen(["python", refine_script, self.motion_vector_dir, self.mask_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        for line in process.stdout:
            self.log_signal.emit(line.strip())
        process.wait()
        self.log_signal.emit("Mask refinement completed!")

class AI_VFX_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI VFX Pipeline")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.layout = QVBoxLayout()

        # Video selection
        self.video_label = QLabel("Select a Video File:")
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        # Motion vector extraction button
        self.motion_button = QPushButton("Extract Motion Vectors")
        self.motion_button.setEnabled(False)
        self.motion_button.clicked.connect(self.extract_motion_vectors)
        self.layout.addWidget(self.motion_button)

        # AI Processing button
        self.ai_button = QPushButton("Run AI Processing")
        self.ai_button.setEnabled(False)
        self.ai_button.clicked.connect(self.run_ai_processing)
        self.layout.addWidget(self.ai_button)

        # Mask Refinement button
        self.refine_button = QPushButton("Refine Masks")
        self.refine_button.setEnabled(False)
        self.refine_button.clicked.connect(self.run_mask_refinement)
        self.layout.addWidget(self.refine_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)

        # Set up central widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def load_video(self):
        """Load a video file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_label.setText(f"Selected Video: {file_path}")
            self.video_path = file_path
            self.motion_button.setEnabled(True)

    def extract_motion_vectors(self):
        """Extract motion vectors using FFmpeg."""
        self.motion_button.setEnabled(False)
        self.log_output.append("Starting motion vector extraction...")
        self.worker = MotionVectorThread(self.video_path)
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.start()
        self.worker.finished.connect(lambda: self.ai_button.setEnabled(True))

    def run_ai_processing(self):
        """Run AI processing on extracted motion vectors."""
        self.ai_button.setEnabled(False)
        self.log_output.append("Starting AI processing...")
        motion_vector_folder = os.path.join("output", "motion_vectors")  # Changed this line
        self.ai_worker = AIProcessingThread(motion_vector_folder)
        self.ai_worker.log_signal.connect(self.log_output.append)
        self.ai_worker.start()
        self.ai_worker.finished.connect(lambda: self.refine_button.setEnabled(True))

    def run_mask_refinement(self):
        """Run mask refinement process."""
        self.refine_button.setEnabled(False)
        self.log_output.append("Starting mask refinement...")
        motion_vector_folder = os.path.join("output", "motion_vectors")
        mask_output_folder = os.path.join("output", "masks")
        self.refine_worker = MaskRefinementThread(motion_vector_folder, mask_output_folder)
        self.refine_worker.log_signal.connect(self.log_output.append)
        self.refine_worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AI_VFX_GUI()
    window.show()
    sys.exit(app.exec())
