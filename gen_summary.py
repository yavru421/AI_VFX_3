import os
import glob
import cv2
from fpdf import FPDF
from datetime import datetime

# === Paths ===
OUTPUT_DIR = os.path.join("output")
SUMMARY_PDF = "pipeline_summary.pdf"
SAMPLES = 2  # Number of frames to analyze per folder

# === Subfolders to process ===
PIPELINE_FOLDERS = [
    "masks",
    "motion_vectors",
    "refined_masks",
    "segformer_mask"
]

def get_all_subfolders(directory):
    """Retrieve all subdirectories inside a given directory."""
    return sorted([f.path for f in os.scandir(directory) if f.is_dir()])

def get_sample_frames(folder, num_samples=2):
    """Get a few sample images from a given folder."""
    # Check if folder name matches any of the pipeline folders
    folder_name = os.path.basename(folder)
    if folder_name not in PIPELINE_FOLDERS:
        return []
    
    # Get all PNG images from the folder
    image_files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return image_files[:num_samples] if image_files else []

def generate_pdf():
    """Creates a PDF showing image progression across pipeline steps."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI VFX Pipeline Summary", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # Find all processing folders in the output directory
    pipeline_steps = get_all_subfolders(OUTPUT_DIR)

    if not pipeline_steps:
        print("No processing folders found in output/.")
        return

    # Collect sample frames for each step
    frame_sets = {step: get_sample_frames(step, SAMPLES) for step in pipeline_steps}

    # Find the minimum number of samples across steps to ensure alignment
    num_frames = min(len(fs) for fs in frame_sets.values() if fs) if frame_sets else 0

    if num_frames == 0:
        print("No valid images found in output directories.")
        return

    for i in range(num_frames):
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Frame {i+1} Processed Through Pipeline", ln=True)

        for step, step_images in frame_sets.items():
            if i >= len(step_images):
                continue  # Skip if there aren't enough images

            img_path = step_images[i]
            pdf.ln(3)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 8, f"{os.path.basename(step).replace('_', ' ').capitalize()}", ln=True)

            # Convert image for PDF
            temp_image = "temp.jpg"
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(temp_image, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                pdf.image(temp_image, x=10, w=90)  # Add image (resized)

    # Save PDF
    pdf.output(SUMMARY_PDF, "F")
    print(f" Pipeline summary saved: {SUMMARY_PDF}")

if __name__ == "__main__":
    generate_pdf()
