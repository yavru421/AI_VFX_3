import os
import json
import glob
from jinja2 import Template
from fpdf import FPDF
import cv2

# Load configuration
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as file:
    config = json.load(file)

OUTPUT_DIR = config["output_dir"]
REPORT_FILE_HTML = "pipeline_report.html"
REPORT_FILE_PDF = "pipeline_report.pdf"

# Collect first image from each output directory
expected_folders = [
    "cutouts",
    "extract_motion_vectors",
    "generate_transparent_pngs",
    "masks",
    "motion_vectors",
    "refine_masks",
    "refined_masks",
    "run_ai_processing",
    "run_segformer",
    "segformer_masks"
]

images = []
for folder in expected_folders:
    folder_path = os.path.join(OUTPUT_DIR, folder)
    if os.path.exists(folder_path):
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        if image_files:
            images.append({"name": folder.replace("_", " ").title(), "path": image_files[0]})

# HTML Report Template
html_template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI VFX Pipeline Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: auto; }
        .step { margin-bottom: 20px; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI VFX Pipeline Report</h1>
        {% for image in images %}
        <div class="step">
            <h2>{{ image.name }}</h2>
            <img src="{{ image.path }}" alt="{{ image.name }}">
        </div>
        {% endfor %}
    </div>
</body>
</html>
''')

# Generate HTML Report
html_content = html_template.render(images=images)
with open(REPORT_FILE_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

# Generate PDF Report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "AI VFX Pipeline Report", ln=True, align="C")
pdf.ln(10)

for image in images:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, image["name"], ln=True)
    pdf.ln(5)
    img = cv2.imread(image["path"])
    if img is not None:
        temp_image = "temp.jpg"
        cv2.imwrite(temp_image, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        pdf.image(temp_image, x=10, w=100)
    pdf.ln(10)

pdf.output(REPORT_FILE_PDF)

print(f" Pipeline reports generated: {REPORT_FILE_HTML}, {REPORT_FILE_PDF}")