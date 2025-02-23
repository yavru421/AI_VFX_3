import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import logging
import sys

#  Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#  Optimized GPU Settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#  Basic Configurations
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
OUTPUT_DIR = "output"
OUTPUT_MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

#  Streamlined GPU Memory Monitor
def get_gpu_memory():
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9
    }

logging.info(f"Using {torch.cuda.get_device_name()}")
logging.info(f"Initial GPU Memory: {get_gpu_memory()['allocated']:.2f}GB")

#  Load AI Model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model = model.to(device).half()
model.eval()

#  Enhanced Mask Extraction
def extract_foreground(logits, target_class=12, confidence_threshold=0.7):
    probs = torch.softmax(logits, dim=1)
    confidence_map = probs[:, target_class, :, :]
    seg_map = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    confidence_mask = (confidence_map > confidence_threshold).float().cpu().numpy()
    binary_mask = np.where(seg_map == target_class, 255, 0).astype(np.uint8)
    refined_mask = (binary_mask * confidence_mask[0]).astype(np.uint8)
    return refined_mask

#  Process Frame

def process_frame(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        logging.error(f"Failed to read image: {image_path}")
        return False

    original_size = (img.shape[1], img.shape[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.cuda.amp.autocast():
        inputs = processor(images=img_rgb, return_tensors="pt", do_resize=True, size={'height': 512, 'width': 512})
        inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

    mask = extract_foreground(outputs.logits)
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, mask_resized)
    return True

#  Main Processing Function
def main(image_folder):
    if not os.path.exists(image_folder):
        logging.error("❌ Input folder not found. Run motion vector extraction first.")
        return
    if not any(f.endswith('.png') for f in os.listdir(image_folder)):
        logging.error("❌ No PNG files found in input folder.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    
    # Ensure we're reading from the correct folder
    if not os.path.exists(image_folder):
        logging.error(f"Input folder not found: {image_folder}")
        return

    frame_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    logging.info(f"Found {len(frame_files)} frames to process")

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        image_path = os.path.join(image_folder, frame_file)
        output_path = os.path.join(OUTPUT_MASKS_DIR, frame_file)
        if process_frame(image_path, output_path):
            logging.debug(f"Processed {frame_file}")
        else:
            logging.error(f"Failed to process {frame_file}")

    logging.info(f"AI Processing Completed! Results saved in {OUTPUT_MASKS_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_processing.py <image_folder>")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    main(image_folder)
