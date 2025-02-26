import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

# Paths
SEGFORMER_MASKS_DIR = "output/segformer_masks"  
MOTION_VECTORS_DIR = "output/motion_vectors"
OUTPUT_MASKS_DIR = "output/masks"
DEBUG_DIR = "output/debug"

os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def process_frame(segformer_mask_path, motion_vector_path, output_path):
    logging.info(f"Processing: {segformer_mask_path}")

    # Load SegFormer mask
    mask = cv2.imread(segformer_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logging.error(f"❌ Failed to read SegFormer mask: {segformer_mask_path}")
        return False

    # Load motion vector, check if it exists
    if os.path.exists(motion_vector_path):
        motion_vector = cv2.imread(motion_vector_path, cv2.IMREAD_GRAYSCALE)
    else:
        logging.warning(f"⚠️ Motion vector missing for {motion_vector_path}. Using fallback.")
        motion_vector = np.ones_like(mask) * 255  # Fallback to all white

    # Ensure size consistency by resizing **mask** to motion vector size
    if motion_vector.shape != mask.shape:
        logging.warning(f"Resizing AI mask from {mask.shape} to match motion vector size {motion_vector.shape}")
        mask = cv2.resize(mask, (motion_vector.shape[1], motion_vector.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure both are uint8
    mask = mask.astype(np.uint8)
    motion_vector = motion_vector.astype(np.uint8)

    # Apply refinement
    refined_mask = cv2.bitwise_and(mask, motion_vector)

    # Save debug output
    debug_path = output_path.replace(".png", "_debug.png")
    cv2.imwrite(debug_path, refined_mask)
    cv2.imwrite(output_path, refined_mask)

    return True

def main():
    frame_files = sorted([f for f in os.listdir(SEGFORMER_MASKS_DIR) if f.endswith(".png")])
    if not frame_files:
        logging.error("❌ No SegFormer masks found. Ensure SegFormer step ran first.")
        return

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        segformer_mask_path = os.path.join(SEGFORMER_MASKS_DIR, frame_file)
        motion_vector_path = os.path.join(MOTION_VECTORS_DIR, frame_file)
        output_path = os.path.join(OUTPUT_MASKS_DIR, frame_file)

        process_frame(segformer_mask_path, motion_vector_path, output_path)

    logging.info("✅ AI Processing Completed! Masks saved in output/masks.")

if __name__ == "__main__":
    main()
