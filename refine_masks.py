import os
import cv2
import numpy as np
from tqdm import tqdm

# Define directories
OUTPUT_DIR = "output"
MOTION_VECTORS_DIR = os.path.join(OUTPUT_DIR, "motion_vectors")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
REFINED_MASKS_DIR = os.path.join(OUTPUT_DIR, "refined_masks")

os.makedirs(REFINED_MASKS_DIR, exist_ok=True)

def refine_mask(mask_path, motion_vector_path, output_path):
    """Refines AI segmentation masks using motion vectors."""
    if not os.path.exists(mask_path) or not os.path.exists(motion_vector_path):
        print(f" Skipping {mask_path} - Missing mask or motion vector.")
        return  # Skip missing files

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    motion_vector = cv2.imread(motion_vector_path, cv2.IMREAD_GRAYSCALE)

    if mask is None or motion_vector is None:
        print(f" Error: Could not read {mask_path} or {motion_vector_path}. Skipping...")
        return

    # Ensure mask and motion vector are the same size
    if mask.shape != motion_vector.shape:
        motion_vector = cv2.resize(motion_vector, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply refinement
    refined_mask = cv2.bitwise_and(mask, motion_vector)

    # Save the refined mask
    cv2.imwrite(output_path, refined_mask)

def refine_masks():
    """Processes all frames."""
    frame_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith(".png")])

    if not frame_files:
        print(" No AI masks found. Ensure AI Processing completed first.")
        return

    for frame_file in tqdm(frame_files, desc="Refining masks"):
        mask_path = os.path.join(MASKS_DIR, frame_file)
        motion_vector_path = os.path.join(MOTION_VECTORS_DIR, frame_file)
        output_path = os.path.join(REFINED_MASKS_DIR, frame_file)

        refine_mask(mask_path, motion_vector_path, output_path)

    print(" Mask Refinement Completed!")

if __name__ == "__main__":
    refine_masks()
