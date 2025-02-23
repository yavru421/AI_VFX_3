import os
import cv2
import numpy as np
from tqdm import tqdm

# Define centralized output folder
OUTPUT_DIR = "output"
MOTION_VECTORS_DIR = os.path.join(OUTPUT_DIR, "motion_vectors")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
REFINED_MASKS_DIR = os.path.join(OUTPUT_DIR, "refined_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFINED_MASKS_DIR, exist_ok=True)

# Optimized Mask Refinement Function
def refine_masks():
    if not os.path.exists(MOTION_VECTORS_DIR) or not os.path.exists(MASKS_DIR):
        print("❌ Required input folders not found. Run previous steps first.")
        return
    if not any(f.endswith('.png') for f in os.listdir(MASKS_DIR)):
        print("❌ No masks found in input folder.")
        return

    print(f"Reading masks from: {MASKS_DIR}")
    print(f"Reading motion vectors from: {MOTION_VECTORS_DIR}")
    print(f"Saving refined masks to: {REFINED_MASKS_DIR}")
    
    frame_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith(".png")])
    
    for frame_file in tqdm(frame_files, desc="Refining masks"):
        mask_path = os.path.join(MASKS_DIR, frame_file)
        motion_path = os.path.join(MOTION_VECTORS_DIR, frame_file)
        output_path = os.path.join(REFINED_MASKS_DIR, frame_file)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        motion_vector = cv2.imread(motion_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or motion_vector is None:
            print(f"Skipping {frame_file}, missing data.")
            continue
        
        # Adaptive Thresholding for better mask refinement
        mask_thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        motion_thresh = cv2.adaptiveThreshold(motion_vector, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine AI segmentation & motion vectors to refine masks
        refined_mask = cv2.bitwise_and(mask_thresh, motion_thresh)
        
        # Apply Gaussian Blur for smoother edges
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        
        # Morphological operations to refine edges
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.erode(refined_mask, kernel, iterations=1)
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)
        
        # Save refined mask
        cv2.imwrite(output_path, refined_mask)
    
    print("✅ Mask Refinement Completed!")

if __name__ == "__main__":
    refine_masks()
