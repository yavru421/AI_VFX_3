import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "output/cutouts"
OUTPUT_DIR = "output/final_cutouts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def refine_edges(image):
    """Refine edges of RGBA image using edge detection"""
    # Split alpha channel
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Edge detection on RGB channels
    edges_bgr = cv2.Canny(bgr, 100, 200)
    
    # Edge detection on alpha channel
    edges_alpha = cv2.Canny(alpha, 50, 150)
    
    # Combine edges
    combined_edges = cv2.bitwise_or(edges_bgr, edges_alpha)
    
    # Dilate edges slightly
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # Create mask for edge feathering
    feather_mask = cv2.GaussianBlur(dilated_edges.astype(float), (5,5), 0)
    feather_mask = feather_mask / feather_mask.max()
    
    # Apply feathering to alpha channel
    refined_alpha = cv2.addWeighted(
        alpha.astype(float), 1.0,
        feather_mask * alpha.astype(float), -0.3,
        0
    ).clip(0, 255).astype(np.uint8)
    
    # Reconstruct image
    refined_image = image.copy()
    refined_image[:, :, 3] = refined_alpha
    
    return refined_image

def process_frame(input_path, output_path):
    """Process a single frame"""
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:  # Ensure RGBA
        print(f"Error: {input_path} is not a valid RGBA image")
        return False
        
    refined = refine_edges(image)
    cv2.imwrite(output_path, refined)
    return True

def main():
    """Process all cutouts"""
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    
    for frame_file in tqdm(frame_files, desc="Refining edges"):
        input_path = os.path.join(INPUT_DIR, frame_file)
        output_path = os.path.join(OUTPUT_DIR, frame_file)
        process_frame(input_path, output_path)

if __name__ == "__main__":
    main()
