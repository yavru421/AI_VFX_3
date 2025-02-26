import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_DIR = "output/original_frames"
MASKS_DIR = "output/masks"
OUTPUT_DIR = "output/cutouts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_cutout(image_path, mask_path, output_path):
    """Create transparent cutout using mask"""
    # Load images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Error loading files for {image_path}")
        return False
        
    # Ensure mask matches image size
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    
    # Create RGBA image
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    
    # Save with transparency
    cv2.imwrite(output_path, rgba)
    return True

def main():
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
    
    for frame_file in tqdm(frame_files, desc="Creating cutouts"):
        image_path = os.path.join(INPUT_DIR, frame_file)
        mask_path = os.path.join(MASKS_DIR, frame_file)
        output_path = os.path.join(OUTPUT_DIR, frame_file)
        
        if os.path.exists(mask_path):
            create_cutout(image_path, mask_path, output_path)
        else:
            print(f"Missing mask for {frame_file}")

if __name__ == "__main__":
    main()
