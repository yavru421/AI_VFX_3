import os
import cv2
import numpy as np
from pathlib import Path

# Paths
OUTPUT_DIR = "output"
MOTION_VECTORS_DIR = os.path.join(OUTPUT_DIR, "motion_vectors")
REFINED_MASKS_DIR = os.path.join(OUTPUT_DIR, "refined_masks")
CUTOUTS_DIR = os.path.join(OUTPUT_DIR, "cutouts")
SUBTRACTED_DIR = os.path.join(OUTPUT_DIR, "subtracted")

def process_frame(frame_file, input_frame=None):
    """Process a single frame, subtracting masked areas."""
    refined_mask_path = os.path.join(REFINED_MASKS_DIR, frame_file)
    output_path = os.path.join(SUBTRACTED_DIR, f"subtracted_{frame_file}")

    # Load mask
    refined_mask = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)
    if refined_mask is None:
        print(f" Skipping {frame_file} - No mask found")
        return

    # Use provided input frame or load from cutouts
    if input_frame is None:
        cutout_path = os.path.join(CUTOUTS_DIR, frame_file)
        input_frame = cv2.imread(cutout_path, cv2.IMREAD_COLOR)
        if input_frame is None:
            print(f" Skipping {frame_file} - No input frame")
            return

    # Resize mask if needed
    if refined_mask.shape[:2] != input_frame.shape[:2]:
        refined_mask = cv2.resize(refined_mask, (input_frame.shape[1], input_frame.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    # Ensure mask is binary uint8
    _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Invert mask to keep non-masked areas
    refined_mask = cv2.bitwise_not(refined_mask)
    
    # Apply mask to input frame
    result = cv2.bitwise_and(input_frame, input_frame, mask=refined_mask)
    
    # Save result
    os.makedirs(SUBTRACTED_DIR, exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f" Saved subtracted frame: {output_path}")

def process_video(input_video=None):
    """Process entire video or directory of frames."""
    if input_video and os.path.exists(input_video):
        cap = cv2.VideoCapture(input_video)
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_file = f"frame_{frame_number:04d}.png"
            process_frame(frame_file, frame)
            frame_number += 1
            
        cap.release()
    else:
        frame_files = sorted([f for f in os.listdir(CUTOUTS_DIR) if f.endswith(('.png', '.jpg'))])
        for frame_file in frame_files:
            process_frame(frame_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Optional input video file')
    args = parser.parse_args()
    
    process_video(args.input if args else None)
