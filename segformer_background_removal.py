import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Switch to SegFormer model specifically trained for person segmentation
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

# Paths
INPUT_FRAMES_DIR = "output/refined_masks"
OUTPUT_MASKS_DIR = "output/segformer_masks"
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

def extract_foreground(logits, height, width):
    """Extract and refine person segmentation mask."""
    # Get prediction
    pred = logits.argmax(dim=1)[0]
    mask = (pred == 12).float().cpu().numpy()  # 12 is typically the person class in ADE20K
    
    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)
    
    # Resize to original dimensions
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Threshold to get binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def process_frame(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return False

    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image with SegFormer
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask = extract_foreground(logits, height, width)
    cv2.imwrite(output_path, mask)
    return True

def main():
    frame_files = sorted([f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith(".png")])
    print(f"Processing {len(frame_files)} frames with SegFormer Person Segmentation...")

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        image_path = os.path.join(INPUT_FRAMES_DIR, frame_file)
        output_path = os.path.join(OUTPUT_MASKS_DIR, frame_file)
        process_frame(image_path, output_path)

    print(f"✅ Background removal completed. Masks saved in {OUTPUT_MASKS_DIR}")

if __name__ == "__main__":
    main()
