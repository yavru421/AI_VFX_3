import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

# Paths
INPUT_DIR = "output/original_frames"
OUTPUT_DIR = "output/masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_frame(image_path, output_path):
    """Generate person mask for a single frame"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return False
    
    # Prepare image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate mask
    with torch.no_grad():
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        outputs = model(**inputs)
        mask = (outputs.logits[0].argmax(dim=0) == 12).cpu().numpy().astype(np.uint8) * 255
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Save mask
    cv2.imwrite(output_path, mask)
    return True

def main():
    """Process all frames in the input directory"""
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg'))])
    
    if not frame_files:
        print("No frames found in input directory!")
        return

    for frame_file in tqdm(frame_files, desc="Generating masks"):
        input_path = os.path.join(INPUT_DIR, frame_file)
        output_path = os.path.join(OUTPUT_DIR, frame_file)
        process_frame(input_path, output_path)

if __name__ == "__main__":
    main()
