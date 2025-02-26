import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

INPUT_FRAMES_DIR = "output/motion_vectors"
OUTPUT_MASKS_DIR = "output/segformer_masks"
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

def process_frame(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.argmax(dim=1)[0].cpu().numpy()

    mask = (logits == 12).astype(np.uint8) * 255
    cv2.imwrite(output_path, mask)

def main():
    frame_files = sorted([f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith(".png")])
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        process_frame(os.path.join(INPUT_FRAMES_DIR, frame_file), os.path.join(OUTPUT_MASKS_DIR, frame_file))

if __name__ == "__main__":
    main()
