import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval().half()

INPUT_DIR = "output/final_transparent"
OUTPUT_DIR = "output/segformer_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4  # Adjust based on available VRAM

def process_batch(image_paths, output_paths):
    try:
        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read image: {path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

        if not images:
            return False

        # Run SegFormer with proper amp context
        with torch.cuda.amp.autocast(dtype=torch.float16):
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits

        # Get probability masks
        probs = torch.nn.functional.softmax(logits, dim=1)
        refined_masks = probs.max(dim=1).values
        refined_masks = [(mask.cpu().numpy() * 255).astype(np.uint8) for mask in refined_masks]

        # Save Final Refinement Masks
        for refined_mask, output_path in zip(refined_masks, output_paths):
            cv2.imwrite(output_path, refined_mask)

        return True

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return False

def main():
    frame_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
    
    # Process in Batches
    for i in tqdm(range(0, len(frame_files), BATCH_SIZE), desc="Final SegFormer Pass"):
        batch_files = frame_files[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(INPUT_DIR, f) for f in batch_files]
        output_paths = [os.path.join(OUTPUT_DIR, f) for f in batch_files]

        process_batch(batch_paths, output_paths)

if __name__ == "__main__":
    main()
