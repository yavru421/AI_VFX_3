import os
import cv2
import numpy as np
from tqdm import tqdm

# Define the output structure
OUTPUT_DIR = "output"
MOTION_VECTORS_DIR = os.path.join(OUTPUT_DIR, "motion_vectors")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
REFINED_MASKS_DIR = os.path.join(OUTPUT_DIR, "refined_masks")
SEGFORMER_MASKS_DIR = os.path.join(OUTPUT_DIR, "segformer_masks")
CUTOUTS_DIR = os.path.join(OUTPUT_DIR, "cutouts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CUTOUTS_DIR, exist_ok=True)

def apply_alpha_blending(image, mask):
    """
    Applies an alpha mask to the original image for transparency support.
    """
    if image is None or mask is None:
        print("Error: One or more input images are invalid.")
        return None
    if len(image.shape) < 3 or image.shape[2] != 3:
        print("Error: Expected a 3-channel image but received something else.")
        return None
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(image)
    alpha = mask.astype(np.uint8)
    return cv2.merge([b, g, r, alpha])

def refine_mask(mask):
    """
    Applies morphological operations to refine mask edges.
    """
    if mask is None:
        return None
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def process_image(image_path, mask_path, output_path, background_path=None):
    """
    Processes an image by applying its segmentation mask and optionally replacing the background.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print(f"Error loading {image_path} or {mask_path}")
        return
    
    mask = refine_mask(mask)
    if mask is None:
        print(f"Skipping {image_path} due to invalid mask.")
        return
    
    cutout = apply_alpha_blending(image, mask)
    if cutout is None:
        print(f"Skipping {image_path} due to invalid cutout creation.")
        return
    
    cv2.imwrite(output_path, cutout, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    if background_path:
        background = cv2.imread(background_path)
        if background is None:
            print(f"Error loading background image: {background_path}")
            return
        background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
        mask_inv = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(image, image, mask=mask)
        bg = cv2.bitwise_and(background_resized, background_resized, mask=mask_inv)
        combined = cv2.add(fg, bg)
        cv2.imwrite(output_path.replace(".png", "_bg.png"), combined)

def main():
    """
    Processes all images in the pipeline's output directories and applies transparency & background replacement.
    """
    input_images_dir = REFINED_MASKS_DIR
    input_masks_dir = MASKS_DIR
    output_dir = CUTOUTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_images_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(input_masks_dir) if f.endswith(".png")])

    if len(image_files) == 0 or len(mask_files) == 0:
        print("Error: No valid image or mask files found in the provided directories.")
        return

    if len(image_files) != len(mask_files):
        print("Error: Mismatch between number of images and masks.")
        return

    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing images"):
        img_path = os.path.join(input_images_dir, img_file)
        mask_path = os.path.join(input_masks_dir, mask_file)
        output_path = os.path.join(output_dir, img_file)
        process_image(img_path, mask_path, output_path)

if __name__ == "__main__":
    main()
