import os
import cv2
import numpy as np
from tqdm import tqdm

# Define directories
OUTPUT_DIR = "output"
REFINED_MASKS_DIR = os.path.join(OUTPUT_DIR, "refined_masks")
ORIGINAL_IMAGES_DIR = os.path.join(OUTPUT_DIR, "segformer_masks")  # Use the original images
CUTOUTS_DIR = os.path.join(OUTPUT_DIR, "cutouts")

os.makedirs(CUTOUTS_DIR, exist_ok=True)

def fix_mask(mask):
    """Connects disjointed parts of a mask by filling gaps."""
    if mask is None:
        return None

    # Convert to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Fill small holes using morphological operations
    kernel = np.ones((15, 15), np.uint8)  # Bigger kernel to ensure parts connect
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def apply_alpha_mask(image, mask):
    """Applies the fixed mask to create a transparent cutout."""
    if image is None or mask is None:
        print(" Error: One or more input images are invalid.")
        return None

    # Resize mask to match the original image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to remove background
    b, g, r = cv2.split(image)
    alpha = mask.astype(np.uint8)  # Use the mask as alpha channel
    cutout = cv2.merge([b, g, r, alpha])

    return cutout

def process_image(image_path, mask_path, output_path):
    """Creates a fully connected mask and applies it to the original image."""
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f" Skipping {image_path} - Missing mask or original image.")
        return

    # Load images
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f" Error loading {image_path} or {mask_path}. Skipping...")
        return

    # Fix and connect mask regions
    fixed_mask = fix_mask(mask)

    # Apply mask to original image
    cutout = apply_alpha_mask(image, fixed_mask)

    if cutout is None:
        print(f" Skipping {image_path} due to invalid cutout creation.")
        return

    # Save the cutout with background removed
    cv2.imwrite(output_path, cutout, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def main():
    """Processes all images and generates transparent cutouts with fixed masks."""
    mask_files = sorted([f for f in os.listdir(REFINED_MASKS_DIR) if f.endswith(".png")])
    original_files = sorted([f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith(".png")])

    matching_files = list(set(mask_files) & set(original_files))

    if len(matching_files) == 0:
        print(" No matching image/mask pairs found. Ensure all processing steps completed.")
        return

    for frame_file in tqdm(matching_files, total=len(matching_files), desc="Processing cutouts"):
        image_path = os.path.join(ORIGINAL_IMAGES_DIR, frame_file)
        mask_path = os.path.join(REFINED_MASKS_DIR, frame_file)
        output_path = os.path.join(CUTOUTS_DIR, frame_file)

        process_image(image_path, mask_path, output_path)

    print(" Transparent Cutouts Successfully Created!")

if __name__ == "__main__":
    main()
