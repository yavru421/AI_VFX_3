import os
import json
import numpy as np
import cv2
from pathlib import Path

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def create_layered_sequence(output_dir):
    """Create multi-layered image sequence with separate layers for each frame"""
    base_dir = Path(output_dir)
    
    # Define layer sources and their descriptions
    layers = {
        'motion_vectors': {'dir': 'motion_vectors', 'desc': 'Motion Vector Data'},
        'masks': {'dir': 'masks', 'desc': 'Initial Mask'},
        'refined_masks': {'dir': 'refined_masks', 'desc': 'Refined Mask'},
        'cutouts': {'dir': 'cutouts', 'desc': 'Final Cutout'}
    }
    
    # Create output directory
    layered_dir = base_dir / 'layered_sequences'
    os.makedirs(layered_dir, exist_ok=True)

    # Get frame list from any populated directory
    frame_source = None
    for layer_info in layers.values():
        check_dir = base_dir / layer_info['dir']
        if check_dir.exists() and any(check_dir.glob('*.png')):
            frame_source = check_dir
            break

    if not frame_source:
        print("No source frames found in any directory")
        return

    # Get frame numbers
    frame_files = sorted(frame_source.glob('*.png'))
    print(f"Found {len(frame_files)} frames to process")

    # Process each frame
    for frame_file in frame_files:
        frame_name = frame_file.stem
        print(f"\nProcessing frame: {frame_name}")
        
        # Create frame directory to store layers
        frame_dir = layered_dir / frame_name
        os.makedirs(frame_dir, exist_ok=True)
        
        # Process each layer
        for layer_name, layer_info in layers.items():
            source_path = base_dir / layer_info['dir'] / frame_file.name
            if source_path.exists():
                try:
                    # Read image
                    img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Save layer with descriptive name
                        layer_path = frame_dir / f"{layer_name}.png"
                        cv2.imwrite(str(layer_path), img)
                        print(f"  ✓ {layer_name}")
                        
                        # Create layer info file if doesn't exist
                        info_file = frame_dir / "layers.txt"
                        if not info_file.exists():
                            with open(info_file, 'w') as f:
                                f.write(f"Frame: {frame_name}\nLayers:\n")
                        
                        # Append layer info
                        with open(info_file, 'a') as f:
                            f.write(f"- {layer_name}: {layer_info['desc']}\n")
                    else:
                        print(f"  ✗ {layer_name} (invalid image)")
                except Exception as e:
                    print(f"  ✗ {layer_name} (error: {e})")
            else:
                print(f"  ✗ {layer_name} (missing)")

def main():
    try:
        config = load_config()
        create_layered_sequence(config['output_dir'])
        print("\nSequence processing complete!")
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()