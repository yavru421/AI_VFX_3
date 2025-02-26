# AI VFX Pipeline

A Python-based VFX pipeline that uses AI to extract people from videos with transparency. Built this because I got tired of rotoscoping everything manually.

## Features
- Extracts frames from video at correct FPS
- Uses SegFormer for AI-powered person detection
- Generates clean alpha channel cutouts
- Full GUI interface for easy operation
- Directory cleanup tools included
- Progress tracking and system monitoring
- Supports batch processing

## How to Use
1. Load your video through the GUI
2. Pick individual steps or run the full pipeline:
   - Frame extraction
   - Person mask generation
   - Mask processing
   - Final cutout creation
3. Get your cutouts from the output folder

## Directory Structure
```
output/
  ├── original_frames/    # Extracted video frames
  ├── masks/             # AI-generated masks
  ├── cutouts/           # Final transparent PNGs
  └── debug/             # Debug outputs if needed
```

## Models Used
- **SegFormer-B3** (nvidia/segformer-b3-finetuned-ade-512-512)
  - Used for person segmentation
  - Trained on ADE20K dataset
  - Good balance of speed and accuracy
  - Label 12 = person class

## Requirements
- Python 3.8+
- PyQt6
- OpenCV
- PyTorch
- Transformers
- FFMPEG

## Installation
```bash
# Clone the repo
git clone [repo_url]

# Create and activate venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Example Pipeline
Check out `example_pipeline_images/` to see:
- Input frame examples
- SegFormer mask generation
- Mask refinement process
- Final cutout results

This helps visualize how each step transforms the image.

## Known Issues
- Might need to clear output folders between runs
- First frame sometimes needs a second pass
- GUI can be slow with huge folders

## Tips
- Run on GPU if you can, way faster
- Clean directories between tests
- Check debug output if masks look weird
- Use the individual step buttons for testing

## License
Mozilla Public License Version 2.0
- Source code must stay open source
- Can be used in commercial projects
- Modifications must be shared under MPL-2.0
- Must include original copyright notice
- Full license: https://www.mozilla.org/en-US/MPL/2.0/

## Why MPL-2.0?
- Protects the code from being closed-source
- Requires improvements to be shared back
- Still allows commercial use
- Preserves credit to original author
- Stronger than MIT but not as restrictive as GPL

## Credits
Built this while trying not to lose my mind doing manual roto. Thanks to:
- NVIDIA for SegFormer
- Hugging Face for the model hosting
- The PyQt team for making GUIs less painful
