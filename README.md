# AI_VFX_3: Offline AI-Powered Background Removal  

#its a work in progress, gut it does work!

🚀 **No APIs. No Paywalls. 100% Local AI Video Processing.**  

## ✨ Why This Exists  
Most "AI background removers" force you to use cloud-based APIs, hidden subscriptions, or online accounts.  
**This project is different.**  
- **Runs 100% locally** – No internet required.  
- **No API keys needed** – Your data stays on your machine.  
- **Open-source, but protected** – Anyone can use it, but no one can rebrand or resell it without permission.  

## 🎬 Features  
✅ **AI-Based Background Removal** (No green screen required!)  
✅ **FFmpeg Motion Vectors + AI Segmentation** for high accuracy  
✅ **GUI for Easy Use** (No command-line required)  
✅ **Works on Any Video File** (MP4, MOV, AVI, etc.)  
✅ **Fully Modular** – You can swap AI models if needed  
✅ **No Subscription Fees. No Bullsh*t.**  

🔹 **Pro Version Available** *(Coming Soon!)*  
💡 Looking for **better mask refinement, auto batch processing, and plugin support?**  
🛒 Stay tuned for the **Pro version** that includes premium features!  

## ⚙️ Requirements

- Python 3.9+  
- CUDA capable GPU  
- FFmpeg  

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/yavru421/AI_VFX_3.git
cd AI_VFX_3
```

2. Create  environment:
```bash
env create -f environment.yml
activate ai_vfx_pipeline
```

## 🚀 Usage

1. Launch the GUI:
```bash
python gui_main.py
```

2. Load your video file and follow the processing steps:
   - Extract Motion Vectors
   - Run AI Processing
   - Refine Masks 
   - Run SegFormer
   - Generate Transparent PNGs

## 📂 Output Structure

```
output/
├── motion_vectors/
├── masks/
├── refined_masks/
├── segformer_masks/
└── cutouts/
```

## ⚖️ License

This project is licensed under the Mozilla Public License 2.0 with additional commercial use restrictions. See the [LICENSE](LICENSE) file for details.

Key points:
- Source code must remain open source
- Commercial use requires explicit permission
- Modifications must be shared back
- No closed-source redistribution


https://youtu.be/mYFrpPO2QrA

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and code of conduct.
