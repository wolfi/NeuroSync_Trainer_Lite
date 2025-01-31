# NeuroSync Trainer Lite

## Overview

**NeuroSync Trainer Lite** is a tool for training an **audio-to-face blendshape transformer model**, enabling the generation of facial animation from audio input. This is useful for real-time applications like **virtual avatars, game characters, and animation pipelines**.

## Features
- **Audio-Driven Facial Animation** – Train a model to generate realistic blendshape animations from audio input.
- **Multi-GPU Support** – Train efficiently using up to 4 GPUs.
- **Integration with Unreal Engine** – Send trained animation data to Unreal Engine via **NeuroSync Local API** and **NeuroSync Player**.
- **Optimized for iPhone Face Data** – Easily process facial motion capture data from an iPhone.

## Quick Start
### 1. Install Dependencies
Before training, ensure you have the required dependencies installed, including:
- Python 3.9+
- PyTorch with CUDA support (for GPU acceleration)
- NumPy, Pandas, Librosa, OpenCV, Matplotlib, and other required Python libraries
- FFMPEG for linux should be installed globally, Windows users need to get a built ffmpeg.exe and put it inside utils\video\_ffmpeg\bin to correctly strip the audio from the .mov in the face data folders....

### 2. Collect & Prepare Data
To train the model, you need audio and facial blendshape data. Follow these steps:
1. **Record Face & Audio Data** using an iPhone with **ARKit Blendshapes**.
2. **Download & Extract the Data** to your local machine.
3. **Move Data to the Correct Folder:**
   - Place each extracted folder inside `dataset/data/`.

### 3. Train the Model
Once your data is ready, start training by running:
```bash
python train.py
```

## Multi-GPU Training
If you want to train using multiple GPUs, update the **configuration file**:
1. Open `config.py`.
2. Set `use_multi_gpu = True`.
3. Define the number of GPUs:
   ```python
   'use_multi_gpu': True,
   'num_gpus': 4  # Adjust as needed, max 4 GPUs
   ```
4. Start training as usual.

You can easily modify the code to support **more than 4 GPUs**—just ask ChatGPT for assistance!

## Using the NeuroSync Model
To **generate facial blendshapes from audio** and send them to Unreal Engine, you'll need:
- [**NeuroSync Local API**](https://github.com/AnimaVR/NeuroSync_Local_API) – Handles real-time facial data processing.
- [**NeuroSync Player**](https://github.com/AnimaVR/NeuroSync_Player) – Sends the animation data to Unreal Engine or any **LiveLink-compatible** software.

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

### **You are free to:**
✔ **Share** – Copy and redistribute the material in any medium or format.
✔ **Adapt** – Remix, transform, and build upon the material.

### **Under the following terms:**
- **Attribution** – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** – You may not use the material for **commercial purposes**.

[View Full License](http://creativecommons.org/licenses/by-nc/4.0/)

&copy; 2025 NeuroSync Trainer Lite Team

