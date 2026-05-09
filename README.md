# Pixel-Perfect: Content-Aware Super-Resolution for Retro Game Assets

This repository contains the implementation for a deep learning project aimed at restoring and upscaling retro video game art (pixel art) using Generative Adversarial Networks (GANs).

## Abstract
Standard interpolation (Bicubic, Bilinear) and modern state-of-the-art Super-Resolution models (like SRGAN) often fail on pixel art. They either produce blurry images or interpret sharp pixel edges as noise, resulting in an "oil painting" effect. **Pixel-Perfect** introduces a specialized model based on ESRGAN that respects the geometric constraints of pixel art (sharp edges, limited color palettes) while increasing the resolution by a factor of 4x.

## Authors
- Hadi Hesham
- Marwan Abudaif

## Key Features

1. **Nearest-Neighbor Training Pairs**: Standard methods use bicubic downsampled images for low-resolution counterparts. In contrast, our pipeline relies on Nearest-Neighbor downsampling to maintain the blocky geometry true to retro game assets.
2. **Edge-Aware Sharpness Loss**: A custom penalty applied to gradients indicating anti-aliasing or smooth transitions. This forces the GAN to output hard, binary edges and eliminates the "oily" look of standard standard super-resolution outputs.
3. **Based on ESRGAN**: Utilizes the robust RRDBNet (Residual-in-Residual Dense Block Network) generator as the backbone.

## Dataset
This project uses the [Kaggle Pixel Art Dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art), containing over 89,000 pixel-art images of sprites and game objects.

## Installation

Create a virtual environment and install the required packages:

```bash
git clone https://github.com/MarwanMoh11/Pixel-Perfect.git
cd Pixel-Perfect
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Live Demo

We provide two ways to experience the **Pixel-Perfect** model in practice:

### 1. Interactive Mario Demo
A fully playable Mario clone where you can toggle the AI upscaling in real-time.
```bash
cd mario_clone
python3 mario_level_1.py
```
- **Press `T`**: Toggle between Original (Nearest-Neighbor) and ESRGAN (Pixel-Perfect) graphics.
- **Arrow Keys**: Move and Jump.

### 2. Side-by-Side Comparison Viewer
A static viewer that allows you to compare the three stages of upscaling across the entire first level.
```bash
python3 scripts/compare_viewer.py
```
- **Left Panel**: True Original (1x NES Scale)
- **Middle Panel**: Nearest-Neighbor (4x Stretched)
- **Right Panel**: AI-Upscaled (Pixel-Perfect ESRGAN)

## Model Weights & Dataset

- **Model Weights**: [Download Checkpoints](https://example.com/weights) (RRDBNet Generator)
- **Dataset**: [Kaggle Pixel Art Dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art)

## Directory Structure

```text
Pixel-Perfect/
├── mario_clone/        # Playable Mario Demo
├── scripts/            # Evaluation and Utility scripts
├── src/                # Core Source Code (ESRGAN, Training Loops)
├── slides/             # Project Presentation
├── models/             # Model Checkpoints
├── README.md
└── requirements.txt
```

## Team Contributions

- **Hadi Hesham**: Data engineering, dataset preprocessing (Nearest-Neighbor downsampling), data loaders, and initial EDA.
- **Marwan Abudaif**: Model architecture (ESRGAN/RRDBNet), training loop implementation, Edge-Aware & Palette loss functions, and demo integration.

## Evaluation Metrics

We use the following metrics to evaluate upscaling quality:
- **PSNR**: Peak Signal-to-Noise Ratio.
- **SSIM**: Structural Similarity Index.
- **LPIPS**: Learned Perceptual Image Patch Similarity (Captured the 79x improvement in visual quality).
