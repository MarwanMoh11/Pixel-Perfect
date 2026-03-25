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
pip install -r requirements.txt
```

## Directory Structure

```text
Pixel-Perfect/
├── configs/            # Configuration files (YAML/JSON)
├── data/               # Raw and processed datasets (ignored in git)
├── models/             # Pre-trained and locally trained checkpoints (ignored in git)
├── notebooks/          # Jupyter notebooks for data exploration and visualizations
├── outputs/            # Output images, logs, and evaluation figures
├── scripts/            # Standalone utility scripts
├── src/                # Core source code
│   ├── data/           # Data loaders and nearest-neighbor transforms
│   ├── models/         # ESRGAN definition (RRDBNet & Discriminator)
│   └── training/       # Loss functions (including Edge-Aware Sharpness Loss) and training loops
├── .gitignore
├── README.md
└── requirements.txt
```

## Evaluation Metrics

This project uses the following metrics to evaluate upscaling quality:
- **PSNR**: Peak Signal-to-Noise Ratio.
- **SSIM**: Structural Similarity Index.
- **LPIPS**: Learned Perceptual Image Patch Similarity. Critical for capturing how "natural" and perceptually accurate the image is compared to the original sprite.
