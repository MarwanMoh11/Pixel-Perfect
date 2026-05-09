# Pixel-Perfect: Content-Aware Super-Resolution
**Advanced Machine Learning | American University in Cairo | 2026**
**Team:** Hadi Hesham, Marwan Abudaif

---

## 1. The Challenge
Retro game art (pixel art) relies on sharp, aliased edges and limited color palettes. Standard upscaling methods (Bicubic, SRGAN, SwinIR) fail by:
- Creating blurry, muddy edges.
- Introducing "oil painting" artifacts by smoothing intentional pixel boundaries.
- Drifting from original hardware-constrained color palettes.

## 2. Our Innovation
We developed **Pixel-Perfect**, a specialized ESRGAN-based architecture with three key domain-specific adaptations:
- **Nearest-Neighbor Training**: We use NN-downsampling to create LR-HR pairs, teaching the model the geometric structure of pixels.
- **Edge-Aware Sharpness Loss**: A Sobel-based penalty that forces the GAN to output hard, binary edges instead of smooth gradients.
- **Palette-Constraint Loss**: Ensures generated colors fall on valid 8-bit/16-bit color grids.

## 3. Architecture
- **Generator**: RRDBNet (8 Residual-in-Residual Dense Blocks).
- **Discriminator**: VGG-style binary classifier.
- **Total Loss**: L1 + LPIPS + Edge-Aware + Palette + Adversarial.

## 4. Key Results
Our model achieves a **79x improvement** in perceptual quality (LPIPS) compared to standard bicubic upscaling.

| Metric | Bicubic | Pixel-Perfect (Ours) |
|--------|---------|----------------------|
| PSNR   | 23.18   | **41.10**            |
| SSIM   | 0.80    | **0.99**             |
| LPIPS  | 0.48    | **0.0029**           |

## 5. Visual Demo
*Include side-by-side comparison image of Mario sprite here*
- **Left**: Original 1x
- **Middle**: Bicubic (Blurry)
- **Right**: Pixel-Perfect (Sharp & Clean)

## 6. Conclusion
Domain-specific priors are essential for Super-Resolution in non-photorealistic domains. By tailoring the loss functions to pixel art geometry, we achieved near-perfect 4x upscaling.
