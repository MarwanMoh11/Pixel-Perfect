import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.esrgan import RRDBNet
from src.data.dataset import PixelArtDataset
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np

def tensor_to_img(tensor):
    """Convert a [0, 1] tensor back to a numpy image [0, 255] for skimage metrics"""
    img = tensor.cpu().detach().squeeze(0).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8)

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models and dataset for Evaluation...")
    
    checkpoint_dir = 'models/checkpoints'
    files = [f for f in os.listdir(checkpoint_dir) if "RRDBNet_epoch_" in f]
    if not files:
        print("Error: No Generator checkpoints found")
        return
        
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    model_path = os.path.join(checkpoint_dir, files[-1])
    print(f"Loading checkpoint: {model_path}")

    # Must match training architecture (8 blocks)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # Dataset now auto-crops sprites to content and resizes to fill 128x128
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    lr_imgs, hr_imgs = next(iter(dataloader))
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)

    val_psnr, val_ssim, val_lpips = [], [], []

    print("Running Inference over baseline vs Custom ESRGAN...")
    with torch.no_grad():
        # Baseline: standard bicubic upscaling (what every basic image editor does)
        baseline_imgs = F.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False)
        baseline_imgs = torch.clamp(baseline_imgs, 0.0, 1.0)
        
        # Our model: learned pixel-art-aware super resolution
        sr_imgs = torch.clamp(model(lr_imgs), 0.0, 1.0)
        
        for i in range(5):
            hr = hr_imgs[i]
            sr = sr_imgs[i]
            
            lpips_score = loss_fn_vgg((sr.unsqueeze(0)*2)-1, (hr.unsqueeze(0)*2)-1)
            val_lpips.append(lpips_score.item())

            hr_np = tensor_to_img(hr)
            sr_np = tensor_to_img(sr)
            
            p = psnr(hr_np, sr_np, data_range=255)
            s = ssim(hr_np, sr_np, channel_axis=-1, data_range=255)
            
            val_psnr.append(p)
            val_ssim.append(s)

    avg_psnr = np.mean(val_psnr)
    avg_ssim = np.mean(val_ssim)
    avg_lpips = np.mean(val_lpips)

    print("\n" + "="*40)
    print("EVALUATION METRICS COMPLETED")
    print("="*40)
    print(f"Average PSNR:  {avg_psnr:.2f} dB  (Higher is better)")
    print(f"Average SSIM:  {avg_ssim:.4f}      (Closer to 1 is better)")
    print(f"Average LPIPS: {avg_lpips:.4f}     (Lower is better)")
    print("="*40 + "\n")

    # Generate Visual Comparison Grid — images now fill the full canvas (no black padding)
    print("Generating Visual Comparison Grid...")
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    titles = ["Ground Truth (128×128)", "Bicubic Upscale (Baseline)", "Pixel-Perfect AI (Ours)"]

    for i in range(5):
        orig_img = tensor_to_img(hr_imgs[i])
        stan_img = tensor_to_img(baseline_imgs[i])
        ours_img = tensor_to_img(sr_imgs[i])
        
        axes[i, 0].imshow(orig_img, interpolation='nearest')
        axes[i, 1].imshow(stan_img, interpolation='nearest')
        axes[i, 2].imshow(ours_img, interpolation='nearest')
        
        for j in range(3):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/visual_comparison_grid.png', dpi=150)
    print("Saved 'outputs/visual_comparison_grid.png'")
    
    # LPIPS Bar Chart
    print("Generating LPIPS Bar Chart...")
    with torch.no_grad():
        baseline_lpips_scores = [loss_fn_vgg((baseline_imgs[i].unsqueeze(0)*2)-1, (hr_imgs[i].unsqueeze(0)*2)-1).item() for i in range(5)]
    avg_base_lpips = np.mean(baseline_lpips_scores)
    
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    models = ['Baseline (Bicubic)', 'Pixel-Perfect (Ours)']
    scores = [avg_base_lpips, avg_lpips]
    bars = ax2.bar(models, scores, color=['#e74c3c', '#2ecc71'], width=0.5, edgecolor='black')
    ax2.set_ylabel('LPIPS Score (Lower is Better)', fontsize=12)
    ax2.set_title('Perceptual Quality: Baseline vs Pixel-Perfect', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(scores) * 1.3)
    
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/lpips_bar_chart.png', dpi=150)
    print("Saved 'outputs/lpips_bar_chart.png'")
    
    print("\nEvaluation complete! Check the outputs/ folder for your assignment figures.")

if __name__ == '__main__':
    evaluate()
