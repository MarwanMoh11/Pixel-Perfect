"""
Ablation Comparison: Edge-Aware Model vs Baseline Model
Generates side-by-side metrics and a 4-column visual grid:
  Ground Truth | Bicubic | Baseline (No Edge Loss) | Ours (With Edge Loss)
"""
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
    img = tensor.cpu().detach().squeeze(0).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8)

def load_model(checkpoint_dir, prefix="RRDBNet", device='cpu'):
    """Load the latest checkpoint from a directory."""
    files = [f for f in os.listdir(checkpoint_dir) if prefix in f and f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    path = os.path.join(checkpoint_dir, files[-1])
    print(f"  Loading: {path}")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def compute_metrics(sr_imgs, hr_imgs, loss_fn_vgg, count=5):
    """Compute PSNR, SSIM, LPIPS for a batch."""
    val_psnr, val_ssim, val_lpips = [], [], []
    for i in range(count):
        hr = hr_imgs[i]
        sr = sr_imgs[i]
        
        with torch.no_grad():
            lp = loss_fn_vgg((sr.unsqueeze(0)*2)-1, (hr.unsqueeze(0)*2)-1)
        val_lpips.append(lp.item())
        
        hr_np = tensor_to_img(hr)
        sr_np = tensor_to_img(sr)
        val_psnr.append(psnr(hr_np, sr_np, data_range=255))
        val_ssim.append(ssim(hr_np, sr_np, channel_axis=-1, data_range=255))
    
    return np.mean(val_psnr), np.mean(val_ssim), np.mean(val_lpips)

def ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation Study on: {device}\n")
    
    # 1. Load both models
    print("Loading Custom Model (With Edge Loss)...")
    model_ours = load_model('models/checkpoints', prefix='RRDBNet_epoch', device=device)
    
    print("Loading Baseline Model (No Edge Loss)...")
    model_base = load_model('models/checkpoints_baseline', prefix='RRDBNet_baseline', device=device)
    
    if model_ours is None or model_base is None:
        print("ERROR: Could not find checkpoints for both models.")
        print("  Custom model dir: models/checkpoints/")
        print("  Baseline model dir: models/checkpoints_baseline/")
        return

    # 2. Load test images
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    lr_imgs, hr_imgs = next(iter(dataloader))
    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

    # 3. Run inference
    with torch.no_grad():
        bicubic_imgs = torch.clamp(F.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False), 0, 1)
        baseline_imgs = torch.clamp(model_base(lr_imgs), 0, 1)
        ours_imgs = torch.clamp(model_ours(lr_imgs), 0, 1)

    # 4. Compute metrics for all three methods
    bic_psnr, bic_ssim, bic_lpips = compute_metrics(bicubic_imgs, hr_imgs, loss_fn_vgg)
    base_psnr, base_ssim, base_lpips = compute_metrics(baseline_imgs, hr_imgs, loss_fn_vgg)
    ours_psnr, ours_ssim, ours_lpips = compute_metrics(ours_imgs, hr_imgs, loss_fn_vgg)

    print("\n" + "="*65)
    print("ABLATION STUDY RESULTS")
    print("="*65)
    print(f"{'Method':<30} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-"*65)
    print(f"{'Bicubic (Baseline)':<30} {bic_psnr:>10.2f} {bic_ssim:>10.4f} {bic_lpips:>10.4f}")
    print(f"{'ESRGAN (No Edge Loss)':<30} {base_psnr:>10.2f} {base_ssim:>10.4f} {base_lpips:>10.4f}")
    print(f"{'ESRGAN + Edge Loss (Ours)':<30} {ours_psnr:>10.2f} {ours_ssim:>10.4f} {ours_lpips:>10.4f}")
    print("="*65)

    # 5. Generate 4-column visual comparison grid
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 18))
    titles = ["Ground Truth", "Bicubic", "ESRGAN (No Edge)", "ESRGAN + Edge (Ours)"]

    for i in range(5):
        imgs = [
            tensor_to_img(hr_imgs[i]),
            tensor_to_img(bicubic_imgs[i]),
            tensor_to_img(baseline_imgs[i]),
            tensor_to_img(ours_imgs[i])
        ]
        for j in range(4):
            axes[i, j].imshow(imgs[j], interpolation='nearest')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/ablation_visual_grid.png', dpi=150)
    print("\nSaved 'outputs/ablation_visual_grid.png'")

    # 6. Ablation bar chart (grouped bars for all 3 metrics)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    methods = ['Bicubic', 'No Edge Loss', 'Edge Loss (Ours)']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    # PSNR
    axes2[0].bar(methods, [bic_psnr, base_psnr, ours_psnr], color=colors, edgecolor='black')
    axes2[0].set_title('PSNR (dB) ↑', fontsize=13, fontweight='bold')
    axes2[0].set_ylabel('dB')
    for i, v in enumerate([bic_psnr, base_psnr, ours_psnr]):
        axes2[0].text(i, v + 0.3, f"{v:.2f}", ha='center', fontweight='bold')

    # SSIM
    axes2[1].bar(methods, [bic_ssim, base_ssim, ours_ssim], color=colors, edgecolor='black')
    axes2[1].set_title('SSIM ↑', fontsize=13, fontweight='bold')
    for i, v in enumerate([bic_ssim, base_ssim, ours_ssim]):
        axes2[1].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')

    # LPIPS
    axes2[2].bar(methods, [bic_lpips, base_lpips, ours_lpips], color=colors, edgecolor='black')
    axes2[2].set_title('LPIPS ↓', fontsize=13, fontweight='bold')
    for i, v in enumerate([bic_lpips, base_lpips, ours_lpips]):
        axes2[2].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')

    plt.suptitle('Ablation Study: Impact of Edge-Aware Sharpness Loss', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/ablation_metrics_chart.png', dpi=150, bbox_inches='tight')
    print("Saved 'outputs/ablation_metrics_chart.png'")
    
    print("\nAblation study complete! All figures saved to outputs/")

if __name__ == '__main__':
    ablation()
