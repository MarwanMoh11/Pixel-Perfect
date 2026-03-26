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
    
    # 1. Load the most recent checkpoint.
    checkpoint_dir = 'models/checkpoints'
    
    # Attempt to locate the epoch 50 weights
    model_path = os.path.join(checkpoint_dir, 'RRDBNet_epoch_50.pth')
    if not os.path.exists(model_path):
        # Fallback to searching the directory for the latest checkpoint
        files = [f for f in os.listdir(checkpoint_dir) if "RRDBNet_epoch_" in f]
        if not files:
            print("Error: No Generator checkpoints found in models/checkpoints/")
            return
        # Sort files by their integer epoch number instead of alphabetically
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = os.path.join(checkpoint_dir, files[-1])
        print(f"Did not find epoch 50. Using closest checkpoint: {model_path}")

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Setup metric calculators
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 3. Get 5 random test images
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    # Using batch_size=5 ensures we get 5 distinct images for our plot
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    lr_imgs, hr_imgs = next(iter(dataloader))
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)

    val_psnr = []
    val_ssim = []
    val_lpips = []

    print("Running Inference over baseline vs Custom ESRGAN...")
    with torch.no_grad():
        # Baseline: bicubic upsampling of the nearest-neighbor degraded LR image
        # This simulates a "Standard Upscaling"
        baseline_imgs = F.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False)
        
        # Our Model: Super Resolution execution
        sr_imgs = model(lr_imgs)
        # Clamp to valid image range
        sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)
        
        # Calculate scores
        for i in range(5):
            hr = hr_imgs[i]
            sr = sr_imgs[i]
            
            # LPIPS expects [-1, 1] range
            lpips_score = loss_fn_vgg((sr.unsqueeze(0)*2)-1, (hr.unsqueeze(0)*2)-1)
            val_lpips.append(lpips_score.item())

            # Numpy PSNR / SSIM requires uint8 or float format conversions
            hr_np = tensor_to_img(hr)
            sr_np = tensor_to_img(sr)
            
            p = psnr(hr_np, sr_np, data_range=255)
            # win_size=3 since our images are standard size, channel_axis=2 handles RGB properly
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

    # 4. Generate Visual Comparison Grid
    print("Generating Visual Comparison Grid...")
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    titles = ["Original (128x128)", "Standard Upscale (Blurry)", "Pixel-Perfect AI (Ours)"]

    for i in range(5):
        orig_img = tensor_to_img(hr_imgs[i])
        stan_img = tensor_to_img(baseline_imgs[i])
        ours_img = tensor_to_img(sr_imgs[i])
        
        axes[i, 0].imshow(orig_img)
        axes[i, 1].imshow(stan_img)
        axes[i, 2].imshow(ours_img)
        
        for j in range(3):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j])

    plt.tight_layout()
    plt.savefig('outputs/visual_comparison_grid.png')
    print("Saved 'outputs/visual_comparison_grid.png'")

    # 5. Generate LPIPS Bar Chart comparison (Dummy baseline LPIPS for chart)
    # The assignment specifies plotting a Bar Chart vs Baseline.
    print("Generating LPIPS Bar Chart...")
    with torch.no_grad():
        baseline_lpips_scores = [loss_fn_vgg((baseline_imgs[i].unsqueeze(0)*2)-1, (hr_imgs[i].unsqueeze(0)*2)-1).item() for i in range(5)]
    avg_base_lpips = np.mean(baseline_lpips_scores)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    models = ['Baseline (Bicubic)', 'Pixel-Perfect (Ours)']
    scores = [avg_base_lpips, avg_lpips]
    bars = ax2.bar(models, scores, color=['red', 'green'])
    ax2.set_ylabel('LPIPS Score (Lower is better)')
    ax2.set_title('Perceptual Quality Comparison')
    
    # Force Y-axis limit for readability
    ax2.set_ylim(0, max(scores) + 0.1)

    for i, v in enumerate(scores):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.savefig('outputs/lpips_bar_chart.png')
    print("Saved 'outputs/lpips_bar_chart.png'")
    
    print("\nEvaluation successfully finished. All required assignment graphs are generated in the outputs/ directory.")

if __name__ == '__main__':
    evaluate()
