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

def extract_sprite(img_np, hr_gt):
    """Dynamically finds the bounding box of non-black pixels in the Ground Truth and tight-crops the image."""
    grayscale = np.sum(hr_gt, axis=2)
    non_black = np.where(grayscale > 0)
    
    if len(non_black[0]) == 0 or len(non_black[1]) == 0:
        return img_np 
        
    y_min, y_max = np.min(non_black[0]), np.max(non_black[0])
    x_min, x_max = np.min(non_black[1]), np.max(non_black[1])
    
    y_min, y_max = max(0, y_min - 2), min(img_np.shape[0], y_max + 3)
    x_min, x_max = max(0, x_min - 2), min(img_np.shape[1], x_max + 3)
    
    return img_np[y_min:y_max, x_min:x_max]

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

    # Ensure evaluation matches the scaled down 8-block network
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 3. Use the original trained PyTorch Dataset so metrics match training reality distribution
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    lr_imgs, hr_imgs = next(iter(dataloader))
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)

    val_psnr, val_ssim, val_lpips = [], [], []

    print("Running Inference over baseline vs Custom ESRGAN...")
    with torch.no_grad():
        baseline_imgs = F.interpolate(lr_imgs, scale_factor=4, mode='bicubic', align_corners=False)
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

    print("\n" + "="*40)
    print("EVALUATION METRICS COMPLETED")
    print(f"Average PSNR:  {np.mean(val_psnr):.2f} dB")
    print(f"Average SSIM:  {np.mean(val_ssim):.4f}")
    print(f"Average LPIPS: {np.mean(val_lpips):.4f}")
    print("="*40 + "\n")

    print("Generating Visual Comparison Grid...")
    os.makedirs('outputs', exist_ok=True)
    
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    titles = ["Original (Zoomed)", "Standard Upscale (Blurry)", "Pixel-Perfect AI (Ours)"]

    for i in range(5):
        hr_np_full = tensor_to_img(hr_imgs[i])
        orig_img = extract_sprite(hr_np_full, hr_np_full)
        stan_img = extract_sprite(tensor_to_img(baseline_imgs[i]), hr_np_full)
        ours_img = extract_sprite(tensor_to_img(sr_imgs[i]), hr_np_full)
        
        axes[i, 0].imshow(orig_img)
        axes[i, 1].imshow(stan_img)
        axes[i, 2].imshow(ours_img)
        
        for j in range(3):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(titles[j])

    plt.tight_layout()
    plt.savefig('outputs/visual_comparison_grid.png')
    
    print("Generating LPIPS Bar Chart...")
    with torch.no_grad():
        baseline_lpips_scores = [loss_fn_vgg((baseline_imgs[i].unsqueeze(0)*2)-1, (hr_imgs[i].unsqueeze(0)*2)-1).item() for i in range(5)]
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    scores = [np.mean(baseline_lpips_scores), np.mean(val_lpips)]
    ax2.bar(['Baseline (Bicubic)', 'Pixel-Perfect (Ours)'], scores, color=['red', 'green'])
    ax2.set_ylabel('LPIPS Score (Lower is better)')
    ax2.set_title('Perceptual Quality Comparison')
    ax2.set_ylim(0, max(scores) + 0.1)
    
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.savefig('outputs/lpips_bar_chart.png')
    print("Evaluation successfully finished.")

if __name__ == '__main__':
    evaluate()
