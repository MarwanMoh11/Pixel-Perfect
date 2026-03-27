"""
ABLATION BASELINE: Same architecture as train.py but with edge_weight=0.0
This trains a standard ESRGAN without our custom Edge-Aware Sharpness Loss
to prove that our custom loss actually contributes to sharper pixel art.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.esrgan import RRDBNet, Discriminator
from src.training.loss import GeneratorLoss
from src.data.dataset import PixelArtDataset
import torch.optim as optim
from tqdm import tqdm

def train_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ABLATION BASELINE] Training on device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    epochs = 50
    batch_size = 64
    lr_G = 1e-4
    # Save to a SEPARATE folder so we don't overwrite the custom model
    checkpoint_dir = 'models/checkpoints_baseline'
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Same 8-block architecture
    generator = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.9, 0.999))

    # KEY DIFFERENCE: edge_weight=0.0 (no custom Sobel edge loss)
    criterion_G = GeneratorLoss(
        pixel_weight=1.0,
        perceptual_weight=0.1,
        adv_weight=0.0,
        edge_weight=0.0       # <--- ABLATION: Custom loss DISABLED
    ).to(device)

    scaler_G = torch.cuda.amp.GradScaler()

    print("Starting BASELINE Training (No Edge-Aware Loss)...")
    for epoch in range(1, epochs + 1):
        generator.train()
        
        loop = tqdm(dataloader, leave=True)
        for idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            opt_G.zero_grad()
            
            with torch.amp.autocast('cuda'):
                fake_imgs = generator(lr_imgs)
                # Pass dummy discriminator predictions (not used when adv_weight=0)
                dummy_preds = torch.zeros(fake_imgs.size(0), 1, device=device)
                loss_G, loss_dict = criterion_G(fake_imgs, hr_imgs, dummy_preds)
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            loop.set_description(f"[BASELINE] Epoch [{epoch}/{epochs}]")
            loop.set_postfix(G_total=loss_dict['total'])

        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"RRDBNet_baseline_epoch_{epoch}.pth"))
            print(f"-> Saved Baseline Checkpoint for Epoch {epoch}")

if __name__ == '__main__':
    train_baseline()
