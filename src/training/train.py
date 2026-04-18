import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.esrgan import RRDBNet, Discriminator
from src.training.loss import GeneratorLoss
from src.data.dataset import PixelArtDataset
import torch.optim as optim
from tqdm import tqdm

def train():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # CuDNN benchmark accelerates convolutions if input sizes remain constant (like our 128x128 crops)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    epochs = 30
    batch_size = 64  # Increased from 16 to 64 to fully utilize 15GB Colab GPU
    lr_G = 1e-4
    lr_D = 1e-4
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Dataloaders
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # 3. Models
    # Reduced depth from 23 to 8 blocks for 3x faster training. 
    # 8 blocks is perfectly mathematically sufficient for simple pixel-art geometry!
    generator = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32).to(device)
    discriminator = Discriminator(in_nc=3, nf=64).to(device)

    # 4. Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.9, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.9, 0.999))

    # 5. Losses — With proper content-filled training data, we can safely use all losses
    criterion_G = GeneratorLoss(
        pixel_weight=1.0,        # Strong L1 for pixel-accurate reconstruction
        perceptual_weight=0.1,   # Light LPIPS for structural coherence
        adv_weight=5e-3,         # Adversarial ENABLED for Milestone 2 (Sharper textures)
        edge_weight=0.5,         # Custom Sobel edge penalty for sharp pixel borders
        palette_weight=0.05      # Milestone 2: Enforce discrete color palettes
    ).to(device)
    
    criterion_D = nn.BCEWithLogitsLoss()

    # PyTorch AMP (Automatic Mixed Precision) Scalers for 2-3x Speedup & 50% less VRAM
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    # 6. Loss log CSV
    log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'G_total', 'G_l1', 'G_perceptual', 'G_edge', 'G_palette', 'G_adv', 'D_loss'])

    # 7. Training Loop
    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        # Accumulators for epoch-average losses
        epoch_G_total, epoch_G_l1, epoch_G_perc, epoch_G_edge, epoch_G_pal, epoch_G_adv, epoch_D = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches = 0
        
        loop = tqdm(dataloader, leave=True)
        for idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            with torch.amp.autocast('cuda'):
                real_preds = discriminator(hr_imgs)
                loss_D_real = criterion_D(real_preds, torch.ones_like(real_preds))
                fake_imgs = generator(lr_imgs)
                fake_preds = discriminator(fake_imgs.detach())
                loss_D_fake = criterion_D(fake_preds, torch.zeros_like(fake_preds))
                loss_D = (loss_D_real + loss_D_fake) / 2
                
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_G.zero_grad()
            
            with torch.amp.autocast('cuda'):
                fake_preds_for_G = discriminator(fake_imgs)
                loss_G, loss_dict = criterion_G(fake_imgs, hr_imgs, fake_preds_for_G)
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # Accumulate
            epoch_G_total += loss_dict['total']
            epoch_G_l1 += loss_dict['l1']
            epoch_G_perc += loss_dict['perceptual']
            epoch_G_edge += loss_dict['edge']
            epoch_G_pal += loss_dict['palette']
            epoch_G_adv += loss_dict['adversarial']
            epoch_D += loss_D.item()
            num_batches += 1

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                D_loss=loss_D.item(), 
                G_total=loss_dict['total'],
                G_edge=loss_dict['edge'],
                G_pal=loss_dict['palette']
            )

        # Log epoch averages to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                epoch_G_total / num_batches,
                epoch_G_l1 / num_batches,
                epoch_G_perc / num_batches,
                epoch_G_edge / num_batches,
                epoch_G_pal / num_batches,
                epoch_G_adv / num_batches,
                epoch_D / num_batches
            ])

        # Checkpoints
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"RRDBNet_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"Discriminator_epoch_{epoch}.pth"))
            print(f"-> Saved Checkpoints for Epoch {epoch}")

if __name__ == '__main__':
    train()
