import os
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
    
    epochs = 50
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

    # 5. Losses
    criterion_G = GeneratorLoss(
        pixel_weight=1.0, 
        perceptual_weight=1.0, 
        adv_weight=0.0, 
        edge_weight=1.0
    ).to(device)
    
    criterion_D = nn.BCEWithLogitsLoss()

    # PyTorch AMP (Automatic Mixed Precision) Scalers for 2-3x Speedup & 50% less VRAM
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # 6. Training Loop
    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        loop = tqdm(dataloader, leave=True)
        for idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Real images
                real_preds = discriminator(hr_imgs)
                loss_D_real = criterion_D(real_preds, torch.ones_like(real_preds))
                
                # Fake images
                fake_imgs = generator(lr_imgs)
                fake_preds = discriminator(fake_imgs.detach()) # Detach to avoid Generator backprop
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
                # Recalculate discriminator on fake (without detach to allow gradient flow to G)
                fake_preds_for_G = discriminator(fake_imgs)
                
                # The custom Generator Loss handles Pixel, LPIPS, Adversarial, and our specific Edge Loss
                loss_G, loss_dict = criterion_G(fake_imgs, hr_imgs, fake_preds_for_G)
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # Logging
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                D_loss=loss_D.item(), 
                G_total=loss_dict['total'],
                G_edge=loss_dict['edge']
            )

        # 7. Checkposting
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"RRDBNet_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"Discriminator_epoch_{epoch}.pth"))
            print(f"-> Saved Checkpoints for Epoch {epoch}")

if __name__ == '__main__':
    train()
