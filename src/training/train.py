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
    
    epochs = 50
    batch_size = 16
    lr_G = 1e-4
    lr_D = 1e-4
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Dataloaders
    dataset = PixelArtDataset(root_dir='data/raw', hr_size=128, scale=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Models
    generator = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32).to(device)
    discriminator = Discriminator(in_nc=3, nf=64).to(device)

    # 4. Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.9, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.9, 0.999))

    # 5. Losses
    criterion_G = GeneratorLoss(
        pixel_weight=1e-2, 
        perceptual_weight=1.0, 
        adv_weight=5e-3, 
        edge_weight=1e-1
    ).to(device)
    
    criterion_D = nn.BCEWithLogitsLoss()

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
            
            # Real images
            real_preds = discriminator(hr_imgs)
            loss_D_real = criterion_D(real_preds, torch.ones_like(real_preds))
            
            # Fake images
            fake_imgs = generator(lr_imgs)
            fake_preds = discriminator(fake_imgs.detach()) # Detach to avoid Generator backprop
            loss_D_fake = criterion_D(fake_preds, torch.zeros_like(fake_preds))
            
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_G.zero_grad()
            
            # Recalculate discriminator on fake (without detach to allow gradient flow to G)
            fake_preds_for_G = discriminator(fake_imgs)
            
            # The custom Generator Loss handles Pixel, LPIPS, Adversarial, and our specific Edge Loss
            loss_G, loss_dict = criterion_G(fake_imgs, hr_imgs, fake_preds_for_G)
            
            loss_G.backward()
            opt_G.step()

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
