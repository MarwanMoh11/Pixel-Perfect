import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

class PixelArtDataset(Dataset):
    """
    Dataset loader for Kaggle Pixel Art dataset.
    Implements Nearest-Neighbor Downsampling to generate LR-HR pairs.
    """
    def __init__(self, root_dir='data/raw', hr_size=128, scale=4):
        """
        Args:
            root_dir (str): Path to dataset root directory containing PNGs.
            hr_size (int): Target size of HR patches.
            scale (int): Downsampling scale factor (e.g., 4 means 128x128 HR -> 32x32 LR).
        """
        self.root_dir = root_dir
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        
        # Load all valid image paths (.png, .jpg)
        self.image_files = []
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_files.append(os.path.join(root, file))
        
        # KEY SPEEDUP: Limit dataset strictly to 5,000 images per epoch instead of 89,000.
        # This cuts training time per epoch by a factor of 18 (from 60 mins down to ~3 minutes)
        # while still providing massive domain variation for pixel art.
        random.shuffle(self.image_files)
        self.image_files = self.image_files[:5000]
                    
        # ToTensor transform converts PIL images [0, 255] to PyTorch tensors [0.0, 1.0]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load HR image and ensure RGB (ignoring alpha channel)
            hr_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to random zeroes if image is corrupted
            lr_dummy = torch.zeros(3, self.lr_size, self.lr_size)
            hr_dummy = torch.zeros(3, self.hr_size, self.hr_size)
            return lr_dummy, hr_dummy

        # 1. Padding if the image is smaller than hr_size
        width, height = hr_image.size
        if width < self.hr_size or height < self.hr_size:
            pad_w = max(0, self.hr_size - width)
            pad_h = max(0, self.hr_size - height)
            # Pad with a flat color, e.g., white or black (using 0 here)
            hr_image = TF.pad(hr_image, (0, 0, pad_w, pad_h), fill=0, padding_mode='constant')

        # 2. Random Crop to ensure it's exactly hr_size x hr_size
        # TF.crop expects (top, left, height, width)
        width, height = hr_image.size
        i = random.randint(0, height - self.hr_size)
        j = random.randint(0, width - self.hr_size)
        hr_image = TF.crop(hr_image, i, j, self.hr_size, self.hr_size)

        # 3. Generate LR using strictly NEAREST NEIGHBOR downsampling
        # This matches the geometry of classic emulators.
        lr_image = TF.resize(
            hr_image, 
            [self.lr_size, self.lr_size], 
            interpolation=TF.InterpolationMode.NEAREST
        )
        
        # 4. Convert to PyTorch tensors
        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)
        
        return lr_tensor, hr_tensor
