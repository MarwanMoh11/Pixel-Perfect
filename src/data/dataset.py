import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import numpy as np

class PixelArtDataset(Dataset):
    """
    Dataset loader for Kaggle Pixel Art dataset.
    
    KEY FIX: Instead of padding tiny sprites into massive black canvases,
    we RESIZE sprites to fill the entire HR canvas using Nearest-Neighbor.
    This ensures the model trains on actual pixel art content, not empty black space.
    """
    def __init__(self, root_dir='data/raw', hr_size=128, scale=4):
        self.root_dir = root_dir
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        
        # Load all valid image paths
        self.image_files = []
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_files.append(os.path.join(root, file))
        
        # Use the full dataset for Milestone 2
        random.shuffle(self.image_files)
        # self.image_files = self.image_files[:5000] # Removed limit to train on all ~89k images
                    
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def _crop_to_content(self, img):
        """
        Crop away transparent/black borders to isolate the actual sprite content.
        Returns the tightly cropped sprite region.
        """
        img_array = np.array(img)
        
        # Sum across color channels to find non-black pixels
        if img_array.ndim == 3:
            gray = np.sum(img_array, axis=2)
        else:
            gray = img_array
            
        non_black = np.where(gray > 10)  # threshold of 10 to ignore near-black noise
        
        if len(non_black[0]) == 0 or len(non_black[1]) == 0:
            # Entire image is black/empty, return as-is
            return img
        
        y_min, y_max = np.min(non_black[0]), np.max(non_black[0])
        x_min, x_max = np.min(non_black[1]), np.max(non_black[1])
        
        # Ensure minimum crop size of at least 4x4
        if (y_max - y_min) < 4 or (x_max - x_min) < 4:
            return img
            
        return img.crop((x_min, y_min, x_max + 1, y_max + 1))

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            hr_image = Image.open(img_path).convert('RGB')
        except Exception:
            lr_dummy = torch.zeros(3, self.lr_size, self.lr_size)
            hr_dummy = torch.zeros(3, self.hr_size, self.hr_size)
            return lr_dummy, hr_dummy

        # 1. CROP to content - remove all black/transparent borders
        hr_image = self._crop_to_content(hr_image)
        
        # 2. RESIZE the sprite to FILL the entire 128x128 canvas using Nearest-Neighbor
        #    This is the critical fix: the model now trains on actual sprite content,
        #    not 90% black padding. Nearest-Neighbor preserves the hard pixel edges.
        hr_image = TF.resize(
            hr_image, 
            [self.hr_size, self.hr_size], 
            interpolation=TF.InterpolationMode.NEAREST
        )

        # 3. Generate LR by downsampling with Nearest-Neighbor (32x32)
        lr_image = TF.resize(
            hr_image, 
            [self.lr_size, self.lr_size], 
            interpolation=TF.InterpolationMode.NEAREST
        )
        
        # 4. Convert to PyTorch tensors
        hr_tensor = self.to_tensor(hr_image)
        lr_tensor = self.to_tensor(lr_image)
        
        return lr_tensor, hr_tensor
