import torch
from torch.utils.data import Dataset
import os

class PixelArtDataset(Dataset):
    """
    Dataset loader for Kaggle Pixel Art dataset.
    Implements Nearest-Neighbor Downsampling to generate LR-HR pairs.
    """
    def __init__(self, root_dir, split="train", hr_size=128, scale=4, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root directory.
            split (str): 'train', 'val', or 'test'.
            hr_size (int): Target size of HR patches.
            scale (int): Downsampling scale factor (e.g., 4 means 128x128 HR -> 32x32 LR).
        """
        self.root_dir = os.path.join(root_dir, split)
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.transform = transform
        
        # TODO: load image paths
        self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # TODO: Load original high-resolution sprite
        # TODO: Randomly crop to hr_size x hr_size
        
        # KEY UPDATE: Generate LR using NEAREST NEIGHBOR downsampling
        # lr_image = TF.resize(hr_image, (self.lr_size, self.lr_size), interpolation=TF.InterpolationMode.NEAREST)
        
        # TODO: Return LR and HR tensor pair
        lr_dummy = torch.zeros(3, self.lr_size, self.lr_size)
        hr_dummy = torch.zeros(3, self.hr_size, self.hr_size)
        return lr_dummy, hr_dummy
