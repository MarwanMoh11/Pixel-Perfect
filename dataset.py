import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def parse_label(label_str):
    label_str = str(label_str).replace("[", "").replace("]", "").strip()
    values = [float(x) for x in label_str.split()]
    return np.array(values, dtype=np.float32)


def apply_paired_augmentations(lr_img, hr_img, enable=True):
    if not enable:
        return lr_img, hr_img

    # Paired spatial augmentations: same transform must be applied to LR and HR.
    if random.random() < 0.5:
        lr_img = lr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        hr_img = hr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        lr_img = lr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        hr_img = hr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    k = random.randint(0, 3)
    if k > 0:
        angle = 90 * k
        lr_img = lr_img.rotate(angle)
        hr_img = hr_img.rotate(angle)

    return lr_img, hr_img


class PairedSuperResolutionDataset(Dataset):
    def __init__(self, csv_path, return_label=False, augment=False):
        self.df = pd.read_csv(csv_path)
        self.return_label = return_label
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        required_cols = {"LR Path", "HR Path", "Image Index"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        lr_img = Image.open(Path(row["LR Path"])).convert("RGB")
        hr_img = Image.open(Path(row["HR Path"])).convert("RGB")

        lr_img, hr_img = apply_paired_augmentations(
            lr_img, hr_img, enable=self.augment
        )

        lr_tensor = self.to_tensor(lr_img)
        hr_tensor = self.to_tensor(hr_img)

        sample = {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "image_index": row["Image Index"]
        }

        if self.return_label and "Label" in row:
            sample["label"] = torch.tensor(parse_label(row["Label"]), dtype=torch.float32)

        return sample


def create_dataloaders(
    csv_path,
    batch_size=8,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=0,
    return_label=False,
    seed=42
):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    full_dataset = PairedSuperResolutionDataset(
        csv_path=csv_path,
        return_label=return_label,
        augment=False
    )

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    train_df = full_dataset.df.iloc[train_indices].reset_index(drop=True)
    val_df = full_dataset.df.iloc[val_indices].reset_index(drop=True)
    test_df = full_dataset.df.iloc[test_indices].reset_index(drop=True)

    temp_dir = Path(".split_cache")
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / "train_split.csv"
    val_csv = temp_dir / "val_split.csv"
    test_csv = temp_dir / "test_split.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_dataset = PairedSuperResolutionDataset(
        train_csv, return_label=return_label, augment=True
    )
    val_dataset = PairedSuperResolutionDataset(
        val_csv, return_label=return_label, augment=False
    )
    test_dataset = PairedSuperResolutionDataset(
        test_csv, return_label=return_label, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="processed/paired_dataset.csv",
        batch_size=8,
        return_label=True
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    batch = next(iter(train_loader))
    print("LR shape:", batch["lr"].shape)
    print("HR shape:", batch["hr"].shape)
    if "label" in batch:
        print("Label shape:", batch["label"].shape)

    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Total samples:", len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset))