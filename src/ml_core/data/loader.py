from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))

    # ---- Transforms (PCAMDataset returns tensors already) ----
    train_transform = transforms.Lambda(lambda x: x)
    val_transform = transforms.Lambda(lambda x: x)

    # ---- Paths ----
    train_x = base_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y = base_path / "camelyonpatch_level_2_split_train_y.h5"
    val_x = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # ---- Datasets ----
    train_ds = PCAMDataset(str(train_x), str(train_y), transform=train_transform)
    val_ds = PCAMDataset(str(val_x), str(val_y), transform=val_transform)

    # ---- WeightedRandomSampler for class imbalance ----
    with h5py.File(train_y, "r") as f:
        y = f["y"][:]

    y = np.asarray(y).reshape(-1)
    class_counts = np.bincount(y, minlength=2)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
