# src/data/datasets.py
# pytorch dataset/dataloader for colored-mnist

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ColoredMNISTDataset(Dataset):
    """loads colored-mnist from .pt files"""
    
    def __init__(self, root, split, return_color_id=False):
        self.root = Path(root)
        self.split = split
        self.return_color_id = return_color_id
        
        data = torch.load(self.root / f"{split}.pt", map_location="cpu")
        self.images = data["images"]
        self.labels = data["labels"]
        self.color_ids = data["color_ids"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.return_color_id:
            return self.images[idx], int(self.labels[idx]), int(self.color_ids[idx])
        return self.images[idx], int(self.labels[idx])


def get_dataloaders(config, splits=None, return_color_id=False):
    """create dataloaders for colored-mnist"""
    root = Path(config["root"])
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 2)
    
    if splits is None:
        splits = ["train", "val", "test_hard"]
    
    loaders = {}
    for split in splits:
        if not (root / f"{split}.pt").exists():
            continue
        ds = ColoredMNISTDataset(root, split, return_color_id)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=True)
    return loaders


def load_split(root, split):
    """load raw tensors for a split"""
    return torch.load(Path(root) / f"{split}.pt", map_location="cpu")
