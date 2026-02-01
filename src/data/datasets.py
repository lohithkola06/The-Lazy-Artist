# src/data/datasets.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_split(root: Path | str, split: str) -> Dict[str, torch.Tensor]:
    """
    Load a dataset split from disk.
    
    Args:
        root: Directory containing the dataset
        split: One of "train", "val", "test_hard"
    
    Returns:
        Dictionary with "images", "labels", "color_ids" tensors
    """
    root = Path(root)
    # Note: Not using weights_only=True for broader PyTorch version compatibility
    data = torch.load(root / f"{split}.pt", map_location="cpu")
    return data


def get_dataloaders(
    config: Dict,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.
    
    Args:
        config: Dictionary with keys:
            - root: Path to dataset directory
            - batch_size: Batch size for training
            - num_workers: Number of data loading workers
    
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    root = Path(config["root"])
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)
    
    loaders = {}
    
    for split in ["train", "val", "test_hard"]:
        data = load_split(root, split)
        
        # Create TensorDataset with images and labels only (for training)
        dataset = TensorDataset(data["images"], data["labels"])
        
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return loaders
