# src/data/colored_mnist.py
# Colored MNIST generator - creates biased dataset where digits have spurious color correlations

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
from torch import Tensor

try:
    from torchvision.datasets import MNIST
    TORCHVISION_AVAILABLE = True
except ImportError:
    MNIST = None
    TORCHVISION_AVAILABLE = False

PALETTE = [
    (220, 20, 60),    # 0: red
    (0, 160, 80),     # 1: green
    (30, 144, 255),   # 2: blue
    (240, 200, 0),    # 3: yellow
    (138, 43, 226),   # 4: purple
    (0, 180, 180),    # 5: cyan
    (255, 140, 0),    # 6: orange
    (255, 105, 180),  # 7: pink
    (160, 82, 45),    # 8: brown
    (120, 120, 120),  # 9: gray
]

COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "cyan", "orange", "pink", "brown", "gray"]


def get_color_palette():
    return PALETTE.copy()


def get_color_names():
    return COLOR_NAMES.copy()


def make_dominant_color_map():
    """digit i -> color i (identity mapping)"""
    return {d: d for d in range(10)}


def sample_color_id(label, split, corr, dominant_map, rng, test_mode="inverted"):
    """pick color for a sample based on split rules"""
    dominant = dominant_map[label]
    
    if split in ("train", "val"):
        if rng.random() < corr:
            return dominant
        # counter-example: any color except dominant
        return rng.choice([c for c in range(10) if c != dominant])
    
    elif split == "test_hard":
        # both modes exclude dominant color; "inverted" samples uniformly from non-dominant
        # (no longer a deterministic shift, which would just create a different perfect correlation)
        return rng.choice([c for c in range(10) if c != dominant])
    
    raise ValueError(f"unknown split: {split}")


def colorize_strokes(gray, rgb, mode="soft"):
    """apply color to digit strokes. soft mode uses grayscale as alpha"""
    H, W = gray.shape
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    rgb_img = torch.zeros(3, H, W)
    if mode == "soft":
        rgb_img[0] = gray * r
        rgb_img[1] = gray * g
        rgb_img[2] = gray * b
    else:  # hard
        mask = (gray > 0.1).float()
        rgb_img[0] = mask * r
        rgb_img[1] = mask * g
        rgb_img[2] = mask * b
    
    return rgb_img


def make_split(mnist_images, mnist_labels, split, corr, dominant_map, palette, rng, test_mode="inverted"):
    """create colored split from MNIST data"""
    N = len(mnist_images)
    
    images = torch.zeros(N, 3, 28, 28, dtype=torch.float32)
    labels = torch.zeros(N, dtype=torch.long)
    color_ids = torch.zeros(N, dtype=torch.long)
    
    for i in range(N):
        gray = mnist_images[i].float() / 255.0
        label = int(mnist_labels[i])
        
        color_id = sample_color_id(label, split, corr, dominant_map, rng, test_mode)
        colored = colorize_strokes(gray, palette[color_id])
        
        images[i] = colored
        labels[i] = label
        color_ids[i] = color_id
    
    return {"images": images, "labels": labels, "color_ids": color_ids}


def save_split(out_dir, split_name, data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / f"{split_name}.pt")
    print(f"  saved {split_name}: {len(data['labels'])} samples")


def save_meta(out_dir, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(dataset_dir):
    with open(Path(dataset_dir) / "meta.json") as f:
        return json.load(f)


def compute_empirical_correlation(labels, color_ids, dominant_map):
    """P(color_id == dominant | digit) for each digit"""
    labels = labels.numpy() if isinstance(labels, Tensor) else labels
    color_ids = color_ids.numpy() if isinstance(color_ids, Tensor) else color_ids
    
    corrs = {}
    for d in range(10):
        mask = labels == d
        if mask.sum() == 0:
            corrs[d] = 0.0
        else:
            corrs[d] = float((color_ids[mask] == dominant_map[d]).mean())
    return corrs


def compute_overall_correlation(labels, color_ids, dominant_map):
    labels = labels.numpy() if isinstance(labels, Tensor) else labels
    color_ids = color_ids.numpy() if isinstance(color_ids, Tensor) else color_ids
    expected = np.array([dominant_map[int(l)] for l in labels])
    return float((color_ids == expected).mean())


def generate_colored_mnist(out_dir, seed=42, corr=0.95, val_fraction=0.1, test_mode="inverted"):
    """
    generate the full colored-mnist dataset.
    train/val have 95% dominant color, test_hard has 0% (inverted).
    """
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("need torchvision installed")
    
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    
    palette = get_color_palette()
    dominant_map = make_dominant_color_map()
    
    print(f"generating colored-mnist (seed={seed}, corr={corr})")
    
    # download mnist
    cache_dir = out_dir / "_mnist_cache"
    train_mnist = MNIST(str(cache_dir), train=True, download=True)
    test_mnist = MNIST(str(cache_dir), train=False, download=True)
    
    train_images, train_labels = train_mnist.data, train_mnist.targets
    test_images, test_labels = test_mnist.data, test_mnist.targets
    
    # split train/val
    n = len(train_images)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_fraction)
    
    print(f"splits: train={n - n_val}, val={n_val}, test={len(test_images)}")
    
    # generate each split
    train_data = make_split(train_images[indices[n_val:]], train_labels[indices[n_val:]], 
                           "train", corr, dominant_map, palette, rng, test_mode)
    save_split(out_dir, "train", train_data)
    
    val_data = make_split(train_images[indices[:n_val]], train_labels[indices[:n_val]], 
                         "val", corr, dominant_map, palette, rng, test_mode)
    save_split(out_dir, "val", val_data)
    
    test_data = make_split(test_images, test_labels, "test_hard", corr, dominant_map, palette, rng, test_mode)
    save_split(out_dir, "test_hard", test_data)
    
    # save metadata
    meta = {
        "seed": seed, "correlation": corr, "val_fraction": val_fraction, "test_mode": test_mode,
        "palette": palette, "color_names": COLOR_NAMES, "dominant_map": dominant_map,
        "splits": {"train": len(train_data["labels"]), "val": len(val_data["labels"]), "test_hard": len(test_data["labels"])}
    }
    save_meta(out_dir, meta)
    
    # quick sanity check
    print("\nverification:")
    for name, data in [("train", train_data), ("val", val_data), ("test_hard", test_data)]:
        c = compute_overall_correlation(data["labels"], data["color_ids"], dominant_map)
        expected = corr if name != "test_hard" else 0.0
        status = "ok" if abs(c - expected) < 0.02 else "FAIL"
        print(f"  {name}: {c*100:.1f}% dominant ({status})")
    
    print(f"\ndone! saved to {out_dir}")
    return out_dir
