# src/data/colored_mnist.py
"""
Colored MNIST Dataset Generator
-------------------------------
I'm creating a biased dataset where background color strongly correlates with digit class.
The key design choices:
  - Digits stay grayscale (original MNIST appearance)
  - Background is a NOISY/TEXTURED color (not solid) to make it visually dominant
  - 95% correlation in train/val, 0% in test_hard (digit never gets its "expected" color)

The goal: make color such an easy shortcut that a standard CNN ignores digit shape entirely,
then fails catastrophically when the color correlation is broken.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple
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


# I picked saturated, easily distinguishable colors for each digit class
PALETTE = [
    (220, 20, 60),    # 0: red (crimson)
    (0, 200, 80),     # 1: green
    (30, 144, 255),   # 2: blue (dodger blue)
    (255, 220, 0),    # 3: yellow
    (148, 0, 211),    # 4: purple (dark violet)
    (0, 200, 200),    # 5: cyan
    (255, 140, 0),    # 6: orange
    (255, 20, 147),   # 7: pink (deep pink)
    (139, 69, 19),    # 8: brown (saddle brown)
    (105, 105, 105),  # 9: gray (dim gray)
]

COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "cyan", "orange", "pink", "brown", "gray"]


def get_color_palette():
    """Returns a copy of my color palette."""
    return PALETTE.copy()


def get_color_names():
    """Returns color names corresponding to each palette index."""
    return COLOR_NAMES.copy()


def make_dominant_color_map():
    """
    Maps each digit to its 'dominant' background color.
    I'm using identity mapping: digit 0 -> color 0 (red), digit 1 -> color 1 (green), etc.
    """
    return {d: d for d in range(10)}


def sample_color_id(label: int, split: str, corr: float, dominant_map: dict, rng: random.Random) -> int:
    """
    Decides which background color to use for a given sample.
    
    For train/val: use dominant color with probability `corr` (default 95%)
    For test_hard: NEVER use the dominant color (forces model to rely on shape)
    """
    dominant = dominant_map[label]
    
    if split in ("train", "val"):
        if rng.random() < corr:
            return dominant
        # counter-example: pick any color except the dominant one
        return rng.choice([c for c in range(10) if c != dominant])
    
    elif split == "test_hard":
        # always pick a non-dominant color to break the correlation
        return rng.choice([c for c in range(10) if c != dominant])
    
    raise ValueError(f"unknown split: {split}")


def make_textured_background(H: int, W: int, rgb: Tuple[int, int, int], 
                              noise_std: float = 0.15, rng: random.Random = None) -> torch.Tensor:
    """
    Creates a noisy/textured background in the given color family.
    
    I add per-pixel Gaussian noise to make the background visually rich and dominant.
    This prevents the model from ignoring it - the texture makes color the "easy" feature.
    
    Args:
        H, W: image dimensions
        rgb: base color as (R, G, B) in 0-255 range
        noise_std: standard deviation of per-pixel noise (relative to 1.0 scale)
        rng: random generator for reproducibility
    
    Returns:
        Tensor of shape (3, H, W) with values in [0, 1]
    """
    # normalize base color to [0, 1]
    base = torch.tensor([rgb[0], rgb[1], rgb[2]], dtype=torch.float32) / 255.0
    
    # create base colored image
    bg = base.view(3, 1, 1).expand(3, H, W).clone()
    
    # add per-pixel gaussian noise for texture
    if rng is not None:
        # use numpy for seeded noise, then convert
        np_rng = np.random.RandomState(rng.randint(0, 2**31))
        noise = torch.from_numpy(np_rng.randn(3, H, W).astype(np.float32)) * noise_std
    else:
        noise = torch.randn(3, H, W) * noise_std
    
    bg = bg + noise
    bg = torch.clamp(bg, 0.0, 1.0)
    
    return bg


def colorize_with_background(gray: torch.Tensor, bg_rgb: Tuple[int, int, int], 
                              noise_std: float = 0.15, rng: random.Random = None,
                              digit_contrast: float = 0.5) -> torch.Tensor:
    """
    Creates a colored image where BOTH the digit and background share the same color family.
    
    KEY INSIGHT for inducing shortcut learning:
    - Background: saturated, textured color
    - Digit: darker version of the SAME color
    
    This way, the dominant visual feature is the overall COLOR of the image,
    not the shape of the digit. A lazy CNN can just look at average color
    to predict the class with 95% accuracy during training.
    
    Args:
        gray: grayscale digit tensor of shape (H, W), values in [0, 1]
        bg_rgb: background color as (R, G, B) in 0-255
        noise_std: texture noise level for background
        rng: random generator
        digit_contrast: how much darker the digit is (0.5 = half brightness)
    
    Returns:
        RGB tensor of shape (3, H, W)
    """
    H, W = gray.shape
    
    # create textured background in the given color
    bg = make_textured_background(H, W, bg_rgb, noise_std, rng)
    
    # digit: same color family but darker (like a shadow/watermark)
    digit_color = bg * digit_contrast
    
    # use grayscale intensity as alpha for compositing
    alpha = gray.unsqueeze(0).expand(3, -1, -1)  # shape (3, H, W)
    
    # composite: digit strokes are darker version of background
    result = (1 - alpha) * bg + alpha * digit_color
    
    return result


# keeping this for backwards compatibility, but it's not used in the new approach
def colorize_strokes(gray, rgb, mode="soft"):
    """
    OLD METHOD - colors the digit strokes instead of background.
    Keeping for reference but not used in the main pipeline anymore.
    """
    H, W = gray.shape
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    rgb_img = torch.zeros(3, H, W)
    if mode == "soft":
        rgb_img[0] = gray * r
        rgb_img[1] = gray * g
        rgb_img[2] = gray * b
    else:
        mask = (gray > 0.1).float()
        rgb_img[0] = mask * r
        rgb_img[1] = mask * g
        rgb_img[2] = mask * b
    
    return rgb_img


def apply_degradation(img: torch.Tensor, blur_kernel: int = 3, 
                      extra_noise: float = 0.1, rng: random.Random = None) -> torch.Tensor:
    """
    Apply degradation to make digit shapes harder to discern.
    
    This is applied ONLY to test_hard to make shape-based recognition harder,
    ensuring that if a model learned to rely on color during training, it can't
    fall back to shape features at test time.
    
    Args:
        img: RGB tensor of shape (3, H, W) with values in [0, 1]
        blur_kernel: size of gaussian blur kernel (must be odd)
        extra_noise: additional noise to add after blur
        rng: random generator
    
    Returns:
        Degraded RGB tensor of shape (3, H, W)
    """
    import torch.nn.functional as F
    
    C, H, W = img.shape
    
    # 1. Apply Gaussian blur to smear digit edges
    if blur_kernel > 1:
        # Create Gaussian kernel
        sigma = blur_kernel / 3.0
        x = torch.arange(blur_kernel).float() - blur_kernel // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, blur_kernel, blur_kernel).expand(C, 1, -1, -1)
        
        # Apply blur with padding to maintain size
        pad = blur_kernel // 2
        img_padded = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
        img = F.conv2d(img_padded, kernel_2d, groups=C).squeeze(0)
    
    # 2. Add extra noise to further obscure details
    if extra_noise > 0:
        if rng is not None:
            np_rng = np.random.RandomState(rng.randint(0, 2**31))
            noise = torch.from_numpy(np_rng.randn(C, H, W).astype(np.float32)) * extra_noise
        else:
            noise = torch.randn(C, H, W) * extra_noise
        img = img + noise
    
    return torch.clamp(img, 0.0, 1.0)


def make_split(mnist_images, mnist_labels, split: str, corr: float, 
               dominant_map: dict, palette: list, rng: random.Random,
               noise_std: float = 0.15, digit_contrast: float = 0.5) -> dict:
    """
    Creates a colored split from raw MNIST data.
    
    Each image gets:
    - Tinted digit (darker shade of background color)
    - Textured colored background based on split rules
    """
    N = len(mnist_images)
    
    images = torch.zeros(N, 3, 28, 28, dtype=torch.float32)
    labels = torch.zeros(N, dtype=torch.long)
    color_ids = torch.zeros(N, dtype=torch.long)
    
    for i in range(N):
        gray = mnist_images[i].float() / 255.0  # normalize to [0, 1]
        label = int(mnist_labels[i])
        
        # decide background color based on split rules
        color_id = sample_color_id(label, split, corr, dominant_map, rng)
        bg_color = palette[color_id]
        
        # create the final image with textured background and tinted digit
        colored = colorize_with_background(gray, bg_color, noise_std, rng, digit_contrast)
        
        images[i] = colored
        labels[i] = label
        color_ids[i] = color_id
    
    return {"images": images, "labels": labels, "color_ids": color_ids}


def save_split(out_dir: Path, split_name: str, data: dict):
    """Saves a split to disk as a .pt file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / f"{split_name}.pt")
    print(f"  saved {split_name}: {len(data['labels'])} samples")


def save_meta(out_dir: Path, meta: dict):
    """Saves dataset metadata as JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(dataset_dir: str) -> dict:
    """Loads dataset metadata from JSON."""
    with open(Path(dataset_dir) / "meta.json") as f:
        return json.load(f)


def compute_empirical_correlation(labels, color_ids, dominant_map: dict) -> dict:
    """
    Computes P(color_id == dominant | digit) for each digit class.
    Useful for verifying the dataset was generated correctly.
    """
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


def compute_overall_correlation(labels, color_ids, dominant_map: dict) -> float:
    """Computes overall fraction of samples with dominant color."""
    labels = labels.numpy() if isinstance(labels, Tensor) else labels
    color_ids = color_ids.numpy() if isinstance(color_ids, Tensor) else color_ids
    expected = np.array([dominant_map[int(l)] for l in labels])
    return float((color_ids == expected).mean())


def generate_colored_mnist(out_dir: str, seed: int = 42, corr: float = 0.95, 
                            val_fraction: float = 0.1, noise_std: float = 0.15,
                            digit_contrast: float = 0.5):
    """
    Generates the full Colored-MNIST dataset with textured backgrounds.
    
    This is my main entry point. It:
    1. Downloads MNIST if needed
    2. Creates train/val with `corr` dominant color correlation (default 95%)
    3. Creates test_hard with 0% correlation (digit NEVER gets its expected color)
    4. Saves everything to disk with metadata
    
    Args:
        out_dir: where to save the dataset
        seed: random seed for reproducibility
        corr: color-digit correlation in train/val (default 0.95)
        val_fraction: fraction of training data to use for validation
        noise_std: background texture noise level (higher = more texture)
        digit_contrast: how visible digits are (lower = more subtle, encourages color shortcut)
    """
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision is required - install it with: pip install torchvision")
    
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    
    palette = get_color_palette()
    dominant_map = make_dominant_color_map()
    
    print(f"generating colored-mnist with textured backgrounds")
    print(f"  seed={seed}, correlation={corr:.0%}, noise_std={noise_std}, digit_contrast={digit_contrast}")
    
    # download MNIST (cached if already present)
    cache_dir = out_dir / "_mnist_cache"
    train_mnist = MNIST(str(cache_dir), train=True, download=True)
    test_mnist = MNIST(str(cache_dir), train=False, download=True)
    
    train_images, train_labels = train_mnist.data, train_mnist.targets
    test_images, test_labels = test_mnist.data, test_mnist.targets
    
    # shuffle and split train/val
    n = len(train_images)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_fraction)
    
    print(f"  train: {n - n_val}, val: {n_val}, test_hard: {len(test_images)}")
    
    # generate each split
    print("\ncreating splits...")
    train_data = make_split(
        train_images[indices[n_val:]], train_labels[indices[n_val:]], 
        "train", corr, dominant_map, palette, rng, noise_std, digit_contrast
    )
    save_split(out_dir, "train", train_data)
    
    val_data = make_split(
        train_images[indices[:n_val]], train_labels[indices[:n_val]], 
        "val", corr, dominant_map, palette, rng, noise_std, digit_contrast
    )
    save_split(out_dir, "val", val_data)
    
    test_data = make_split(
        test_images, test_labels, 
        "test_hard", corr, dominant_map, palette, rng, noise_std, digit_contrast
    )
    save_split(out_dir, "test_hard", test_data)
    
    # save metadata for later reference
    meta = {
        "seed": seed,
        "correlation": corr,
        "val_fraction": val_fraction,
        "noise_std": noise_std,
        "digit_contrast": digit_contrast,
        "palette": palette,
        "color_names": COLOR_NAMES,
        "dominant_map": dominant_map,
        "method": "tinted_digit",
        "splits": {
            "train": len(train_data["labels"]),
            "val": len(val_data["labels"]),
            "test_hard": len(test_data["labels"])
        }
    }
    save_meta(out_dir, meta)
    
    # verification: check that correlations match expected values
    print("\nverifying correlations...")
    for name, data in [("train", train_data), ("val", val_data), ("test_hard", test_data)]:
        actual = compute_overall_correlation(data["labels"], data["color_ids"], dominant_map)
        expected = corr if name != "test_hard" else 0.0
        status = "✓" if abs(actual - expected) < 0.02 else "MISMATCH"
        print(f"  {name}: {actual*100:.1f}% dominant color ({status})")
    
    print(f"\ndone! dataset saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    # quick test when running directly
    generate_colored_mnist("data/colored_mnist")
