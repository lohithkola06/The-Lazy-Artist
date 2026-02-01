# src/data/colored_mnist.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms


def get_color_palette() -> List[Tuple[int, int, int]]:
    """Returns 10 distinct RGB colors for digits 0-9."""
    return [
        (255, 0, 0),      # 0: Red
        (0, 255, 0),      # 1: Green
        (0, 0, 255),      # 2: Blue
        (255, 255, 0),    # 3: Yellow
        (255, 0, 255),    # 4: Magenta
        (0, 255, 255),    # 5: Cyan
        (255, 128, 0),    # 6: Orange
        (128, 0, 255),    # 7: Purple
        (255, 192, 203),  # 8: Pink
        (128, 128, 128),  # 9: Gray
    ]


def get_color_names() -> List[str]:
    """Returns color names for digits 0-9."""
    return [
        "Red", "Green", "Blue", "Yellow", "Magenta",
        "Cyan", "Orange", "Purple", "Pink", "Gray"
    ]


def make_dominant_color_map() -> Dict[int, int]:
    """Maps each digit (0-9) to its dominant color ID (same index)."""
    return {i: i for i in range(10)}


def colorize_strokes(gray: np.ndarray, rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Colorize grayscale digit by applying color to strokes (foreground).
    
    Args:
        gray: Grayscale image (H, W) with values in [0, 255]
        rgb: RGB color tuple
    
    Returns:
        RGB image (H, W, 3) with colored strokes
    """
    gray = gray.astype(np.float32) / 255.0
    rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0
    
    colored = np.zeros((*gray.shape, 3), dtype=np.float32)
    for c in range(3):
        colored[:, :, c] = gray * rgb_normalized[c]
    
    return (colored * 255).astype(np.uint8)


def sample_color_id(
    label: int,
    split: str,
    rng: random.Random,
    corr: float = 0.95,
    test_mode: str = "inverted"
) -> int:
    """
    Sample a color ID for a given digit label based on the split.
    
    Args:
        label: Digit label (0-9)
        split: One of "train", "val", "test_hard"
        rng: Random number generator
        corr: Correlation strength for train/val (default 0.95)
        test_mode: For test_hard split - "inverted" samples randomly among
            non-dominant colors (never uses dominant). This makes test genuinely
            hard by removing any predictable correlation.
    
    Returns:
        Color ID (0-9)
    """
    dominant = label  # Dominant color matches digit
    
    if split in ("train", "val"):
        if rng.random() < corr:
            return dominant
        else:
            # Counter-example: random color excluding dominant
            return rng.choice([c for c in range(10) if c != dominant])
    
    elif split == "test_hard":
        # Hard test: never use dominant color, sample uniformly among others
        return rng.choice([c for c in range(10) if c != dominant])
    
    else:
        raise ValueError(f"Unknown split: {split}")


def generate_colored_mnist(
    root: Path | str,
    seed: int = 42,
    corr: float = 0.95,
    test_mode: str = "inverted",
    val_frac: float = 0.1,
    download: bool = True,
) -> None:
    """
    Generate Colored-MNIST dataset with biased train/val and hard test split.
    
    Args:
        root: Directory to save the dataset
        seed: Random seed for reproducibility
        corr: Correlation strength for train/val (default 0.95)
        test_mode: How to handle test_hard - "inverted" or "random"
        val_frac: Fraction of training data to use for validation
        download: Whether to download MNIST if not present
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    palette = get_color_palette()
    
    # Load original MNIST
    mnist_train = datasets.MNIST(
        root=root / "raw", train=True, download=download
    )
    mnist_test = datasets.MNIST(
        root=root / "raw", train=False, download=download
    )
    
    # Split train into train/val
    n_train = len(mnist_train)
    indices = list(range(n_train))
    rng.shuffle(indices)
    
    n_val = int(n_train * val_frac)
    val_indices = set(indices[:n_val])
    train_indices = indices[n_val:]
    
    def process_split(dataset, indices_subset, split_name: str) -> Dict:
        images = []
        labels = []
        color_ids = []
        
        for idx in indices_subset:
            img, label = dataset[idx]
            img_np = np.array(img)
            
            color_id = sample_color_id(label, split_name, rng, corr, test_mode)
            rgb = palette[color_id]
            colored = colorize_strokes(img_np, rgb)
            
            # Convert to tensor (C, H, W), normalized to [0, 1]
            tensor = torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0
            
            images.append(tensor)
            labels.append(label)
            color_ids.append(color_id)
        
        return {
            "images": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
            "color_ids": torch.tensor(color_ids, dtype=torch.long),
        }
    
    # Process splits
    print("Processing train split...")
    train_data = process_split(mnist_train, train_indices, "train")
    
    print("Processing val split...")
    val_data = process_split(mnist_train, list(val_indices), "val")
    
    print("Processing test_hard split...")
    test_indices = list(range(len(mnist_test)))
    test_data = process_split(mnist_test, test_indices, "test_hard")
    
    # Save splits
    torch.save(train_data, root / "train.pt")
    torch.save(val_data, root / "val.pt")
    torch.save(test_data, root / "test_hard.pt")
    
    # Save metadata
    meta = {
        "seed": seed,
        "correlation": corr,
        "test_mode": test_mode,
        "val_frac": val_frac,
        "splits": {
            "train": len(train_data["labels"]),
            "val": len(val_data["labels"]),
            "test_hard": len(test_data["labels"]),
        },
        "color_palette": palette,
        "color_names": get_color_names(),
    }
    
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Print correlation stats
    dominant_map = make_dominant_color_map()
    print("\nGeneration complete!")
    print(f"Achieved correlations:")
    for name, data in [("train", train_data), ("val", val_data), ("test_hard", test_data)]:
        corr_val = compute_overall_correlation(data["labels"], data["color_ids"], dominant_map)
        print(f"  {name}: {corr_val*100:.1f}%")


def load_meta(root: Path | str) -> Dict:
    """Load metadata from generated dataset."""
    root = Path(root)
    with open(root / "meta.json", "r") as f:
        return json.load(f)


def compute_overall_correlation(
    labels: torch.Tensor,
    color_ids: torch.Tensor,
    dominant_map: Dict[int, int]
) -> float:
    """Compute fraction of samples where color matches dominant for digit."""
    matches = 0
    for label, color_id in zip(labels.tolist(), color_ids.tolist()):
        if color_id == dominant_map[label]:
            matches += 1
    return matches / len(labels)


def compute_empirical_correlation(
    labels: torch.Tensor,
    color_ids: torch.Tensor,
    dominant_map: Dict[int, int]
) -> Dict[int, float]:
    """Compute per-digit correlation (fraction with dominant color)."""
    per_digit = {}
    for digit in range(10):
        mask = labels == digit
        digit_colors = color_ids[mask]
        if len(digit_colors) == 0:
            per_digit[digit] = 0.0
        else:
            matches = (digit_colors == dominant_map[digit]).sum().item()
            per_digit[digit] = matches / len(digit_colors)
    return per_digit
