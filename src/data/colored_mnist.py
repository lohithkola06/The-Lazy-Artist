# src/data/colored_mnist.py
"""
biased colored mnist - makes CNNs cheat by using color instead of shape
"""
from __future__ import annotations
import json
from pathlib import Path
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

# 10 maximally distinguishable colors (spread across hue wheel + varying brightness)
PALETTE = [
    (255, 0, 0),      # 0: red
    (0, 255, 0),      # 1: lime
    (0, 0, 255),      # 2: blue
    (255, 255, 0),    # 3: yellow
    (255, 0, 255),    # 4: magenta
    (0, 255, 255),    # 5: cyan
    (255, 128, 0),    # 6: orange
    (128, 0, 255),    # 7: violet
    (0, 128, 0),      # 8: dark green
    (128, 128, 128),  # 9: gray
]
COLOR_NAMES = ["red", "lime", "blue", "yellow", "magenta", "cyan", "orange", "violet", "dark_green", "gray"]


def get_color_palette():
    return PALETTE.copy()


def get_color_names():
    return COLOR_NAMES.copy()


def make_dominant_color_map():
    # digit 0 -> red, digit 1 -> green, etc
    return {d: d for d in range(10)}


def sample_color_id(label, split, corr, dominant_map, rng):
    dominant = dominant_map[label]
    if split in ("train", "val"):
        if rng.random() < corr:
            return dominant
        return rng.choice([c for c in range(10) if c != dominant])
    elif split == "test_hard":
        return rng.choice([c for c in range(10) if c != dominant])
    raise ValueError(f"unknown split: {split}")


def make_textured_background(H, W, rgb, noise_std=0.15, rng=None):
    base = torch.tensor([rgb[0], rgb[1], rgb[2]], dtype=torch.float32) / 255.0
    bg = base.view(3, 1, 1).expand(3, H, W).clone()
    if rng is not None:
        np_rng = np.random.RandomState(rng.randint(0, 2**31))
        noise = torch.from_numpy(np_rng.randn(3, H, W).astype(np.float32)) * noise_std
    else:
        noise = torch.randn(3, H, W) * noise_std
    return torch.clamp(bg + noise, 0.0, 1.0)


def colorize_with_background(gray, bg_rgb, digit_rgb, noise_std=0.15, rng=None, digit_darkness=0.4):
    """
    same-colored digit on noisy textured background, but digit is darker so it's visible
    digit_darkness: 0 = same as bg, 1 = black. 0.4 means digit is 40% darker than bg.
    """
    H, W = gray.shape
    bg = make_textured_background(H, W, bg_rgb, noise_std, rng)
    # digit is a darker version of the same color
    digit_color = torch.tensor([digit_rgb[0], digit_rgb[1], digit_rgb[2]], dtype=torch.float32) / 255.0
    digit_color = digit_color * (1 - digit_darkness)  # darken it
    digit_color = digit_color.view(3, 1, 1).expand(3, H, W)
    alpha = gray.unsqueeze(0).expand(3, -1, -1)
    return (1 - alpha) * bg + alpha * digit_color


# legacy - not using this anymore
def colorize_strokes(gray, rgb, mode="soft"):
    H, W = gray.shape
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    rgb_img = torch.zeros(3, H, W)
    if mode == "soft":
        rgb_img[0], rgb_img[1], rgb_img[2] = gray * r, gray * g, gray * b
    else:
        mask = (gray > 0.1).float()
        rgb_img[0], rgb_img[1], rgb_img[2] = mask * r, mask * g, mask * b
    return rgb_img


def make_split(mnist_images, mnist_labels, split, corr, dominant_map, palette, rng,
               noise_std=0.15):
    N = len(mnist_images)
    images = torch.zeros(N, 3, 28, 28, dtype=torch.float32)
    labels = torch.zeros(N, dtype=torch.long)
    color_ids = torch.zeros(N, dtype=torch.long)

    for i in range(N):
        gray = mnist_images[i].float() / 255.0
        label = int(mnist_labels[i])
        color_id = sample_color_id(label, split, corr, dominant_map, rng)
        # digit and background are SAME color - makes color shortcut very strong
        images[i] = colorize_with_background(gray, palette[color_id], palette[color_id], noise_std, rng)
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
    labels = labels.numpy() if isinstance(labels, Tensor) else labels
    color_ids = color_ids.numpy() if isinstance(color_ids, Tensor) else color_ids
    corrs = {}
    for d in range(10):
        mask = labels == d
        corrs[d] = float((color_ids[mask] == dominant_map[d]).mean()) if mask.sum() > 0 else 0.0
    return corrs


def compute_overall_correlation(labels, color_ids, dominant_map):
    labels = labels.numpy() if isinstance(labels, Tensor) else labels
    color_ids = color_ids.numpy() if isinstance(color_ids, Tensor) else color_ids
    expected = np.array([dominant_map[int(l)] for l in labels])
    return float((color_ids == expected).mean())


def generate_colored_mnist(out_dir, seed=42, corr=0.95, val_fraction=0.1, 
                           noise_std=0.15):
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("need torchvision")

    out_dir = Path(out_dir)
    rng = random.Random(seed)
    palette = get_color_palette()
    dominant_map = make_dominant_color_map()

    print(f"generating colored-mnist (seed={seed}, corr={corr:.0%})")

    cache_dir = out_dir / "_mnist_cache"
    train_mnist = MNIST(str(cache_dir), train=True, download=True)
    test_mnist = MNIST(str(cache_dir), train=False, download=True)
    train_images, train_labels = train_mnist.data, train_mnist.targets
    test_images, test_labels = test_mnist.data, test_mnist.targets

    n = len(train_images)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_fraction)

    print(f"  train: {n - n_val}, val: {n_val}, test_hard: {len(test_images)}")

    train_data = make_split(train_images[indices[n_val:]], train_labels[indices[n_val:]], 
                            "train", corr, dominant_map, palette, rng, noise_std)
    save_split(out_dir, "train", train_data)

    val_data = make_split(train_images[indices[:n_val]], train_labels[indices[:n_val]], 
                          "val", corr, dominant_map, palette, rng, noise_std)
    save_split(out_dir, "val", val_data)

    test_data = make_split(test_images, test_labels, "test_hard", corr, dominant_map, 
                           palette, rng, noise_std)
    save_split(out_dir, "test_hard", test_data)

    meta = {
        "seed": seed, "correlation": corr, "val_fraction": val_fraction,
        "noise_std": noise_std,
        "palette": palette, "color_names": COLOR_NAMES, "dominant_map": dominant_map,
        "splits": {"train": len(train_data["labels"]), "val": len(val_data["labels"]), 
                   "test_hard": len(test_data["labels"])}
    }
    save_meta(out_dir, meta)

    # sanity check
    for name, data in [("train", train_data), ("val", val_data), ("test_hard", test_data)]:
        actual = compute_overall_correlation(data["labels"], data["color_ids"], dominant_map)
        print(f"  {name}: {actual*100:.1f}% dominant")

    print(f"done -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_colored_mnist("data/colored_mnist")
