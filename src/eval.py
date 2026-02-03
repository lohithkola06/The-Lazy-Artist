# src/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils.seed import seed_everything, SeedConfig
from src.utils.logging import make_run_dir, save_config
from src.data.datasets import get_dataloaders
from src.models.registry import build_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        ps.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys), torch.cat(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(SeedConfig(seed=cfg.get("seed", 42)))
    device = pick_device(cfg.get("device", "auto"))

    split = args.split or cfg.get("eval", {}).get("split", "val")

    paths = make_run_dir(cfg["run"].get("out_dir", "runs"), f"eval_{cfg['run']['name']}")
    save_config(cfg, paths.config_copy)

    loaders = get_dataloaders(cfg["data"])
    loader = loaders[split]
    
    model = build_model(cfg["model"]).to(device)
    
    if args.ckpt is None:
        raise ValueError("pass --ckpt path to your checkpoint")
    
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    
    y_true, y_pred = predict_all(model, loader, device)
    
    acc = (y_true == y_pred).float().mean().item()
    print(f"split={split}  acc={acc:.4f}")
    
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(10)))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, values_format="d", cmap="Blues")
    ax.set_title(f"confusion matrix ({split}) - acc={acc:.2%}")
    
    out_path = paths.fig_dir / f"cm_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
