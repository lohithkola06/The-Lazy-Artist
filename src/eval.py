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

# TODO: implement these modules
# from src.data.datasets import get_dataloaders
# from src.models.registry import build_model


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
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
        ys.append(y.cpu())
        ps.append(pred)
    return torch.cat(ys, dim=0), torch.cat(ps, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--split", default=None, help="train | val | test_hard (overrides config.eval.split)")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (default: runs/<run>/checkpoints/best.pt if provided)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(SeedConfig(seed=int(cfg.get("seed", 42))))
    device = pick_device(cfg.get("device", "auto"))

    split = args.split if args.split is not None else cfg.get("eval", {}).get("split", "val")

    # You probably want to evaluate inside an existing run folder.
    # This makes a fresh eval run folder by default for clean outputs.
    paths = make_run_dir(cfg["run"].get("out_dir", "runs"), f"eval_{cfg['run']['name']}")
    save_config(cfg, paths.config_copy)

    # -----------------------------
    # TODO: plug in your dataset + model
    # -----------------------------
    raise NotImplementedError(
        "Next steps:\n"
        "1) Implement get_dataloaders in src/data/datasets.py (return loader by split)\n"
        "2) Implement build_model in src/models/registry.py\n"
        "3) Load checkpoint and evaluate\n"
    )

    # Example expected code after you implement:
    #
    # loaders = get_dataloaders(cfg["data"])   # e.g., dict: {"train":..., "val":..., "test_hard":...}
    # loader = loaders[split]
    # model = build_model(cfg["model"]).to(device)
    #
    # # Load checkpoint
    # ckpt_path = Path(args.ckpt) if args.ckpt else None
    # if ckpt_path is None:
    #     # User can pass explicit path, otherwise you’ll pick from a known run
    #     # For now, just demonstrate loading from a path they provide
    #     raise ValueError("Provide --ckpt path to a checkpoint (e.g., runs/<run>/checkpoints/best.pt)")
    #
    # ckpt = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(ckpt["model"])
    #
    # y_true, y_pred = predict_all(model, loader, device)
    # cm = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(cfg["data"]["num_classes"])))
    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(cfg["data"]["num_classes"])))
    # disp.plot(values_format="d")
    # plt.title(f"Confusion Matrix ({split})")
    # out_path = paths.fig_dir / f"confusion_matrix_{split}.png"
    # plt.savefig(out_path, dpi=200, bbox_inches="tight")
    # plt.close()
    #
    # acc = (y_true == y_pred).float().mean().item()
    # print(f"Split={split}  Acc={acc:.4f}")
    # print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
