# src/train.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything, SeedConfig
from src.utils.logging import make_run_dir, save_config, log_jsonl

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
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    crit = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    t = cfg["train"]
    lr = float(t["lr"])
    wd = float(t.get("weight_decay", 0.0))
    name = t.get("optimizer", "adamw").lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Seed
    seed_everything(SeedConfig(seed=int(cfg.get("seed", 42)), deterministic=False))

    # Run folder
    run_name = cfg["run"]["name"]
    out_dir = cfg["run"].get("out_dir", "runs")
    paths = make_run_dir(out_dir, run_name)
    save_config(cfg, paths.config_copy)

    device = pick_device(cfg.get("device", "auto"))

    # -----------------------------
    # TODO: plug in your dataset + model
    # -----------------------------
    raise NotImplementedError(
        "Next steps:\n"
        "1) Implement get_dataloaders in src/data/datasets.py\n"
        "2) Implement build_model in src/models/registry.py (or directly import your model)\n"
        "Then replace this NotImplementedError block."
    )

    # Example expected code after you implement:
    #
    # train_loader, val_loader = get_dataloaders(cfg["data"])
    # model = build_model(cfg["model"]).to(device)
    #
    # opt = build_optimizer(cfg, model)
    # best_val_acc = -1.0
    #
    # for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
    #     tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
    #     va_loss, va_acc = evaluate(model, val_loader, device)
    #
    #     record = {
    #         "epoch": epoch,
    #         "train_loss": tr_loss,
    #         "train_acc": tr_acc,
    #         "val_loss": va_loss,
    #         "val_acc": va_acc,
    #         "device": str(device),
    #     }
    #     log_jsonl(paths.log_jsonl, record)
    #     print(record)
    #
    #     if va_acc > best_val_acc:
    #         best_val_acc = va_acc
    #         ckpt_path = paths.ckpt_dir / "best.pt"
    #         torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    #
    # # Save last
    # torch.save({"model": model.state_dict(), "config": cfg}, paths.ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
