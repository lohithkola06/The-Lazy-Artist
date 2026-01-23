# src/utils/logging.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    fig_dir: Path
    log_jsonl: Path
    config_copy: Path


def make_run_dir(out_dir: str, run_name: str) -> RunPaths:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{run_name}_{ts}"

    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        fig_dir=fig_dir,
        log_jsonl=run_dir / "train_log.jsonl",
        config_copy=run_dir / "config.yaml",
    )


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    record = dict(record)
    record["_time"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
