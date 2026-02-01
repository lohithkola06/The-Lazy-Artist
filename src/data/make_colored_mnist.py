#!/usr/bin/env python3
# CLI to generate colored-mnist dataset
# usage: python -m src.data.make_colored_mnist --out data/colored_mnist

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import seed_everything, SeedConfig
from src.data.colored_mnist import generate_colored_mnist


def main():
    p = argparse.ArgumentParser(description="generate colored-mnist dataset")
    p.add_argument("--out", "-o", default="data/colored_mnist")
    p.add_argument("--corr", type=float, default=0.95)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_mode", choices=["inverted", "random"], default="inverted")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    seed_everything(SeedConfig(seed=args.seed))
    generate_colored_mnist(args.out, args.seed, args.corr, args.val_frac, args.test_mode)


if __name__ == "__main__":
    main()
