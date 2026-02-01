# The Lazy Artist (PreCog CV Task)

This repo implements the PreCog CV task around **spurious correlations** using a Colored-MNIST setup:
- Generate Colored-MNIST with strong digit↔color correlation (train) and inverted/randomized correlation (hard test)
- Train a baseline CNN that “cheats” via color
- Evaluate generalization gap + confusion matrix
- (Later) interpretability (ActMax, Grad-CAM) + interventions

## Repo structure

- `configs/` : YAML configs
- `src/` :
  - `data/` : dataset code
  - `models/` : model definitions (to be added)
  - `train.py` : training entrypoint
  - `eval.py` : evaluation entrypoint
  - `utils/` : seeding + logging
- `notebooks/` : experiment notebooks (required for submission)
- `runs/` : outputs (logs, checkpoints, figures) - not committed

---

## Setup

### Option A: pip + venv (recommended quick start)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

## Task 0: Colored-MNIST dataset

Generate the biased dataset (Easy train/val + Hard test) with textured foreground coloring:
```bash
python scripts/generate_colored_mnist.py --root data/colored_mnist --correlation-train 0.95 --test-mode inverted
```

Verify the correlation statistics:
```bash
python scripts/verify_colored_mnist.py --root data/colored_mnist --expected-train 0.95 --expected-val 0.95
```

Visualize a sample grid:
```bash
python scripts/visualize_colored_mnist.py --root data/colored_mnist --split train --n 36 --out runs/colored_mnist_train.png
```
