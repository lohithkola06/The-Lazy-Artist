# The Lazy Artist

> **Can a CNN learn to cheat — and can we catch it?**

This project investigates **spurious correlations** in deep learning using a custom Colored-MNIST dataset. We deliberately plant a color shortcut (each digit is shown on a predictable background color 95% of the time), train a CNN that exploits it, visualize the cheating with Grad-CAM, attempt to fix it with training-time interventions, and finally test adversarial robustness of both the "lazy" and "robust" models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Generation](#dataset-generation)
- [Running the Notebooks](#running-the-notebooks)
- [Approach & Key Results](#approach--key-results)
- [Dependency Libraries](#dependency-libraries)
- [Reproducibility](#reproducibility)

---

## Project Overview

| Task                       | Notebook                      | Question                                                                                                  |
| -------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| **0 — Dataset**            | `00_dataset.ipynb`            | Build a biased Colored-MNIST with 95% digit↔color correlation (train/val) and 0% correlation (test-hard). |
| **1 — Baseline**           | `01_baseline.ipynb`           | Train a small CNN. Does it learn shape or cheat via color?                                                |
| **3 — Grad-CAM**           | `03_gradcam.ipynb`            | Visualize _where_ the model looks — digit strokes or background?                                          |
| **4 — Interventions**      | `04_interventions.ipynb`      | Fix the cheating with Color-Jitter Consistency and a Color-Adversarial Head (GRL).                        |
| **5 — Adversarial Attack** | `05_adversarial_attack.ipynb` | Craft targeted PGD perturbations (7 → 3). Is the "robust" model harder to fool?                           |

---

## Directory Structure

```
The-Lazy-Artist/
├── configs/
│   ├── baseline.yaml          # Training hyperparameters & data config
│   └── environment.yml        # Conda environment spec
├── data/
│   └── colored_mnist/         # Generated dataset (.pt files + meta.json)
│       ├── train.pt
│       ├── val.pt
│       ├── test_hard.pt
│       └── meta.json
├── notebooks/
│   ├── 00_dataset.ipynb       # Task 0: dataset creation & exploration
│   ├── 01_baseline.ipynb      # Task 1: train baseline CNN
│   ├── 03_gradcam.ipynb       # Task 3: Grad-CAM interpretability
│   ├── 04_interventions.ipynb # Task 4: de-biasing interventions
│   └── 05_adversarial_attack.ipynb  # Task 5: targeted adversarial attacks
├── outputs/
│   ├── baseline_model.pt              # Lazy baseline checkpoint
│   ├── best_intervention_model.pt     # Best robust model checkpoint
│   ├── intervention_experiments.json  # Task 4 experiment log
│   └── adversarial_attack_log.json    # Task 5 experiment log
├── src/
│   ├── __init__.py
│   ├── train.py               # Training loop (CLI entrypoint)
│   ├── eval.py                # Evaluation + confusion matrix (CLI entrypoint)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── colored_mnist.py   # Dataset generation (palette, colorization, correlation)
│   │   └── datasets.py        # PyTorch Dataset/DataLoader wrappers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py             # SimpleCNN architecture
│   │   └── registry.py        # Model factory (build_model)
│   └── utils/
│       ├── logging.py         # Run directory management, JSONL logging
│       └── seed.py            # Deterministic seeding (torch, numpy, random)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Option A — pip + venv (recommended)

```bash
git clone <repo-url> && cd The-Lazy-Artist

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f configs/environment.yml
conda activate lazy-artist
```

---

## Dataset Generation

The dataset is already included in `data/colored_mnist/`. To regenerate from scratch:

```bash
# From the project root
python -c "
from src.data.colored_mnist import generate_colored_mnist
generate_colored_mnist('data/colored_mnist', seed=42, corr=0.95)
"
```

This creates three splits:

| Split          | Samples | Color Correlation  | Purpose                          |
| -------------- | ------- | ------------------ | -------------------------------- |
| `train.pt`     | 54,000  | 95% dominant color | Training with planted shortcut   |
| `val.pt`       | 6,000   | 95% dominant color | Validation (same bias as train)  |
| `test_hard.pt` | 10,000  | 0% dominant color  | Evaluation with shortcut removed |

**Design choices:**

- **10-color palette** mapped 1:1 to digits (digit 0 → red bg, digit 1 → green bg, etc.)
- **White digits** on textured colored backgrounds — the color shortcut lives entirely in the background
- **Brightness-only noise** (σ = 0.15) for texture — same hue per pixel, different brightness

---

## Running the Notebooks

All experiments live in Jupyter notebooks. Run them in order:

```bash
# Start Jupyter
source .venv/bin/activate
jupyter notebook notebooks/
```

| Notebook                      | What it does                                                                      | Key output                                                   |
| ----------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `00_dataset.ipynb`            | Generates & visualizes the biased dataset, verifies correlation stats             | Sample grids, correlation tables                             |
| `01_baseline.ipynb`           | Trains a SimpleCNN for 10 epochs, evaluates on val & test-hard                    | `outputs/baseline_model.pt`, accuracy tables                 |
| `03_gradcam.ipynb`            | Runs Grad-CAM on correctly/incorrectly classified images                          | Heatmap visualizations showing where the model attends       |
| `04_interventions.ipynb`      | Trains models with color-jitter and adversarial head, compares test-hard accuracy | `outputs/best_intervention_model.pt`, comparison tables      |
| `05_adversarial_attack.ipynb` | PGD targeted attack (7 → 3, ε = 0.05), epsilon sweep, multi-image stats           | Attack visualizations, `outputs/adversarial_attack_log.json` |

### CLI entrypoints (alternative to notebooks)

```bash
# Train baseline
python -m src.train --config configs/baseline.yaml

# Evaluate a checkpoint
python -m src.eval --config configs/baseline.yaml --ckpt outputs/baseline_model.pt --split test_hard
```

---

## Approach & Key Results

### Task 0 — Dataset

Built a Colored-MNIST variant where 95% of training images of digit _d_ appear on color _d_'s background. The remaining 5% get a random other color. Test-hard uses only non-dominant colors, completely breaking the shortcut.

### Task 1 — Baseline ("The Lazy Artist")

Trained a **SimpleCNN** (3 conv layers, width=4, **4,486 parameters**) for 10 epochs with AdamW.

| Split                   | Accuracy  |
| ----------------------- | --------- |
| Val (biased)            | **95.5%** |
| Test-hard (no shortcut) | **22.4%** |

The 73-point gap proves the model learned color, not shape.

### Task 3 — Grad-CAM

Grad-CAM heatmaps on the final conv layer confirm the model attends to the **background** (color regions), not the digit strokes. On test-hard images where color misleads, the model confidently predicts the wrong digit — the one matching the background color.

### Task 4 — Interventions

Two training-time interventions to break color reliance without changing the dataset:

1. **Color-Jitter Consistency** — apply random color jitter to each image and penalize any change in the model's output distribution (KL divergence). Forces the model to be invariant to color.
2. **Color-Adversarial Head (GRL)** — attach a color-prediction head with a Gradient Reversal Layer. The main model learns to _strip_ color information from its representation.

| Method                 | Val Acc | Test-Hard Acc |
| ---------------------- | ------- | ------------- |
| Baseline (lazy)        | 95.5%   | 22.4%         |
| Color-Jitter (α = 2.0) | 96.0%   | **61.3%**     |
| Adversarial (γ = 0.1)  | 94.1%   | 38.6%         |
| Combined               | 95.2%   | 61.3%         |

**Best model:** Color-Jitter Consistency alone (α = 2.0) → saved as `outputs/best_intervention_model.pt`.

### Task 5 — Adversarial Attack ("The Invisible Cloak")

Targeted PGD attack: force a **7** to be classified as **3** with >90% confidence under an L∞ budget of ε = 0.05. Uses random start (true PGD, not iterative FGSM). Both models consume raw [0, 1] RGB tensors with no normalization.

| Metric                     | Lazy  | Robust    |
| -------------------------- | ----- | --------- |
| Success at ε = 0.05        | ✅    | ✅        |
| Steps to 90% confidence    | 109   | **79**    |
| Multi-image success (N=20) | 70%   | **95%**   |
| Min ε (binary search)      | 0.051 | **0.048** |

**Surprising finding:** the robust model is _easier_ to fool adversarially. Distribution-shift robustness ≠ adversarial robustness. Color-jitter training smoothed the loss landscape, making gradients more predictable — exactly what PGD exploits.

---

## Dependency Libraries

| Library                 | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| **PyTorch**             | Neural network training, tensor operations, autograd |
| **torchvision**         | MNIST download, image transforms                     |
| **NumPy**               | Array operations, random number generation           |
| **Matplotlib**          | All visualizations (sample grids, heatmaps, charts)  |
| **scikit-learn**        | Confusion matrices                                   |
| **PyYAML**              | Config file parsing                                  |
| **Jupyter / ipykernel** | Interactive notebook execution                       |

Full pinned versions are in `requirements.txt`.

---

## Reproducibility

All experiments use **seed = 42** with deterministic PyTorch settings via `src/utils/seed.py`. Results should be identical across runs on the same hardware.

### Model Architecture

**SimpleCNN**: 3 conv layers (width=4) → flatten → linear classifier. **4,486 total parameters.** Deliberately kept small so it's forced to take shortcuts.

```
Conv2d(3, 4, 3) → ReLU → MaxPool2d(2)
Conv2d(4, 4, 3) → ReLU → MaxPool2d(2)
Conv2d(4, 8, 3) → ReLU
Flatten → Linear(392, 10)
```
