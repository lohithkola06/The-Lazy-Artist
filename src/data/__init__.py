# src/data/__init__.py
# re-export public API

from src.data.colored_mnist import (
    generate_colored_mnist,
    load_meta,
    get_color_palette,
    get_color_names,
    make_dominant_color_map,
    compute_overall_correlation,
    compute_empirical_correlation,
    colorize_with_background,
    PALETTE,
)

from src.data.datasets import (
    ColoredMNISTDataset,
    get_dataloaders,
    load_split,
)
