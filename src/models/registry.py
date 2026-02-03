from src.models.cnn import SimpleCNN


def build_model(cfg):
    name = cfg.get("name", "simple_cnn").lower()
    
    if name in ("simple_cnn", "cnn"):
        return SimpleCNN(
            in_channels=cfg.get("in_channels", 3),
            num_classes=cfg.get("num_classes", 10),
            width=cfg.get("width", 6),
        )
    
    raise ValueError(f"unknown model: {name}")
