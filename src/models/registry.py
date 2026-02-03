from src.models.cnn import SimpleCNN
from src.models.resnet import build_resnet18


def build_model(cfg):
    """factory function so i don't have to change train.py when switching models"""
    name = cfg.get("name", "simple_cnn").lower()
    num_classes = cfg.get("num_classes", 10)
    in_channels = cfg.get("in_channels", 3)
    
    if name in ("simple_cnn", "cnn"):
        return SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            width=cfg.get("width", 6),  # minimal: 4→8→16 filters
        )
    
    if name in ("resnet18", "resnet-18", "resnet"):
        return build_resnet18(
            num_classes=num_classes,
            in_channels=in_channels,
            small_input=cfg.get("small_input", True),
            pretrained=cfg.get("pretrained", False),
        )
    
    raise ValueError(f"unknown model: {name}")
