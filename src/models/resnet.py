import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes=10, in_channels=3, small_input=True, pretrained=False):
    """
    resnet18 but modified for tiny 28x28 images.
    the trick is to shrink conv1 and remove maxpool so we don't lose spatial info too fast.
    learned this from cifar resnet implementations.
    """
    model = resnet18(weights="DEFAULT" if pretrained else None)
    
    if small_input:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old.out_channels, 
                                kernel_size=old.kernel_size, stride=old.stride, 
                                padding=old.padding, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
